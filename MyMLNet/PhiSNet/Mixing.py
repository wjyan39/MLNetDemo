import torch 
import torch.nn as nn
import numpy as np

from BasicUtility.O3.O3Tensor import SphericalTensor, O3Tensor
from BasicUtility.O3.O3Utility import CGCoupler, CGPCoupler, get_rsh_cg_coefficients

class SelfMixing(nn.Module):
    """
    Mixes features of different L orders within a given SphericalTensor
    Args:
        metadata ::torch.LongTensor:: The metadata of the input SphericalTensor.
        order_out ::int:: The maximum output L order required during tensor product coupling.
        group ::string:: 'so3' for SphericalTensor, 'o3' for O3Tensor
    """
    def __init__(self, metadata:torch.LongTensor, order_out, group="so3"):
        super().__init__()
        assert metadata.dim() == 1
        self.metadata_in = metadata 
        self.order_out = order_out
        if order_out < metadata.shape[0]:
            self.order_out =  metadata.shape[0]
        group = group.lower()
        if group == "so3":
            self.CGCoupler = CGCoupler(metadata, metadata, max_l=self.order_out, overlap_out=False, trunc_in=False)
            self.tensor_class = SphericalTensor
        elif group == "o3":
            self.CGCoupler = CGPCoupler(metadata, metadata, max_l=self.order_out, overlap_out=False, trunc_in=False)
            self.tensor_class = O3Tensor
        else:
            raise NotImplementedError
        self.group = group 
        num_channels_in = torch.sum(metadata).item() 
        num_channels_out = torch.sum(self.CGCoupler.metadata_out).item() 
        self.register_parameter("keep_coeff", torch.nn.Parameter(torch.Tensor(num_channels_in)))
        self.register_parameter("mix_coeff", torch.nn.Parameter(torch.Tensor(num_channels_out)))
        self._init_parameters()

    def _init_parameters(self):
        torch.nn.init.uniform_(self.keep_coeff, a=-np.sqrt(3), b=np.sqrt(3))
        torch.nn.init.uniform_(self.mix_coeff, a=-np.sqrt(3), b=np.sqrt(3))  
        # determine metadata_out 
        metadata_out = torch.zeros_like(self.CGCoupler.metadata_out)
        for (lpout, _, _, degeneracy) in self.CGCoupler.valid_coupling_ids:
            metadata_out[lpout] = max(metadata_out[lpout], degeneracy)
        # generate the flatenned idx for input, tensor product output and final coupled output
        tmp_zeros = torch.zeros(metadata_out.shape[0] - self.metadata_in.shape[0]).long()
        metadata_in = torch.cat([self.metadata_in, tmp_zeros], dim=0)
        metadata_tmp = torch.stack([metadata_in, metadata_out], dim=0)
        if self.group == "so3":
            stepwise = 1
            n_irreps_per_l =  torch.arange(start=0, end=metadata_out.shape[0]) * 2 + 1 
        elif self.group == "o3":
            stepwise = 2 
            n_irreps_per_l =  torch.arange(start=0, end=metadata_out.shape[0]//2) * 2 + 1
            n_irreps_per_l = n_irreps_per_l.repeat_interleave(2)
        else:
            raise NotImplementedError
        repid_offsets = torch.cumsum(
            metadata_tmp * n_irreps_per_l.unsqueeze(0), dim=1
        )
        repid_offsets = torch.cat(
            [torch.LongTensor([[0], [0]]), repid_offsets[:, :-1]], dim=1
        ).long() 
        # generate channels for tensor product output
        repids_tp_out = []
        tmp = [[] for _ in range(metadata_out.shape[0])]
        ## loop over each valid coupling (l1, l2, l) to decide the output index
        for (lpout, _, _, degeneracy) in self.CGCoupler.valid_coupling_ids:
            l_degeneracy = 2 * (lpout // stepwise) + 1
            ls_segement = torch.arange(l_degeneracy).repeat_interleave(degeneracy)
            ns_segement = torch.arange(degeneracy).repeat(l_degeneracy)
            repids_tp_out_3j = (
                repid_offsets[1, lpout]
                + ls_segement * metadata_out[lpout]
                + ns_segement
            ).view(l_degeneracy, -1)
            tmp[lpout].append(repids_tp_out_3j) 
        for tmp_list in tmp:
            repids_tp_out.append(torch.cat(tmp_list, dim=1).view(-1))
        # generate channels for final sum up  
        metadata_cp = torch.minimum(self.metadata_in, metadata_out[:self.metadata_in.shape[0]])
        repids_in, repids_out = [], []
        for cur_lp in range(self.metadata_in.shape[0]):
            l_degeneracy = 2 * (cur_lp // stepwise) + 1
            ls_segement = torch.arange(l_degeneracy).repeat_interleave(metadata_cp[cur_lp]) 
            ns_segement = torch.arange(metadata_cp[cur_lp]).repeat(l_degeneracy) 
            repids_in_3j = (
                repid_offsets[0, cur_lp]
				+ ls_segement * metadata_tmp[0, cur_lp]
				+ ns_segement
            )
            repids_out_3j = (
                repid_offsets[1, cur_lp]
				+ ls_segement * metadata_tmp[1, cur_lp]
				+ ns_segement
			)
            repids_in.append(repids_in_3j)
            repids_out.append(repids_out_3j) 
        self.register_buffer("repids_in", torch.cat(repids_in).long())
        self.register_buffer("repids_out", torch.cat(repids_out).long())
        self.register_buffer("metadata_out", metadata_out)
        self.register_buffer("repids_tp_out", torch.cat(repids_tp_out).long())
        self.register_buffer("out_layout", self.tensor_class.generate_rep_layout_1d_(metadata_out))

    def forward(self, x):
        """
        Rules:
            k_{l_3} \odot x^{(l_3)} + \sum_{l_1} \sum_{l_2} s_{l_3, l_2, l_1} \odot (x^{(l_1)} \ocross x^{(l_2)})
        """
        assert len(x.rep_dims) == 1
        assert torch.all(x.metadata[0].eq(self.metadata_in))
        coupling_dim = x.rep_dims[0]
        
        # k_{l_3} \odot x^{(l_3)}, as (1)
        tmp_x = x 
        broadcast_shape_k = tuple(
            self.keep_coeff.shape[0] if d == coupling_dim else 1 for d in range(x.ten.dim())
        )
        tmp_x = tmp_x.scalar_mul(self.keep_coeff.view(broadcast_shape_k))
        
        # s_{l_3, l_2, l_1} \odot (x^{(l_1)} \ocross x^{(l_2)}) as (2)
        cat_tp_out = self.CGCoupler(x, x) 
        cat_tp_out.ten = cat_tp_out.ten * 0.5
        broadcast_shape_m = tuple(
            self.mix_coeff.shape[0] if d == coupling_dim else 1 for d in range(x.ten.dim())
        )
        cat_tp_out = cat_tp_out.scalar_mul(self.mix_coeff.view(broadcast_shape_m))
        # \sum_{l_1} \sum_{l_2} {(2)} as (3)
        out_shape = tuple(
            self.out_layout.shape[1] if d == coupling_dim else x.ten.shape[d] for d in range(x.ten.dim())
        )
        out_tp_ten = torch.zeros(
            out_shape, dtype=x.ten.dtype, device=x.device
        ).index_add_(
            coupling_dim, self.repids_tp_out, cat_tp_out.ten
        )
        # (1) + (3)
        out_tp_ten.index_select(
            dim=coupling_dim, index=self.repids_out
        ).add_(
            tmp_x.ten.index_select(
                dim=coupling_dim, index=self.repids_in
            )
        )
        
        return self.tensor_class(
            data_ten=out_tp_ten,
            rep_dims=(coupling_dim,),
            metadata=self.metadata_out.unsqueeze(0)
        )
        

class PairMixing(torch.nn.Module):
    def __init__(self, envelop, filter_channels:int, metadata_1:torch.LongTensor, metadata_2:torch.LongTensor, order_out:int, group="so3"):
        super().__init__()
        assert metadata_1.dim() == 1
        assert metadata_2.dim() == 1 
        # pair mixing occurs between 2 node spherical tensors in the same layer,
        # so ensure that they have same layout. 
        assert metadata_1.shape[0] == metadata_2.shape[0]
        self.order_out = max(order_out, metadata_1.shape[0])
        group = group.lower()
        if group == "so3":
            self.CGCoupler = CGCoupler(metadata_1, metadata_2, max_l=self.order_out, overlap_out=False, trunc_in=False)
            self.tensor_class = SphericalTensor
        elif group == "o3":
            self.CGCoupler = CGPCoupler(metadata_1, metadata_2, max_l=self.order_out, overlap_out=False, trunc_in=False)
            self.tensor_class = O3Tensor
        else:
            raise NotImplementedError
        self.envelop = envelop # polynomial expansions of radial function
        self.filter_channels = filter_channels
        self.metadata_in = torch.stack([metadata_1, metadata_2], dim=0)
        self._init_params()
    
    def _init_params(self):
        valid_coupling_ids = self.CGCoupler.valid_coupling_ids
        metadata_out = torch.zeros_like(self.CGCoupler.metadata_out)
        if self.group == "so3":
            n_irreps_per_l = torch.arange(start=0, end=metadata_out.shape[0]) * 2 + 1
            stepwise = 1 
        elif self.group == "o3":
            n_irreps_per_l = torch.arange(start=0, end=metadata_out.shape[0]//2) * 2 + 1
            n_irreps_per_l = n_irreps_per_l.repeat_interleave(2)
            stepwise = 2 
        # get the metadata of the overlap reduction 
        cur_l_out = -1 
        for (lout, _, _, degeneracy) in valid_coupling_ids:
            if lout > cur_l_out:
                cur_l_out = lout
            if degeneracy > metadata_out[cur_l_out]:
                metadata_out[cur_l_out] = degeneracy
            
        # generate flatten indexing for various tensor
        max_n_out = metadata_out.max().item()
        repid_offsets_out = torch.cumsum(metadata_out * n_irreps_per_l, dim=0)
        repid_offsets_out = torch.cat(
            [torch.LongTensor([0]), repid_offsets_out[:-1]], dim=0
        )
        repids_out, matrix_select_idx, matid_to_fanin = [], [], []
        out_reduce_idx, out_reduce_mask = [], []
        mat_offset, dst_offset = 0, 0 
        for (lpout, lpin1, lpin2, degeneracy) in valid_coupling_ids:
            lin1, lin2, lout = lpin1//stepwise, lpin2//stepwise, lpout//stepwise 
            cg_source = get_rsh_cg_coefficients(lin1, lin2, lout) 
            cg_segment = cg_source.repeat_interleave(degeneracy, dim=1)
            ns_segment = torch.arange(degeneracy).repeat(cg_source.shape[1]) 
            # Calculating the representation IDs for the coupling tensors
            repids_out_3j = (
                repid_offsets_out[lpout]
                + (cg_segment[2] + lout) * metadata_out[lpout]
                + ns_segment
            )
            repids_out.append(repids_out_3j)

            matrix_select_idx.append(
                (torch.arange(1, dtype=torch.long) + mat_offset).repeat(n_irreps_per_l[lpout])
            )
            matid_to_fanin.append(
                torch.full((1,), self.filter_channels, dtype=torch.long)
            )
            mat_offset += 1

            for m in range(n_irreps_per_l[lpout]):
                dst_idx_lpm = (
                    dst_offset 
                    + torch.arange(max_n_out, dtype=torch.long) 
                ).contiguous 
                dst_mask_lpm = torch.zeros(
                    max_n_out, dtype=torch.long
                )
                dst_mask_lpm[:degeneracy] = 1
                out_reduce_idx.append(dst_idx_lpm)
                out_reduce_mask.append(dst_mask_lpm)
                dst_offset += degeneracy

        self.register_buffer("metadata_out", metadata_out)
        self.register_buffer("repids_cp_out", torch.cat(repids_out).long())
        self.register_buffer("matrix_select_idx", torch.cat(matrix_select_idx))
        self.register_buffer("out_reduce_mask", torch.cat(out_reduce_mask).bool())
        self.register_buffer("out_reduce_idx", torch.cat(out_reduce_idx)[self.out_reduce_mask])
        self.matid_to_fanin = torch.cat(matid_to_fanin)
        self.n_mats = mat_offset
        self.n_gathered_mats = self.matrix_select_idx.shape[0]
        self.linears = torch.nn.Parameter(
            torch.zeros(self.n_mats, self.filter_channels, max_n_out)
        )
        self._reset_params()
    
    def _reset_params(self):
        with torch.no_grad():
            for lid, fan_in in enumerate(self.matid_to_fanin):
                bound = torch.sqrt(1 / fan_in)
                self.linears.data[lid].uniform_(-bound, bound) 
    
    def forward(self, x, y, r:torch.Tensor):
        assert len(x.rep_dims) == 1
        assert len(y.rep_dims) == 1 
        assert x.rep_dims[0] == y.rep_dims[0]
        assert x.rep_dims[0] == x.ten.dim() - 1 
        assert torch.all(x.metadata[0].eq(self.metadata_in[0]))
        assert torch.all(y.metadata[0].eq(self.metadata_in[1]))
        in_r = r.view(-1, r.shape[-1])
        # generate polynomial expansion of r
        # shape is (-1, filter_channels)
        geom_filter:torch.Tensor = self.envelop(in_r, self.filter_channels)
        # 
        padded_in_filter = torch.index_select(
            geom_filter, dim=1, index=torch.arange(self.filter_channels).repeat(self.n_gathered_mats)
        )
        padded_in_filter = padded_in_filter.view(
            in_r.shape[0], self.n_gathered_mats, self.filter_channels
        ).transpose(0, 1)
        gathered_linears = torch.index_select(
            self.linears, dim=0, index=self.matrix_select_idx
        )
        padded_out_ten = (
            torch.bmm(padded_in_filter, gathered_linears)
            .transpose(0, 1).contiguous()
            .view(in_r.shape[0], -1)
        )
        out_ten = torch.zeros(
            in_r.shape[0], self.CGCoupler.out_layout.shape[1], dtype=in_r.dtype, device=in_r.device
        ).index_add_(1, self.out_reduce_idx, padded_out_ten[:, self.out_reduce_mask]) 
        out_ten = out_ten.view(*x.ten.shape[:-1], -1)
        # 
        coupling_dim = x.rep_dims[0]
        cat_tp_out = self.CGCoupler(x, y)
        cat_tp_out.ten = cat_tp_out.ten * out_ten

        out_shape = tuple(
            self.out_layout.shape[1] if d == coupling_dim else x.ten.shape[d] for d in range(x.ten.dim())
        )
        out_tp_ten = torch.zeros(
            out_shape, dtype=x.dtype, device=x.device
        ).index_add_(
            coupling_dim, self.repids_cp_out, cat_tp_out
        )

        return self.tensor_class(
            data_ten=out_tp_ten,
            rep_dims=(coupling_dim,),
            metadata=self.metadata_out
        )

if __name__ == "__main__":
    metadata_x = torch.LongTensor([3, 3, 1])
    metadata_fix = torch.LongTensor([[3, 3, 1]])
    x_ten = torch.randn(4, 6, 17, dtype=torch.double) 
    x = SphericalTensor(x_ten, (2,), metadata_fix)
    selfMixer = SelfMixing(metadata_x, order_out=3) 
    y = selfMixer(x) 
    print(y.shape) 
    pass   




