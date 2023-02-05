import torch
import numpy as np  
import math 
from typing import Tuple
from torch.nn import Embedding

from BasicUtility.O3.O3Tensor import SphericalTensor, O3Tensor
from BasicUtility.O3.O3Utility import SumsqrContraction1d, NormContraction1d
from BasicUtility.O3.O3Utility import CGCoupler, CGPCoupler, get_rsh_cg_coefficients

class IELin(torch.nn.Module):
    r"""
    Irreducible Representation wise Linear Layer.
    This module takes in a spherical tensor and perform linear transformation among
    the feature channels spanned by each irreducible representation index, (l(p), m).
    ..math::
        \mathbf{h}^{\mathrm{out}}_{l,m} = \mathbf{W}^{l} \cdot \mathbf{h}^{\mathrm{in}}_{l,m} 
    Matrix multiplication always takes place on the last dimension of the spherical tensors.
    """
    def __init__(self, metadata_in, metadata_out, group="o3", maxpadding=True):
        super().__init__()
        assert metadata_in.dim() == 1 
        assert len(metadata_in) == len(metadata_out) 
        self._metadata_in = metadata_in 
        self._metadata_out = metadata_out 
        group = group.lower() 
        if group == "so3":
            self.tensor_class = SphericalTensor
            self.n_irreps_per_l = torch.arange(start=0, end=metadata_in.size(0)) * 2 + 1 
        elif group == "o3":
            self.tensor_class = O3Tensor 
            n_irreps_per_l = torch.arange(start=0, end=metadata_in.size(0)//2) * 2 + 1
            self.n_irreps_per_l = n_irreps_per_l.repeat_interleave(2)
        else:
            raise NotImplementedError(f"The declaring group {group} has not been supported yet.") 

        if maxpadding:
            padding_size_in = metadata_in.max().item() 
            padding_size_out = metadata_out.max().item() 
        else:
            raise NotImplementedError 
        
        vector_padding_idx = []
        vector_padding_mask = []
        src_offset = 0 
        for lp_idx, nin_lpm in enumerate(metadata_in):
            nin_lpm = nin_lpm.item() 
            nout_lpm = metadata_out[lp_idx].item() 
            # jump the loop with no input feature channels
            if nin_lpm == 0:
                continue 
            bmv_degeneracy_lpm = -(nout_lpm // (-padding_size_out)) 
            # jump the loop with no output feature channels, also skip the associated input channels
            if bmv_degeneracy_lpm == 0:
                src_offset += nin_lpm * self.n_irreps_per_l[lp_idx] 
                continue 
            for m in range(self.n_irreps_per_l[lp_idx]):
                # feature channels in this lpm 
                vector_padding_idx_lpm = torch.zeros(-(nin_lpm // (-padding_size_in)) * padding_size_in, dtype=torch.long) 
                vector_padding_mask_lpm = torch.zeros(-(nin_lpm // (-padding_size_in)) * padding_size_in, dtype=torch.long) 
                # select the current feature channnels
                vector_padding_idx_lpm[:nin_lpm] = (torch.arange(nin_lpm, dtype=torch.long) + src_offset) 
                vector_padding_mask_lpm[:nin_lpm] = 1 
                # 
                vector_padding_idx.append(vector_padding_idx_lpm.repeat(bmv_degeneracy_lpm)) 
                vector_padding_mask.append(vector_padding_mask_lpm.repeat(bmv_degeneracy_lpm)) 
                src_offset += nin_lpm
        # register buffers 
        self.register_buffer("vector_padding_idx", torch.cat(vector_padding_idx)) 
        self.register_buffer("vector_padding_mask", torch.cat(vector_padding_mask).bool())
        
        # generate indexes for linear layers, parameters are shared among all m = 2l+1 channels for each l.
        matrix_select_idx = [] 
        matid_to_fanin = []
        mat_offset = 0 
        for lp_idx, nin_lpm in enumerate(metadata_in):
            nin_lpm = nin_lpm.item() 
            nout_lpm = metadata_out[lp_idx].item() 
            nblocks_lp = (-(nin_lpm // (-padding_size_in))) * (-(nout_lpm // (-padding_size_out))) 
            if nblocks_lp > 0:
                # determine the index of linear layer acting on a given lp representation with m-order degeneracy
                matrix_select_idx.append(
                    (torch.arange(nblocks_lp, dtype=torch.long) + mat_offset).repeat(self.n_irreps_per_l[lp_idx])
                )
                # logging the feature channels for each linear layer 
                matid_to_fanin.append(
                    torch.full((nblocks_lp,), nin_lpm, dtype=torch.long) 
                )
                mat_offset += nblocks_lp 
        # register buffers
        self.register_buffer("matrix_select_idx", torch.cat(matrix_select_idx)) 
        self.matid_to_fanin = torch.cat(matid_to_fanin) 
        self.n_mats = mat_offset 
        self.n_gathered_mats = self.matrix_select_idx.shape[0] 

        # generate output indexes where those matrix productions shall reduce to via summation 
        out_reduce_idx = []
        out_reduce_mask = [] 
        dst_offset = 0 
        for lp_idx, nout_lpm in enumerate(metadata_out):
            nin_lpm = metadata_in[lp_idx].item() 
            nout_lpm = nout_lpm.item() 
            # go pass the undesired lp irrep 
            if nout_lpm == 0:
                continue 
            bmv_degeneracy_lpm = -(nout_lpm // (-padding_size_out)) 
            in_j_degeneracy_lpm = -(nin_lpm // (-padding_size_in)) 
            if in_j_degeneracy_lpm == 0:
                dst_offset += nout_lpm * self.n_irreps_per_l[lp_idx] 
                continue 
            for m in range(self.n_irreps_per_l[lp_idx]):
                dst_idx_lpm = (
                    (dst_offset + torch.arange(padding_size_out * bmv_degeneracy_lpm, dtype=torch.long)) 
                    .view(bmv_degeneracy_lpm, 1, padding_size_out)
                    .expand(bmv_degeneracy_lpm, in_j_degeneracy_lpm, padding_size_out) 
                    .contiguous().view(-1)
                ) 
                dst_mask_lpm = torch.zeros(padding_size_out * bmv_degeneracy_lpm, dtype=torch.long) 
                dst_mask_lpm[:nout_lpm] = 1 
                dst_mask_lpm = (
                    dst_mask_lpm.view(bmv_degeneracy_lpm, 1, padding_size_out)
                    .expand(bmv_degeneracy_lpm, in_j_degeneracy_lpm, padding_size_out) 
                    .contiguous().view(-1)
                ) 
                out_reduce_idx.append(dst_idx_lpm) 
                out_reduce_mask.append(dst_mask_lpm) 
                dst_offset += nout_lpm 
        # register buffers 
        self.register_buffer("out_reduce_mask", torch.cat(out_reduce_mask).bool())
        self.register_buffer("out_reduce_idx", torch.cat(out_reduce_idx)[self.out_reduce_mask]) 

        # linear layers are maintained as matrices of parameters 
        self.linears = torch.nn.Parameter(
            torch.zeros(self.n_mats, padding_size_in, padding_size_out) 
        )
        self.padding_size_in = padding_size_in 
        self.padding_size_out = padding_size_out 

        self.register_buffer("out_layout", self.tensor_class.generate_rep_layout_1d_(self._metadata_out)) 
        self.num_out_channels = torch.sum(self._metadata_out).item() 
        self.reset_parameters() 
    
    def reset_parameters(self):
        """
        initialize parameters for each linear layer 
        """
        with torch.no_grad():
            for lid, fan_in in enumerate(self.matid_to_fanin):
                bound = torch.sqrt(1/fan_in)
                self.linears.data[lid].uniform_(-bound, bound) 

    def forward(self, x:SphericalTensor) -> SphericalTensor:
        """"""
        assert len(x.rep_dims) == 1 
        assert x.rep_dims[-1] == x.ten.dim() - 1 
        assert torch.all(x.metadata[-1].eq(self._metadata_in)), (
            f"Expected the SphericalTensor x and self._metadata_in to have the "
            f"same irrep metadata along the last dimension, got {x.metadata[-1]}"
            f" and {self._metadata_in} instead"
        )
        # generate input tensor via reshaping 
        # first to (-1, total_num_feature)
        in_ten = x.ten.view(-1, x.ten.shape[-1])
        # generate (nmat, padding_size_in) input tensors, other dimensions are left
        padded_in_ten = torch.index_select(
            in_ten, dim=1, index=self.vector_padding_idx
        ).mul_(self.vector_padding_mask) 
        padded_in_ten = padded_in_ten.view(
            x.ten.shape[0], self.n_gathered_mats, self.padding_size_in
        ).transpose(0, 1) # left n_mat to be the leading dimension 
        gathered_linears = torch.index_select(
            self.linears, dim=0, index=self.matrix_select_idx
        )
        # linear transfromation
        # first matrix multiplication
        padded_out_ten = (
            # (nmat, -1, padding_size_in) * (nmat, padding_size_in, padding_size_out)
            torch.bmm(padded_in_ten, gathered_linears)
            .transpose(0, 1) # shape (-1, nmat, padding_size_out)
            .contiguous()   
            .view(in_ten.shape[0],-1) 
        ) 
        out_ten = torch.zeros(
            in_ten.shape[0], self.out_layout.shape[1], dtype=in_ten.dtype, device=in_ten.device
        ).index_add_(
            1, self.out_reduce_idx, padded_out_ten[:, self.out_reduce_mask] 
        ) 
        out_ten = out_ten.view(*x.ten.shape[:-1], -1) 
        out_metadata = self._metadata_out.unsqueeze(0) 
        return self.tensor_class(
            out_ten,
            rep_dims=x.rep_dims,
            metadata=out_metadata,
            rep_layout=x.rep_layout[:-1] + (self.out_layout.data,),
            num_channels=x.num_channels[:-1] + (self.num_out_channels,)
        )


class IELinSerial(torch.nn.module):
    r"""
    Irrep-wise Equivariant Linear Layer. 
    This module stores linear layers for each lpm separately.
    """
    def __init__(self, metadata_in, metadata_out, group="o3"):
        super().__init__() 
        assert metadata_in.dim() == 1
        assert len(metadata_in) == len(metadata_out) 
        self._metadata_in = metadata_in
        self._metadata_out = metadata_out 
        group = group.lower()
        if group == "so3":
            self.tensor_class = SphericalTensor
            self.n_irreps_per_l = torch.arange(start=0, end=metadata_in.size(0)) * 2 + 1 
        elif group == "o3":
            self.tensor_class = O3Tensor 
            n_irreps_per_l = torch.arange(start=0, end=metadata_in.size(0)//2) * 2 + 1
            self.n_irreps_per_l = n_irreps_per_l.repeat_interleave(2)
        else:
            raise NotImplementedError(f"The declaring group {group} has not been supported yet.") 
        # Linear Layers for each lpm 
        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(metadata_in[lidx], metadata_out[lidx], bias=False)
                if self._metadata_out[lidx] > 0 and self._metadata_in[lidx] > 0 else None 
                for lidx, _ in enumerate(metadata_in)
            ]
        ) 
        self._end_inds_in = torch.cumsum(self._metadata_in * self.n_irreps_per_l, dim=0) 
        self._start_indx_in = torch.cat(torch.LongTensor([0]), self._end_inds_in[:-1]) 
        self.register_buffer("out_layout", self.tensor_class.generate_rep_layout_1d_(metadata_out)) 
        self.num_out_channels = torch.sum(self._metadata_out).item() 
        self.reset_parameters() 

    def reset_parameters(self):
        for linear in self.linears:
            if linear is None:
                continue 
            torch.nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), mode="fan_in") 

    def forward(self, x:SphericalTensor) -> SphericalTensor:
        assert x.rep_dims[-1] == x.ten.dim() - 1 
        assert torch.all(x.metadata[-1].eq(self._metadata_in)) 
        outs = [] 
        for l, linear_l in enumerate(self.linears):
            if linear_l is None:
                if self._metadata_out[l] > 0:
                    outs.append(
                        torch.zeros(
                            *x.shape[:-1], self._metadata_out[l] * self.n_irreps_per_l[l], 
                            dtype=x.ten.dtype, 
                            device=x.ten.device
                        )
                    ) 
                    continue 
            in_l = x.ten[..., self._start_indx_in[l]:self._end_inds_in[l]].contiguous() 
            out_l = linear_l(
                in_l.unflatten(-1, (self.n_irreps_per_l[l], self._metadata_in[l]))
            ) 
            outs.append(out_l.view(*x.shape[:-1], self._metadata_out[l] * self.n_irreps_per_l[l]))
        out_ten = torch.cat(outs, dim=-1) 
        out_metadata = x.metadata.clone()
        out_metadata[-1] = self._metadata_out 
        out_rep_layout = x.rep_layout[:-1] + (self.out_layout.data,) 
        return self.tensor_class(
            data_ten=out_ten,
            rep_dims=x.rep_dims,
            metadata=out_metadata,
            rep_layout=out_rep_layout,
            num_channels=x.num_channels[:-1] + (self.num_out_channels,)
        )


class EvNorm1d(torch.nn.Module):
    r"""
    Equivariant Normalization Layer.
    """
    def __init__(self, num_channels, n_invariant_channels=0):
        super().__init__()
        self._num_channels = num_channels 
        self._n_invariant_channels = n_invariant_channels 
    
    def forward(self, x:SphericalTensor) -> Tuple[torch.Tensor, SphericalTensor]:
        if self._n_invariant_channels == 0:
            divisor = (
                SumsqrContraction1d.apply(x.ten, x.rep_layout[0][2], (x.ten.shape[0], x.num_channels[0]), 1)
            .add(1).sqrt()
            )
            x1 = divisor - 1
            divisor_broadcasted = torch.index_select(divisor, dim=1, index=x.rep_layout[0][2]) 
            x2 = x.ten.div(divisor_broadcasted) 
        else:
            assert self._n_invariant_channels <= x.metadata[0][0] 
            x10 = x.ten[:, :self._n_invariant_channels]
            divisor = (
                SumsqrContraction1d.apply(
                    x.ten[:, self._n_invariant_channels:],
                    x.rep_layout[0][2, self._n_invariant_channels:] - self._n_invariant_channels,
                    (x.ten.shape[0], x.num_channels[0] - self._n_invariant_channels),
                    1,
                ).add(1).sqrt()
            )
            x11 = divisor - 1 
            x1 = torch.cat([x10, x11], dim=1)
            divisor_broadcasted = torch.index_select(
                divisor, dim=1, 
                index=(x.rep_layout[0][2, self._n_invariant_channels:] - self._n_invariant_channels)
            ) 
            x2 = x.ten[:, self._n_invariant_channels:].div(divisor_broadcasted) 
        return x1, x2 


class EvMLP1d(torch.nn.Module):
    def __init__(self, metadata, norm, activation_func, dropout=0.0):
        super().__init__() 
        self.n_invariant_channels = metadata[0].item() 
        self.num_channels = torch.sum(metadata).item() 
        self.evnorm = EvNorm1d(self.num_channels, self.n_invariant_channels) 
        self.dropout = dropout 
        if norm is not None:
            self.mlp = torch.nn.Sequential(
                norm, 
                torch.nn.Linear(self.num_channels),
                activation_func,
                torch.nn.Linear(self.num_channels, self.num_channels),
                torch.nn.Dropout(p=self.dropout)
            )
        else:
            self.mlp = torch.nn.Linear(self.num_channels, self.num_channels) 

    def forward(self, x:SphericalTensor) -> SphericalTensor:
        x1, x2 = self.evnorm(x) 
        xi = self.mlp(x1) 
        gate = torch.index_select(
            xi[:, self.n_invariant_channels:],
            dim = 1, 
            index = (
                x.rep_layout[0][2, self.n_invariant_channels:] - self.n_invariant_channels 
            ) 
        )
        return x.self_like(
            new_data_ten=torch.cat([xi[:, :self.n_invariant_channels], x2.mul(gate)], dim=-1)
        )


class KernelBroadcast(torch.nn.Module):
    """
    Specialized module for broadcasting scalar features with a SO(3) kernel 
    of singleton channel-sizes per angular quantum number l. 
    """
    def __init__(self, target_metadata):
        super().__init__()
        self.in_num_channels = len(target_metadata) 
        self.target_num_channels = torch.sum(target_metadata).item() 
        filter_broadcast_idx = []
        src_idx = 0 
        for l, n_lm in enumerate(target_metadata):
            for m in range(2*l+1):
                filter_broadcast_idx.append(
                    torch.full((n_lm.item(),), src_idx, dtype=torch.long) 
                )
                src_idx += 1 
        self.register_buffer("filter_broadcast_idx", torch.cat(filter_broadcast_idx)) 
        self.register_buffer("feat_broadcast_idx", SphericalTensor.generate_rep_layout_1d_(target_metadata)[2,:])
    
    def forward(self, rshs:SphericalTensor, feat:torch.Tensor) -> torch.Tensor:
        assert self.in_num_channels == rshs.num_channels[0]
        assert self.target_num_channels == feat.shape[-1]
        broadcasted_rshs = torch.index_select(
            rshs.ten, dim=rshs.ten.dim()-1, index=self.filter_broadcast_idx
        )
        broadcasted_feat = torch.index_select(
            feat, dim=feat.dim()-1, index=self.feat_broadcast_idx
        )
        return broadcasted_feat * broadcasted_rshs 


class RepNorm1d(torch.nn.Module):
    r"""
    The Representation Normalization layer.
    """
    def __init__(self, num_channels, norm="batch", momentum=0.1, eps=1e-2, n_invariant_channels=0, mode="raw", invariant_mode="l2"):
        super().__init__()
        self._num_channels = num_channels
        self._n_invariant_channels = n_invariant_channels
        self._eps = eps
        self._mode = mode
        self._invariant_mode = invariant_mode 
        if norm == "batch":
            self.norm = torch.nn.BatchNorm1d(num_features=num_channels, momentum=momentum, affine=False) 
        elif norm == "node":
            self.norm = torch.nn.LayerNorm(normalized_shape=num_channels, elementwise_affine=False) 
        elif norm == "none":
            self.norm = None 
        else:
            raise NotImplementedError 
        if self._mode == "raw":
            self.beta = torch.nn.Parameter(0.5 + torch.rand(self._num_channels - self._n_invariant_channels) * 1.0) 
        elif self._mode == "inv":
            self.betainv = torch.nn.Parameter(torch.rand(self._num_channels - self._n_invariant_channels) * 10.0) 
        else:
            raise NotImplementedError 
    
    def forward(self, x:SphericalTensor) -> Tuple[torch.Tensor, SphericalTensor]:
        if self.norm is None:
            if self._invariant_mode == "l2":
                x0 = NormContraction1d.apply(
                    x.ten, x.rep_layout[0][2], (x.ten.shape[0], x.num_channels[0]), 1, 1e-04
                )
            elif self._invariant_mode == "sumsqr":
                x0 = SumsqrContraction1d.apply(
                    x.ten, x.rep_layout[0][2], (x.ten.shape[0], x.num_channels[0]), 1
                ) 
            else:
                raise NotImplementedError 
            return x0, x 
        assert x.ten.dim() == 2 
        if self._mode == "raw":
            beta = self.beta 
        elif self._mode == "inv":
            beta = self.betainv.abs().add(self._eps).reciprocal() 
        else:
            raise NotImplementedError 
        
        if self._n_invariant_channels == 0:
            if self._invariant_mode == "l2":
                x0 = NormContraction1d.apply(
                    x.ten, x.rep_layout[0][2], (x.ten.shape[0], x.num_channels[0]), 1, 1e-04,
                )
            elif self._invariant_mode == "sumsqr":
                x0 = SumsqrContraction1d.apply(
                    x.ten, x.rep_layout[0][2], (x.ten.shape[0], x.num_channels[0]), 1
                )
            else:
                raise NotImplementedError
            x1 = self.norm(x0)
            divisor = x0.add(beta).abs().add(self._eps)
            divisor_broadcasted = torch.index_select(divisor, dim=1, index=x.rep_layout[0][2])
            x2_ten = x.ten.div(divisor_broadcasted)
        else:
            x0 = x.invariant(mode=self._invariant_mode)
            assert self._n_invariant_channels <= x.metadata[0][0]
            x1 = self.norm(x0) 
            if self._invariant_mode == "uest":
                divisor = (
                    x0[:, self._n_invariant_channels:].pow(2).add(1+beta.pow(2)) + self._eps**2
                ).sqrt() 
            else:
                divisor = (
                    torch.abs(x0[:, self._n_invariant_channels:].add(beta)) + self._eps
                ) 
            divisor_broadcasted = torch.index_select(
                divisor, dim=1, index=(x.rep_layout[0][2, self._n_invariant_channels:] - self._n_invariant_channels) 
            )
            x2_ten = torch.cat(
                [torch.ones_like(x1[:, :self._n_invariant_channels]), x.ten[:, self._n_invariant_channels:].div(divisor_broadcasted)], dim=1 
            )
        x2 = x.self_like(x2_ten) 
        return x1, x2 


class SpherEmbed(torch.nn.Module):
    """
    Spherical Embedding Layer for given node feature (l=0 invariants).
    """
    def __init__(self, target_metadata:torch.LongTensor, group="o3", Zmax=87):
        super().__init__()
        assert target_metadata.dim() == 1
        group = group.lower()
        if group == "so3":
            self.tensor_class = SphericalTensor
            self.n_irreps_per_l = torch.arange(start=0, end=target_metadata.size(0)) * 2 + 1 
        elif group == "o3":
            self.tensor_class = O3Tensor 
            n_irreps_per_l = torch.arange(start=0, end=target_metadata.size(0)//2) * 2 + 1
            self.n_irreps_per_l = n_irreps_per_l.repeat_interleave(2)
        else:
            raise NotImplementedError(f"The declaring group {group} has not been supported yet.")

        self._metadata = target_metadata
        self.Zmax = Zmax 
        self.embedding = Embedding(target_metadata[0].item(), Zmax)
    
    def forward(self, Z:torch.LongTensor) -> SphericalTensor:
        x_ten = torch.zeros(*Z.shape[:-1], torch.sum(self._metadata * self.n_irreps_per_l).item())
        n_invariants = self._metadata[0].item()
        x_ten[..., :n_invariants] = self.embedding(Z) 
        return self.tensor_class(
            data_ten=x_ten,
            rep_dims=(x_ten.dim()-1,),
            metadata=self._metadata,
        )


class SelfMixing(torch.nn.Module):
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
        num_channels_out = torch.sum(self.CGCoupler.metadata_out, dim=1).item() 
        self.register_parameter("keep_coeff", torch.nn.Parameter(torch.Tensor(num_channels_in)))
        self.register_parameter("mix_coeff", torch.nn.Parameter(torch.Tensor(num_channels_out)))
        self._init_parameters()

    def _init_parameters(self):
        torch.nn.init.uniform_(self.keep_coeff, a=-np.sqrt(3), b=np.sqrt(3))
        torch.nn.init.uniform_(self.mix_coeff, a=-np.sqrt(3), b=np.sqrt(3)) 
        if self.group == "so3":
            dummy_coupler = CGCoupler(self.metadata_in, self.metadata_in, max_l=self.order_out, overlap_out=True, trunc_in=False)
            metadata_out = dummy_coupler.metadata_out[0, :self.metadata_in.shape[0]] 
            n_irreps_per_l =  torch.arange(start=0, end=self.metadata_in.shape[0]) * 2 + 1 
        elif self.group == "o3":
            dummy_coupler = CGPCoupler(self.metadata_in, self.metadata_in, max_l=self.order_out, overlap_out=True, trunc_in=False)
            metadata_out = dummy_coupler.metadata_out[0, :self.metadata_in.shape[0]]
            n_irreps_per_l =  torch.arange(start=0, end=self.metadata_in.shape[0]//2) * 2 + 1
            n_irreps_per_l = n_irreps_per_l.repeat_interleave(2)
        else:
            raise NotImplementedError 

        metadata_tmp = torch.stack([self.metadata_in, metadata_out], dim=0)
        repid_offsets = torch.cumsum(
            metadata_tmp * n_irreps_per_l.unsqueeze(0), dim=1
        )
        repid_offsets = torch.cat(
            [torch.LongTensor([[0], [0]]), repid_offsets[:, :-1]], dim=1
        ).long() 
        # generate representation channels for sum up
        metadata_cp = torch.minimum(self.metadata_in, metadata_out)
        repids_in, repids_out = [], []
        for cur_l in range(self.metadata_in.shape[0]):
            repids_in_3j = (
                repid_offsets[0, cur_l] + torch.arange(metadata_cp[cur_l])
            )
            repids_out_3j = (
                repid_offsets[1, cur_l] + torch.arange(metadata_cp[cur_l])
            )
            repids_in.append(repids_in_3j)
            repids_out.append(repids_out_3j) 
        self.register_buffer("repids_in", torch.cat(repids_in).long())
        self.register_buffer("repids_out", torch.cat(repids_out).long())
        self.register_buffer("metadata_out", dummy_coupler.metadata_out)
        self.register_buffer("repids_cp_out", dummy_coupler.repids_out)
        self.register_buffer("out_layout", dummy_coupler.out_layout)

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
            out_shape, dtype=x.dtype, device=x.device
        ).index_add_(
            coupling_dim, self.repids_cp_out, cat_tp_out
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
            metadata=self.metadata_out
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

      




