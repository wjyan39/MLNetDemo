import torch 
import torch.nn as nn 

from BasicUtility.O3.O3Layers import IELin
from BasicUtility.O3.O3Tensor import SphericalTensor, O3Tensor 
from BasicUtility.ActivationFunc import activation_getter 
from PhiSNet.Mixing import SelfMixing

class SphLinear(nn.Module):
    """
    Spherical Linear layer in PhiSNet.
    """
    def __init__(self, metadata_in:torch.LongTensor, target_metadata:torch.LongTensor, max_l:int, group="so3"):
        assert target_metadata.shape[0] == max_l - 1
        super.__init__()
        self.metadata_in = metadata_in
        self.metadata_out = target_metadata
        self.group = group
        self.SelfMix = SelfMixing(metadata=metadata_in, order_out=max_l, group=group) 
        inter_metadata = self.SelfMix.metadata_out 
        self.IELinear = IELin(metadata_in=inter_metadata, metadata_out=target_metadata, group=group)

    def forward(self, x:SphericalTensor) -> SphericalTensor:
        assert torch.all(x.metadata[-1].eq(self.metadata_in))
        mix_x = self.SelfMix(x) 
        return self.IELinear(mix_x)


class Residual(nn.Module):
    """
    Residual Block in PhiSNet
    """
    def __init__(self, metadata:torch.LongTensor, activation, group="so3"):
        assert metadata.dim() == 1
        super().__init__()
        order = metadata.shape[0]
        self.metadata_in = metadata 
        self.sphlin1 = SphLinear(metadata_in=metadata, target_metadata=metadata, max_l=order, group=group)
        self.sphlin2 = SphLinear(metadata_in=metadata, target_metadata=metadata, max_l=order, group=group)
        self.activation = activation_getter(activation) 
    
    def forward(self, x:SphericalTensor) -> SphericalTensor:
        assert torch.all(x.metadata[-1].eq(self.metadata_in))
        x_cut = x 
        sph_x = self.sphlin2(self.activation(self.sphlin1(self.activation(x))))
        out_ten = x_cut.ten + sph_x.ten 
        return x.self_like(out_ten)


class ScaLinear(nn.Module):
    """
    Linear Transformation from scalar features toward each spherical representation.
    """
    def __init__(self, num_features, target_metadata:torch.LongTensor, group="so3"):
        assert target_metadata.dim() == 1
        super().__init__()
        self.num_features = num_features
        self.metadata_out = target_metadata
        self.nmat = target_metadata.shape[0] 
        self.max_n_out = target_metadata.max().item()
        self.linears = torch.nn.Parameter(
            torch.zeros(self.nmat, num_features, self.max_n_out) 
        )
        self._init_params(group)
    
    def _init_params(self, group:str):
        group = group.lower()
        if group == "so3":
            n_irreps_per_l = torch.arange(start=0, end=self.metadata_out.shape[0]) * 2 + 1
            self.tensor_class = SphericalTensor
        elif group == "o3":
            n_irreps_per_l = torch.arange(start=0, end=self.metadata_out.shape[0]//2) * 2 + 1
            n_irreps_per_l = n_irreps_per_l.repeat_interleave(2)
            self.tensor_class = O3Tensor
        else:
            raise NotImplementedError

        repid_offsets_out = torch.cumsum(self.metadata_out * n_irreps_per_l, dim=0)
        repid_offsets_out = torch.cat(
            [torch.LongTensor([0]), repid_offsets_out[:-1]], dim=0
        ) 
        
        matrix_select_idx, matid_to_fanin = [], []
        out_reduce_idx, out_reduce_mask = [], []
        for lidx in range(self.nmat):
            matrix_select_idx.append(
                (torch.arange(1, dtype=torch.long) + lidx).repeat(n_irreps_per_l[lidx])
            )
            matid_to_fanin.append(
                torch.full((1,), self.num_features, dtype=torch.long)
            )
            cur_n_out = self.metadata_out[lidx].item()
            for m in range(n_irreps_per_l[lidx]):
                dst_idx_lpm = (
                    dst_offset 
                    + torch.arange(self.max_n_out, dtype=torch.long) 
                ).contiguous()
                dst_mask_lpm = torch.zeros(
                    self.max_n_out, dtype=torch.long
                )
                dst_mask_lpm[:cur_n_out] = 1
                out_reduce_idx.append(dst_idx_lpm)
                out_reduce_mask.append(dst_mask_lpm)
                dst_offset += cur_n_out
        
        self.register_buffer("matrix_select_idx", torch.cat(matrix_select_idx))
        self.register_buffer("out_reduce_mask", torch.cat(out_reduce_mask).bool())
        self.register_buffer("out_reduce_idx", torch.cat(out_reduce_idx)[self.out_reduce_mask])
        self.register_buffer("out_layout", self.tensor_class.generate_rep_layout_1d_(self.metadata_out))
        self.matid_to_fanin = torch.cat(matid_to_fanin)
        self.n_gathered_mats = self.matrix_select_idx.shape[0]
        self._reset_params()

    def _reset_params(self):
        with torch.no_grad():
            for lid, fan_in in enumerate(self.matid_to_fanin):
                bound = torch.sqrt(1 / fan_in)
                self.linears.data[lid].uniform_(-bound, bound)
    
    def forward(self, x:torch.Tensor):
        assert x.shape[-1] == self.num_features
        in_ten = x.view(-1, self.num_features)
        padded_in_feat = torch.index_select(
            in_ten, dim=1, index=torch.arange(self.num_features).repeat(self.n_gathered_mats)
        )
        padded_in_feat = padded_in_feat.view(
            in_ten.shape[0], self.n_gathered_mats, self.num_features
        ).transpose(0, 1)
        gathered_linears = torch.index_select(
            self.linears, dim=0, index=self.matrix_select_idx
        )
        padded_out_ten = (
            torch.bmm(padded_in_feat, gathered_linears)
            .transpose(0, 1).contiguous()
            .view(in_ten.shape[0], -1)
        )
        out_ten = torch.zeros(
            in_ten.shape[0], self.out_layout.shape[1], dtype=x.dtype, device=x.device
        ).index_add_(1, self.out_reduce_idx, padded_out_ten[:, self.out_reduce_mask]) 
        out_ten = out_ten.view(*x.shape[:-1], -1)

        return self.tensor_class(
            data_ten=out_ten,
            rep_dims=(x.dim()-1,),
            metadata=self.metadata_out
        )

