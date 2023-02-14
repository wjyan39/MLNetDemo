import os
import torch 
import torch.nn as nn

from joblib import Memory

from BasicUtility.O3.O3Tensor import SphericalTensor, O3Tensor
from BasicUtility.O3.O3Utility import csh_to_rsh, get_clebsh_gordan_coefficient

memory = Memory(os.path.join(".", ".tp_cache"), verbose=0)

class TPExpansion(nn.Module):
    """
    Tensor Production Expansion Layer.
    Inverse to tensor product contractions, used to expand SphericalTensor 
    irreps to (2l_1 + 1) x (2l_2 + 1) matrix that represents its contribution 
    to the direct sum representation of the tensor product of two irreps of 
    degree l1 and l2. 
    """
    def __init__(self, metadata:torch.LongTensor, l1:int, l2:int, overlap_out=True, dtype=torch.double):
        assert metadata.dim() == 1
        assert metadata.minimum().item() > 0
        super().__init__()
        max_l_in = metadata.shape[0]
        self.max_l = min(max_l_in, l1 + l2) 
        self.l1 = l1
        self.l2 = l2 
        self.metadata = metadata
        self.dtype = dtype
        self._init_params(overlap_out)
    
    def _init_params(self, overlap_out:bool):
        n_irreps_per_l = torch.arange(start=0, end=self.max_l) * 2 + 1
        
        repid_offsets_in = torch.cumsum(
            self.metadata * n_irreps_per_l.unsqueeze(0), dim=1
        )
        repid_offsets_in = torch.cat(
            [torch.LongTensor([[0], [0]]), repid_offsets_in[:, :-1]], dim=1
        ).long() 

        degeneracy = self.metadata.minimum().item()
        cg_tilde, repids_in, repids_out = [], [], [] 
        block_dim = (2*self.l1 + 1) * (2*self.l2 + 1)
        out_blk_offset = 0
        # loop over l
        for l in range(self.max_l + 1):
            if self.l1 + self.l2 < l or abs(self.l1 - self.l2) > l:
                continue 
            # l --> l1 \osum l2
            cg_source = get_rsh_cg_coefficients_all(self.l1, self.l2, l) 
            cg_segment = cg_source.repeat_interleave(degeneracy, dim=1)
            ns_segment = torch.arange(degeneracy).repeat(cg_source.shape[1]) 
            repids_in_3j = (
                repid_offsets_in[l] 
                + (cg_segment[2] + l) * self.metadata[l]
                + ns_segment
            )
            repids_out_blk = (
                torch.arange(block_dim * degeneracy) + out_blk_offset
            )
            if not overlap_out:
                out_blk_offset += block_dim * degeneracy

            cg_tilde.append(cg_segment[3])
            repids_in.append(repids_in_3j) 
            repids_out.append(repids_out_blk)
        self.reshaping = True if not overlap_out else False
        self.register_buffer("cg_tilde", torch.cat(cg_tilde).type(self.dtype))
        self.register_buffer("repids_in", torch.cat(repids_in).long())
        self.register_buffer("repids_out", torch.cat(repids_out).long())

    def forward(self, x:SphericalTensor) -> torch.Tensor:
        assert len(x.rep_dims) == 1
        assert torch.all(x.metadata[0].eq(self.metadata))
        coupling_dim = x.rep_dims[0]
        x_tilde = torch.index_select(x.ten, dim=coupling_dim, index=self.repids_in) 
        broadcast_shape = tuple(
            self.cg_tilde.shape[0] if d == coupling_dim else 1 for d in range(x.ten.dim())
        )
        tp_tilde = x_tilde * self.cg_tilde.view(broadcast_shape) 
        tp_out_shape = tuple(
            self.repids_out.shape[0] if d == coupling_dim else x.ten.shape[d] for d in range(x.ten.dim())
        )
        tp_out:torch.Tensor = torch.zeros(
            tp_out_shape, dtype=x_tilde.dtype, device=x_tilde.device 
        ).index_add_(
            coupling_dim, self.repids_in, tp_tilde 
        )
        out_ten = tp_out.index_add_(
            coupling_dim, self.repids_out, tp_out
        )
        if self.reshaping:
            block_dim = (2*self.l1 + 1) * (2*self.l2 + 1)
            out_ten = out_ten.view(*x.ten.shape[:coupling_dim], -1, block_dim, *x.ten.shape[coupling_dim+1:])
        return out_ten


@memory.cache 
def get_rsh_cg_coefficients_all(j1, j2, j):
    csh_cg = torch.zeros(2*j1 + 1, 2*j2 + 1, 2*j + 1, dtype=torch.double) 
    for m1 in range(-j1, j1+1):
        for m2 in range(-j2, j2+1):
            if abs(m1 + m2) > j:
                continue 
            csh_cg[j1 + m1, j2 + m2, j + m1 + m2] = get_clebsh_gordan_coefficient(j1, j2, m1, m2, m1+m2) 
    c2r_j1, c2r_j2, c2r_j = csh_to_rsh(j1), csh_to_rsh(j2), csh_to_rsh(j)         
    # making the coefficients all real 
    rsh_cg = torch.einsum(
        "abc, ai, bj, ck -> ijk", csh_cg.to(torch.double), c2r_j1, c2r_j2, c2r_j.conj()
    )*(-1j)**(j1+j2+j) 
    assert torch.allclose(rsh_cg.imag, torch.zeros_like(csh_cg)), print(csh_cg, rsh_cg) 
    return cg_idx(rsh_cg.real, j1, j2, j) 

def cg_idx(coeffs, j1, j2, j):
    j1s = torch.arange(-j1, j1+1).view(2*j1 + 1, 1, 1).expand_as(coeffs) 
    j2s = torch.arange(-j2, j2+1).view(1, 2*j2 + 1, 1).expand_as(coeffs) 
    js  = torch.arange(-j, j + 1).view(1, 1,  2*j + 1).expand_as(coeffs) 
    return torch.stack(
        [j1s, j2s, js, coeffs], dim=0
    )