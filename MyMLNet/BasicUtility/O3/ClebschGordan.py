from joblib import Memory 
from sympy import N 
from sympy.physics.quantum.cg import CG 

import torch 
from BasicUtility.O3.O3Tensor import SphericalTensor, O3Tensor 
from BasicUtility.O3.Wigner import csh_to_rsh 

import os 
memory = Memory(os.path.join("./", "._o3_cache"), verbose=0) 

def get_clebsch_gordan_coefficient(j1, j2, j, m1, m2, m):
    return float(N(CG(j1, m1, j2, m2, j, m).doit())) 

@memory.cache 
def get_rsh_cg_coefficients(j1, j2, j):
    csh_cg = torch.zeros(2*j1+1, 2*j2+1, 2*j+1, dtype=torch.double) 
    for m1 in range(-j1, j1+1):
        for m2 in range(-j2, j2+1):
            if m1 + m2 < -j or m1 + m2 > j:
                continue 
            csh_cg[j1+m1, j2+m2, j+m1+m2] = get_clebsch_gordan_coefficient(j1, j2, j, m1, m2, m1+m2)
    
    c2r_j1, c2r_j2, c2r_j = csh_to_rsh(j1), csh_to_rsh(j2), csh_to_rsh(j) 
    rsh_cg = torch.einsum(
        "abc, ai, bj, ck -> ijk", csh_cg.to(torch.cdouble), c2r_j1, c2r_j2, c2r_j.conj()
    ) * (-1j) ** (j1 + j2 + j) 
    
    assert torch.allclose(rsh_cg.imag, torch.zeros_like(csh_cg)), print(csh_cg, rsh_cg) 
    return cg_compactify(rsh_cg.real, j1, j2, j) 

def cg_compactify(coeffs, j1, j2, j):
    j1s = torch.arange(-j1, j1+1).view(2*j1+1, 1, 1).expand_as(coeffs) 
    j2s = torch.arange(-j2, j2+1).view(1, 2*j2+1, 1).expand_as(coeffs)
    js = torch.arange(-j, j+1).view(1, 1, 2*j+1).expand_as(coeffs) 
    nonzero_mask = coeffs.abs() > 1e-12 
    return torch.stack(
        [j1s[nonzero_mask], j2s[nonzero_mask], js[nonzero_mask], coeffs[nonzero_mask]], dim=0
    )

class CGCoupler(torch.nn.Module):
    r"""
    General vectorized Clebsch-Gordan coupling module.
    Note:
        Once the Coupler is initialized, a compact view of representation indices is generated for 
        vectorizing Clebsch-Gordan coupling between two SphericalTensor. This tabulating step can 
        be time-consuming, depending on the input SphericalTensors. The parameters of CGCoupler are
        saved in case the users intend to use as inference. 
    Args:
        metadata_1 ::torch.LongTensor:: The representation metadata of the first tensor to be coupled.
        metadata_2 ::torch.LongTensor:: The representation metadata of the second tensor to be coupled.
        parity ::int:: The parity to be retained during coupling. 0: default, no parity selection; 
            1:Polar tensor; -1: Pseudo tensor.
        max_l ::int:: Maximum L allowed for coupling output.
        overlap_out ::bool:: If true, coupling outputs from the same input feature index but different
            (l1, l2) pairs will be accumulated to the same representation index of the output Spherical-
            -Tensor, otherwise, apply concatenation. 
        trunc_in ::bool:: If true, the allowed feature indices (n) will be further truncated such that
            for each set of terms (l1, l2, n), the coupling results will saturate all possible (l_out, n)
            values of the output SphericalTensor.
        dtype ::torch.dtype:: The dtype for tensor to be passed in coupling, must be specified beforehand.
    """

    def __init__(self, metadata_1:torch.LongTensor, metadata_2:torch.LongTensor, max_l, parity=0, overlap_out=True, trunc_in=True, dtype=torch.double):
        super().__init__() 
        metadata_1 = torch.LongTensor(metadata_1)
        metadata_2 = torch.LongTensor(metadata_2) 
        assert metadata_1.dim() == 1
        assert metadata_2.dim() == 1
        # in_size1, in_size2 = metadata_1.shape[0], metadata_2.shape[0]
        self.metadata_out = None 
        self.metadata_in1 = metadata_1 
        self.metadata_in2 = metadata_2 
        self.parity = parity
        # max_l should be below L1 + L2
        self.max_l = min(max_l, metadata_1.shape[0] + metadata_2.shape[0] - 2) 
        self.dtype = dtype
        # generate initialization settings
        self._init_params(overlap_out, trunc_in) 
        self.register_buffer("out_layout", SphericalTensor.generate_rep_layout_1d_(self.metadata_out)) 
    
    def _init_params(self, overlap_out:bool, trunc_in:bool):
        in_size1 = self.metadata_in1.shape[0]
        in_size2 = self.metadata_in2.shape[0] 
        in_size = max(in_size1, in_size2) 
        n_irreps_per_l = torch.arange(start=0, end=in_size) * 2 + 1
        # zero padding 
        tmp_zeros = torch.zeros(abs(in_size1 - in_size2)).long()
        if in_size1 < in_size2:
            self.metadata_in1 = torch.cat([self.metadata_in1, tmp_zeros], dim=0)
        elif in_size1 > in_size2:
            self.metadata_in2 = torch.cat([self.metadata_in2, tmp_zeros], dim=0)  
        assert self.metadata_in1.shape[0] == self.metadata_in2.shape[0]
        metadata_in = torch.stack([self.metadata_in1, self.metadata_in2], dim=0) 
        repid_offsets_in = torch.cumsum(
            metadata_in * n_irreps_per_l.unsqueeze(0), dim=1
        )
        repid_offsets_in = torch.cat(
            [torch.LongTensor([[0], [0]]), repid_offsets_in[:, :-1]], dim=1
        ).long() 

        max_n_out = torch.zeros(self.max_l).long()
        max_n_out[:in_size] = torch.maximum(self.metadata_in1, self.metadata_in2) 
        max_n_out[in_size:] = max_n_out[in_size-1].item() 
        cg_tilde, repids_in1, repids_in2, repids_out = [], [], [], [] 
        valid_coupling_ids = [] 
        metadata_out = torch.zeros_like(max_n_out) 
        # loop over l_out, deal each valid coupling pair, get metadata_out
        for lout in range(self.max_l + 1):
            # loop over l1
            for lin1 in range(in_size1):
                # loop over l2 
                for lin2 in range(in_size2):
                    # check if parity maintaining is required 
                    coupling_parity = (-1) ** (lout + lin1 + lin2)
                    if not self.parity == 0:
                        if self.parity != coupling_parity:
                            continue 
                    # check if current (l1, l2) yields valid l_out
                    if lin1 + lin2 < lout or abs(lin1 - lin2) > lout:
                        continue 
                    # determine feature_channels in current coupling 
                    if trunc_in:
                        if lin1 + lin2 > self.max_l:
                            continue 
                        degeneracy = min(
                            metadata_in[0, lin1],
                            metadata_in[1, lin2],
                            max_n_out[lin1 + lin2]
                        ) 
                    else:
                        degeneracy = min(
                            metadata_in[0, lin1],
                            metadata_in[1, lin2],
                            max_n_out[lout] 
                        )
                    if not overlap_out:
                        # concatenation
                        metadata_out[lout] += degeneracy
                    elif degeneracy > metadata_out[lout]:
                        # accumulation
                        metadata_out[lout] = degeneracy 
                    # record valid coupling pattern 
                    if degeneracy > 0:
                        valid_coupling_ids.append((lout, lin1, lin2, degeneracy)) 
        # offsets of output tensor
        repid_offsets_out = torch.cumsum(metadata_out * n_irreps_per_l, dim=0) 
        repid_offsets_out = torch.cat(
            [torch.LongTensor([0]), repid_offsets_out[:-1]], dim=0
        )
        out_ns_offset, lout_last = 0, 0
        # Generate flattened coupling coefficients
        # items in valid_coupling_ids is in increasing order of lout 
        for (lout, lin1, lin2, degeneracy) in valid_coupling_ids:
            if lout > lout_last:
                out_ns_offset = 0
            cg_source = get_rsh_cg_coefficients(lin1, lin2, lout) 
            cg_segment = cg_source.repeat_interleave(degeneracy, dim=1)
            ns_segment = torch.arange(degeneracy).repeat(cg_source.shape[1]) 
            # Calculating the representation IDs for the coupling tensors
            repids_in1_3j = (
                repid_offsets_in[0, lin1] 
                + (cg_segment[0] + lin1) * metadata_in[0, lin1]
                + ns_segment
            )
            repids_in2_3j = (
                repid_offsets_in[1, lin2]
                + (cg_segment[1] + lin2) * metadata_in[1, lin2]
                + ns_segment
            )
            repids_out_3j = (
                repid_offsets_out[lout]
                + (cg_segment[2] + lout) * metadata_out[lout]
                + out_ns_offset
                + ns_segment
            )

            cg_tilde.append(cg_segment[3])
            repids_in1.append(repids_in1_3j)
            repids_in2.append(repids_in2_3j) 
            repids_out.append(repids_out_3j)
            if not overlap_out:
                out_ns_offset += degeneracy
            lout_last = lout 
        
        self.register_buffer("cg_tilde", torch.cat(cg_tilde).type(self.type)) 
        self.register_buffer("repids_in1", torch.cat(repids_in1).long())
        self.register_buffer("repids_in2", torch.cat(repids_in2).long())
        self.register_buffer("repids_out", torch.cat(repids_out).long())
        self.metadata_out = metadata_out 

    def forward(self, x1:SphericalTensor, x2:SphericalTensor) -> SphericalTensor:
        assert len(x1.rep_dims) == 1
        assert len(x2.rep_dims) == 1
        assert x1.rep_dims[0] == x2.rep_dims[0]
        coupling_dim = x1.rep_dims[0]
        assert torch.all(x1.metadata[0].eq(self.metadata_in1))
        assert torch.all(x2.metadata[0].eq(self.metadata_in2))
        # select the features for coupling in a vectorized way 
        x1_tilde = torch.index_select(x1.ten, dim=coupling_dim, index=self.repids_in1)
        x2_tilde = torch.index_select(x2.ten, dim=coupling_dim, index=self.repids_in2)
        # broadcast cg coefficients 
        broadcast_shape = tuple(
            self.cg_tilde.shape[0] if d == coupling_dim else 1 for d in range(x1.ten.dim())
        )
        # tensor product: multiplication  
        out_tilde = x1_tilde * x2_tilde * self.cg_tilde.view(broadcast_shape)  
        out_shape = tuple(
            self.out_layout.shape[1] if d == coupling_dim else x1.ten.shape[d] for d in range(x1.ten.dim())
        )
        # tensor product: sum 
        out_ten = torch.zeros(
            out_shape, dtype=x1_tilde.dtype, device=x1_tilde.device
        ).index_add_(
            coupling_dim, self.repids_out, out_tilde
        )

        return SphericalTensor(
            data_ten=out_ten, 
            rep_dims=(coupling_dim,), 
            metadata=self.metadata_out.unsqueeze(0), 
            rep_layout=(self.out_layout,)
        )

class CGPCoupler(torch.nn.Module):
    """
    Parity-aware vectorized Clebsch-Gordan coupling module.
    From CGCoupler, with a little modification.
    """
    def __init__(self, metadata_1:torch.LongTensor, metadata_2:torch.LongTensor, max_l, overlap_out=True, trunc_in=True, dtype=torch.double):
        super().__init__()
        metadata_1 = torch.LongTensor(metadata_1)
        metadata_2 = torch.LongTensor(metadata_2) 
        assert metadata_1.dim() == 1
        assert metadata_2.dim() == 1
        assert metadata_1.shape[0] % 2 == 0
        assert metadata_2.shape[0] % 2 == 0 
        self.metadata_out = None 
        self.metadata_in1 = metadata_1 
        self.metadata_in2 = metadata_2 
        # max_l should be below L1 + L2
        self.max_l = min(max_l, (metadata_1.shape[0] + metadata_2.shape[0]) // 2 - 2) 
        self.dtype = dtype
        # generate initialization settings
        self._init_params(overlap_out, trunc_in) 
        self.register_buffer("out_layout", SphericalTensor.generate_rep_layout_1d_(self.metadata_out)) 
    
    def _init_params(self, overlap_out:bool, trunc_in:bool):
        in_size1 = self.metadata_in1.shape[0]
        in_size2 = self.metadata_in2.shape[0] 
        in_size = max(in_size1, in_size2) 
        n_irreps_per_l = torch.arange(start=0, end=in_size // 2) * 2 + 1
        n_irreps_per_lp = n_irreps_per_l.repeat_interleave(2) 
        # zero padding 
        tmp_zeros = torch.zeros(abs(in_size1 - in_size2)).long()
        if in_size1 < in_size2:
            self.metadata_in1 = torch.cat([self.metadata_in1, tmp_zeros], dim=0)
        elif in_size1 > in_size2:
            self.metadata_in2 = torch.cat([self.metadata_in2, tmp_zeros], dim=0)  
        assert self.metadata_in1.shape[0] == self.metadata_in2.shape[0]
        metadata_in = torch.stack([self.metadata_in1, self.metadata_in2], dim=0) 
        repid_offsets_in = torch.cumsum(
            metadata_in * n_irreps_per_lp.unsqueeze(0), dim=1
        )
        repid_offsets_in = torch.cat(
            [torch.LongTensor([[0], [0]]), repid_offsets_in[:, :-1]], dim=1
        ).long()
        
        max_n_out = torch.zeros(self.max_l).long()
        max_n_out[:in_size] = torch.maximum(self.metadata_in1, self.metadata_in2) 
        max_n_out[in_size:] = max_n_out[in_size-1].item()
        cg_tilde, repids_in1, repids_in2, repids_out = [], [], [], []
        valid_coupling_ids = []
        metadata_out = torch.zeros_like(max_n_out)
        
        for lout in range(self.max_l + 1):
            for pout in (1, -1):
                for lin1 in range(in_size1 // 2):
                    for lin2 in range(in_size2 // 2):
                        for pin1 in (1, -1):
                            for pin2 in (1, -1):
                                coupling_parity = (-1) ** (lout + lin1 + lin2)
                                # pairity selection 
                                if pin1 * pin2 * coupling_parity != pout:
                                    continue
                                # angular momentum selection 
                                if lin1 + lin2 < lout or abs(lin1 - lin2) > lout:
                                    continue
                                
                                lpin1 = 2 * lin1 + (1 - pin1) // 2
                                lpin2 = 2 * lin2 + (1 - pin2) // 2
                                lpout = 2 * lout + (1 - pout) // 2
                                
                                if trunc_in:
                                    if lin1 + lin2 > self.max_l:
                                        continue 
                                    degeneracy = min(
                                        metadata_in[0, lpin1],
                                        metadata_in[1, lpin2],
                                        max_n_out[2 * (lin1 + lin2) + (1 - pout) // 2]
                                    )
                                else:
                                    degeneracy = min(
                                        metadata_in[0, lpin1],
                                        metadata_in[1, lpin2],
                                        max_n_out[lpout]
                                    )
                                if not overlap_out:
                                    # concatenation
                                    metadata_out[lout] += degeneracy
                                elif degeneracy > metadata_out[lout]:
                                    # accumulation
                                    metadata_out[lout] = degeneracy 
                                # record valid coupling pattern 
                                if degeneracy > 0:
                                    valid_coupling_ids.append((lout, lin1, lin2, degeneracy))
        
        repid_offsets_out = torch.cumsum(metadata_out * n_irreps_per_lp, dim=0)
        repid_offsets_out = torch.cat(
            [torch.LongTensor([0]), repid_offsets_out[:-1]], dim=0
        )
        out_ns_offset, lpout_last = 0, 0
        # Generate flattened coupling coefficients
        # items in valid_coupling_ids is in increasing order of lout 
        for (lpout, lpin1, lpin2, degeneracy) in valid_coupling_ids:
            if lpout > lpout_last:
                out_ns_offset = 0
            lin1, lin2, lout = lpin1//2, lpin2//2, lpout//2 
            cg_source = get_rsh_cg_coefficients(lin1, lin2, lout) 
            cg_segment = cg_source.repeat_interleave(degeneracy, dim=1)
            ns_segment = torch.arange(degeneracy).repeat(cg_source.shape[1]) 
            # Calculating the representation IDs for the coupling tensors
            repids_in1_3j = (
                repid_offsets_in[0, lpin1] 
                + (cg_segment[0] + lin1) * metadata_in[0, lpin1]
                + ns_segment
            )
            repids_in2_3j = (
                repid_offsets_in[1, lpin2]
                + (cg_segment[1] + lin2) * metadata_in[1, lpin2]
                + ns_segment
            )
            repids_out_3j = (
                repid_offsets_out[lpout]
                + (cg_segment[2] + lout) * metadata_out[lpout]
                + out_ns_offset
                + ns_segment
            )

            cg_tilde.append(cg_segment[3])
            repids_in1.append(repids_in1_3j)
            repids_in2.append(repids_in2_3j) 
            repids_out.append(repids_out_3j)
            if not overlap_out:
                out_ns_offset += degeneracy
            lpout_last = lpout 

        self.register_buffer("cg_tilde", torch.cat(cg_tilde).type(self.type)) 
        self.register_buffer("repids_in1", torch.cat(repids_in1).long())
        self.register_buffer("repids_in2", torch.cat(repids_in2).long())
        self.register_buffer("repids_out", torch.cat(repids_out).long())
        self.metadata_out = metadata_out

    def forward(self, x1:O3Tensor, x2:O3Tensor) -> O3Tensor:
        assert len(x1.rep_dims) == 1
        assert len(x2.rep_dims) == 1
        assert x1.rep_dims[0] == x2.rep_dims[0]
        coupling_dim = x1.rep_dims[0]
        assert torch.all(x1.metadata[0].eq(self.metadata_in1))
        assert torch.all(x2.metadata[0].eq(self.metadata_in2))
        # select the features for coupling in a vectorized way 
        x1_tilde = torch.index_select(x1.ten, dim=coupling_dim, index=self.repids_in1)
        x2_tilde = torch.index_select(x2.ten, dim=coupling_dim, index=self.repids_in2)
        # broadcast cg coefficients 
        broadcast_shape = tuple(
            self.cg_tilde.shape[0] if d == coupling_dim else 1 for d in range(x1.ten.dim())
        )
        # tensor product: multiplication  
        out_tilde = x1_tilde * x2_tilde * self.cg_tilde.view(broadcast_shape)  
        out_shape = tuple(
            self.out_layout.shape[1] if d == coupling_dim else x1.ten.shape[d] for d in range(x1.ten.dim())
        )
        # tensor product: sum 
        out_ten = torch.zeros(
            out_shape, dtype=x1_tilde.dtype, device=x1_tilde.device
        ).index_add_(
            coupling_dim, self.repids_out, out_tilde
        )

        return O3Tensor(
            data_ten=out_ten, 
            rep_dims=(coupling_dim,), 
            metadata=self.metadata_out.unsqueeze(0), 
            rep_layout=(self.out_layout,)
        )


