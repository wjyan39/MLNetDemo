"""
Utilities for converting AO basis Quamtum Chemistry calculated 
feature matrices to machine learning convention, compatible 
with SphericalTensor data structure.
This file deals with PySCF form. 
"""

import torch 
import torch.nn as nn 
from BasicUtility.O3.O3Tensor import SphericalTensor

from pathlib import Path 

class OneBodyToSpherical(nn.Module):
    def __init__(self, metadata, rep_dim):
        super.__init__()
        self.rep_dim = rep_dim 
        self.metadata = torch.LongTensor(metadata)
        offset = 0
        out_ids = []
        for l, n in enumerate(self.metadata):
            m_idx = torch.arange(offset, offset + n* (2*l+1)).long() 
            # out_idx = in_idx.view(-1, 2*l + 1).t().flip(0).contiguous().view(-1) 
            if l == 1:
                # p orbital: x, y, z -> x, z, y
                m_idx = m_idx.view(-1, 2*l + 1)[:, (1, 3, 2)].contiguous().view(-1)
            out_ids.append(m_idx) 
            offset += n * (2*l + 1)
        self.register_buffer("out_ids", out_ids) 
        self.register_buffer("out_layout", SphericalTensor.generate_rep_layout_1d_(self.metadata)) 
        self.num_channels_1d = self.metadata.sum().item() 
    
    def forward(self, feat_ten:torch.Tensor) -> SphericalTensor:
        sp_feat_ten = torch.index_select(feat_ten, dim=self.rep_dim, index=self.out_ids) 
        return SphericalTensor(
            sp_feat_ten,
            (self.rep_dim,),
            self.metadata.unsqueeze(0),
            (self.num_channels_1d,),
            (self.rep_layout),
        )

m_idx_map_pyscf = {
    0:[0], 
    1:[0, 2, 1], # input (x, y, z) -> (x, z, y)
    2:[0, 1, 2, 3, 4], 
    3:[0, 1, 2, 3, 4, 5, 6], 
    4:[0, 1, 2, 3, 4, 5, 6, 7, 8],
}

m_idx_map_qcore = {
    0:[0],
    1:[2, 1, 0],
    2:[4, 3, 2, 1, 0],
}


class TwoBodyToSpherical(nn.Module):
    def __init__(self, metadata, rep_dims, basis_layout:dict, valence_only=True, m_idx_map=m_idx_map_pyscf):
        """
        Converting two-body features (order 2 tensor) loaded from QC Calculation to torch_gauge's 2d SphericalTensor.
        The returned 2d-SO(3) tensor is of symmetric layout along the rep_dims.        
        Args:
            basis_layout :: dictionary :: statements by the ao dictionary (value) of a typical element (key), e.g.
                {"C":["1s", "2s", "3s", "2p", "3p", "3d"], "H":["1s", "2s"],} etc. 
        Note:
            The layout of two-body feature tensor depends on the basis set employed during QC calculation.
        """
        super().__init__()
        self.metadata = torch.LongTensor(metadata) 
        self.rep_dims = rep_dims 
        self.angular_map = {
            "s": 0,
            "p": 1,
            "d": 2,
            "f": 3,
            "g": 4, 
            "h": 5,
        }
        self._generate_buffers(basis_layout, valence_only, m_idx_map)
    
    def _generate_buffers(self, basis_layout, valence_only, m_idx_map): 
        out_repid_map = {}
        n_irreps_per_l = torch.arange(self.metadata.shape[0], dtype=torch.long) * 2 + 1
        rep_offsets = torch.cat(
            [torch.LongTensor([0]), torch.cumsum(self.metadata * n_irreps_per_l, dim=0)[:-1]]
        )
        self.num_channels_1d = self.metadata.sum().item()
        self.num_reps_1d = int((self.metadata * n_irreps_per_l).sum().item()) 
        for element, aos_map in basis_layout.items():
            repid_map = torch.zeros(len(aos_map), dtype=torch.long) 
            dst_n_current = 0 
            l_last = 0 
            m_qc = 0 
            for ao_id, ao_name in enumerate(aos_map):
                n, l = int(ao_name[0]), self.angular_map[ao_name[1]] 
                if l > l_last:
                    if m_qc != 0:
                        raise ValueError(f"Invalid basis index: element={element}, l={l}, m={m_qc}.")
                    elif l not in m_idx_map:
                        raise KeyError(f"Angular momentum quamtum number l={l} not defined in m_idx_map.")
                    dst_n_current = 0 
                if not valence_only:
                    dst_n_current = n - l  
                # m index according to given m_idx_map 
                m_out = m_idx_map[l][m_qc]
                repid_map[ao_id] = rep_offsets[l] + dst_n_current + m_out*self.metadata[l] 
                # update m and n 
                if m_qc == 2 * l:
                    m_qc = 0
                    if valence_only:
                        dst_n_current += 1 
                    if dst_n_current > self.metadata[l]:
                        raise ValueError(f"Number of shells exceeds the metadata: element={element}, l={l}, n={dst_n_current}")
                else:
                    m_qc += 1
                l_last = l 
            out_repid_map[element] = torch.nn.Parameter(repid_map, requires_grad=False) 
        self.out_repid_map = torch.nn.ParameterDict(out_repid_map) 
        self.register_buffer("out_rep_layout", SphericalTensor.generate_rep_layout_1d_(self.metadata))

    def forward(self, atomsybs, feat_ten:torch.Tensor):
        dst_rep_ids_1d = torch.cat([self.out_repid_map[ele] for ele in atomsybs]) 
        dst_offsets_1d = torch.arange(
            len(atomsybs), dtype=torch.long, device=feat_ten.device
        ).repeat_interleave(
            torch.tensor([len(self.out_repid_map[ele]) for ele in atomsybs], dtype=torch.long, device=feat_ten.device) 
        )
        dst_flat_ids_1d = dst_offsets_1d * self.num_reps_1d + dst_rep_ids_1d 
        sp_2body_feat_ten_flat = torch.zeros(
            *feat_ten.shape[: self.rep_dims[0]], (self.num_reps_1d * len(atomsybs)) ** 2, *feat_ten.shape[self.rep_dims[1] + 1 :],
            dtype = feat_ten.dtype,
            device = feat_ten.device
        )
        # Scatter interleaved representations to the padded 2d layout
        dst_flat_ids_2d = (
            dst_flat_ids_1d.unsqueeze(1) * (self.num_reps_1d * len(atomsybs)) + dst_flat_ids_1d.unsqueeze(0)
        ).view(-1) 
        sp_2body_feat_ten_flat.index_add_(self.rep_dims[0], dst_flat_ids_2d, feat_ten.flatten(*self.rep_dims)) 
        sp_2body_feat_ten = (
            sp_2body_feat_ten_flat.view(
                *feat_ten.shape[:self.rep_dims[0]], 
                len(atomsybs), 
                self.num_reps_1d, 
                len(atomsybs), 
                self.num_reps_1d,
                *feat_ten.shape[self.rep_dims[1]+1 :] 
            ).transpose(self.rep_dims[1], self.rep_dims[1] + 1)
            .contiguous()
        )
        return SphericalTensor(
            sp_2body_feat_ten,
            rep_dims=(d+2 for d in self.rep_dims),
            metadata=self.metadata.unsqueeze(0).repeat(2, 1),
            num_channels=(self.num_channels_1d, self.num_channels_1d),
            rep_layout=(self.out_rep_layout, self.out_rep_layout)
        )

class OneBodySphericalToInterleaved(nn.Module):
    def __init__(self, metadata, basis_set="def2-tzvp-jkfit", padding_size=128, m_idx_map=m_idx_map_pyscf):
        """
        Converting model-predicted spherical tensor to QC calculation form layout.
        """
        super().__init__()
        self.metadata = torch.LongTensor(metadata) 
        self.padding_size = padding_size 
        out_repid_map = torch.zeros(120, self.padding_size, dtype=torch.long)
        out_repid_mask = torch.zeros(120, self.padding_size, dtype=torch.unit8)
        if basis_set == "def2-tzvp-jkfit":
            basis_metadata = {
                # In the order of (ns, np, nd, nf, ng, nh)
                # Experimental so manually parsed...
                1: (2, 2, 2, 0, 0, 0),
                6: (10, 8, 5, 1, 1, 0),
                7: (10, 8, 4, 2, 1, 0),
                8: (10, 8, 4, 2, 1, 0),
                9: (10, 8, 4, 2, 1, 0),
                15: (13, 11, 9, 4, 1, 0),
                16: (13, 11, 9, 4, 1, 0),
                17: (13, 11, 9, 4, 1, 0),
            }
        else:
            raise NotImplementedError
        n_irreps_per_l = torch.arange(len(metadata), dtype=torch.long)
        rep_offsets = torch.cat(
            [torch.LongTensor([0]), torch.cumsum(self.metadata * n_irreps_per_l, dim=0)[:-1]]
        )
        self.num_channels_1d = self.metadata.sum().item()
        self.num_reps_1d = int((self.metadata * n_irreps_per_l).sum().item())
        for element, shell_counts in basis_metadata.items():
            src_id = 0
            for l, N_l in enumerate(shell_counts):
                for dst_n_current in range(N_l):
                    if dst_n_current > self.metadata[l]:
                        raise ValueError(
                            f"Number of shells exceeding the metadata: element={element}, l={l}, n={dst_n_current} "
                        )
                    for m_qc in range(2 * l + 1):
                        # Reversing the magnetic quantum numbers to match the ordering
                        m_out = m_idx_map[l][m_qc]
                        out_repid_map[element, src_id] = rep_offsets[l] + dst_n_current + m_out * self.metadata[l]
                        out_repid_mask[element, src_id] = 1
                        src_id += 1
        self.register_buffer("out_repid_map", out_repid_map)
        self.register_buffer("out_repid_mask", out_repid_mask.bool())

    def forward(self, atomic_numbers, src_ten:torch.Tensor): 
        assert src_ten.dim() == 2 
        dst_rep_ids = self.out_repid_map[atomic_numbers, :]
        dst_rep_mask = self.out_repid_mask[atomic_numbers, :]
        out_ten_padded = torch.gather(src_ten, dim=1, index=dst_rep_ids)
        out_ten = out_ten_padded[dst_rep_mask]
        return out_ten 
    
    def reverse_atomic(self, atomic_numbers, out_ten:torch.Tensor):
        assert out_ten.dim() == 1 
        dst_rep_ids = self.out_repid_map[atomic_numbers, :]
        dst_rep_mask = self.out_repid_mask[atomic_numbers, :]
        out_ten_padded = torch.zeros(dst_rep_ids.shape, device=out_ten.device, dtype=out_ten.dtype)
        out_ten_padded[dst_rep_mask] = out_ten 
        src_ten = torch.zeros(atomic_numbers.shape[0], self.num_reps_1d, device=out_ten.device, dtype=out_ten.dtype)
        src_ten = src_ten.scatter_add_(dim=1, index=dst_rep_ids, src=out_ten_padded)
        return src_ten

THIS_FOLDER = Path(__file__).parent
RAW_3IDX = THIS_FOLDER / "orbnet2_overlap3idx.pt"
REDUCTION_DIM = 84

class OneBodyReduction(nn.Module):
    def __init__(self, metadata, basis_layout:dict, valence_only=True, file_3idx=RAW_3IDX, m_idx_map=m_idx_map_pyscf):
        super().__init__()
        self.metadata = torch.LongTensor(metadata)
        self.angular_map = {
            "s": 0, 
            "p": 1,
            "d": 2,
            "f": 3,
            "g": 4,
            "h": 5,
        }
        self.atomic_number_map = {
            "H": 1,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
        }
        n_irreps_per_l = torch.arange(len(metadata), dtype=torch.long)
        rep_offsets = torch.cat(
            [torch.LongTensor([0]), torch.cumsum(self.metadata * n_irreps_per_l, dim=0)[:-1]]
        )
        self.num_channels_1d = self.metadata.sum().item()
        self.num_reps_1d = int((self.metadata * n_irreps_per_l).sum().item())
        raw_3idx = torch.load(str(file_3idx.absolute()))
        self.n_out = raw_3idx["H"].shape[0]
        reduction_coefficients = torch.zeros(REDUCTION_DIM, self.num_reps_1d**2, self.n_out)
        self.possible_atomic_numbers = [self.atomic_number_map[ele] for ele in basis_layout.keys() if ele in self.atomic_number_map.keys()]
        for element, aos_map in basis_layout.items():
            if not element in self.atomic_number_map:
                continue
            src_3idx = raw_3idx[element]
            repid_map = torch.zeros(len(aos_map), dtype=torch.long)
            dst_n_current = 0
            l_last = 0
            m_qc = 0
            for ao_id, ao_name in enumerate(aos_map):
                n, l = int(ao_name[0]), self.angular_map[ao_name[1]]
                if l > l_last:
                    if m_qc != 0:
                        raise ValueError(f"Invalid basis index: element={element}, l={l}, m={m_qc}")
                    elif l not in m_idx_map:
                        raise KeyError(f"Angular momentum quamtum number l={l} not defined in m_idx_map.")
                    dst_n_current = 0
                if not valence_only:
                    dst_n_current = n - l
                # m index according to given m_idx_map 
                m_out = m_idx_map[l][m_qc]
                repid_map[ao_id] = rep_offsets[l] + dst_n_current + m_out * self.metadata[l]
                # Update m and n
                if m_qc == 2 * l:
                    m_qc = 0
                    if valence_only:
                        dst_n_current += 1
                    if dst_n_current > self.metadata[l]:
                        raise ValueError(
                            f"Number of shells exceeding the metadata: element={element}, l={l}, n={dst_n_current} "
                        )
                else:
                    m_qc += 1
                l_last = l
            # 
            padded_3idx_flat = torch.zeros(
                self.n_out, self.num_reps_1d ** 2, dtype=src_3idx.dtype 
            )
             # Scatter interleaved idx coefficients to the padded 2d layout
            dst_flat_ids_2d = (repid_map.unsqueeze(1) * self.num_reps_1d + repid_map.unsqueeze(0)).view(-1)
            padded_3idx_flat.index_add_(1, dst_flat_ids_2d, src_3idx.flatten(1, 2))
            reduction_coefficients[self.atomic_number_map[element]] = padded_3idx_flat.transpose(0, 1)

        self.register_buffer("reduction_coefficients", reduction_coefficients, persistent=False)

    def forward(self, atomic_numbers, feat_ten:torch.Tensor):
        assert feat_ten.dim() == 4 
        assert feat_ten.shape[1:3] == (
            self.num_reps_1d, self.num_reps_1d 
        )
        out = torch.zeros(
            feat_ten.shape[0], feat_ten.shape[3], self.n_out, device=feat_ten.device, dtype=feat_ten.dtype 
        )
        for ele in self.possible_atomic_numbers:
            elemask = atomic_numbers == ele
            out[elemask] = feat_ten[elemask].flatten(1, 2).transpose(1, 2).matmul(self.reduction_coefficients[ele])
        return out.transpose(1, 2)

