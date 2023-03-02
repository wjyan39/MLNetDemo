"""
Utilities for converting AO basis Quamtum Chemistry calculated 
feature matrices to machine learning convention, compatible 
with SphericalTensor data structure.
"""

import torch 
import torch.nn as nn 
from BasicUtility.O3.O3Tensor import SphericalTensor, O3Tensor

from pathlib import Path 

class OneBodyToSpherical(nn.Module):
    def __init__(self, metadata, rep_dim):
        super.__init__()
        self.rep_dim = rep_dim 
        self.metadata = torch.LongTensor(metadata)
        offset = 0
        out_ids = []
        for l, n in enumerate(self.metadata):
            in_idx = torch.arange(offset, offset + n* (2*l+1)).long() 
            out_idx =  in_idx.view(-1, 2*l + 1).t().flip(0).contiguous().view(-1) 
            out_ids.append(out_idx) 
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
        
class TwoBodyToSpherical(nn.Module):
    def __init__(self, metadata, rep_dims, basis_layout, valence_only=True):
        """
        Converting two-body features (order 2 tensor) loaded from QC Calculation to torch_gauge's 2d SphericalTensor.
        The returned 2d-SO(3) tensor is of symmetric layout along the rep_dims.
        
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
        