"""
Modified Gate Activation defined in Equiformer.
Build upon torch_gauge. 
"""

import torch 
import torch.nn as nn 

from torch_gauge.o3 import SphericalTensor, O3Tensor 
from BasicUtility.ActivationFunc import activation_getter 

class Gate(nn.Module):
    def __init__(self, metadata:torch.LongTensor, act_scalar="silu", act_invariant="sigmoid"):
        """
        Args:
            act_scalar: non-linear activation function type for l=0 scalar features in input SphericalTensor.
            act_invariant: non-linear activation function type for higher-order tensor invariants. 
        """
        self._act1 = act_scalar 
        self._act2 = act_invariant 
        self.act1 = activation_getter(self._act1)
        self.act2 = activation_getter(self._act2) 
        self._metadata = metadata 
        self.num_l0 = metadata[0].item()
        self.scalar_mask = (SphericalTensor.generate_rep_layout_1d_(self._metadata)[0] == 0)
    
    def forward(self, x:SphericalTensor):
        assert len(x.rep_dims) == 1
        assert torch.all(self._metadata, x.metadata[0]) 
        irrep_dim = x.rep_dims[0]

        x_inv = x.invariant() 
        gate_out = x.scalar_mul(self.act2(x_inv)).ten.view(-1, *x.shape[irrep_dim:])
        x_inv = x_inv.view(-1, x.num_channels, *x.shape[irrep_dim+1:])
        gate_out[:1, self.scalar_mask, :] = self.act1(x_inv)[:1, :self.num_l0, :]

        return x.self_like(gate_out.view(*x.shape))





