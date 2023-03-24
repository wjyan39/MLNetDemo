"""
Modified Gate Activation defined in Equiformer.
Build upon torch_gauge. 
"""

import torch 
import torch.nn as nn 

from torch_gauge.o3 import SphericalTensor, O3Tensor 
from BasicUtility.ActivationFunc import activation_getter 

class SigmoidGate(nn.Module):
    def __init__(self, act_invariant="sigmoid"):
        """
        Args:
            act_scalar: non-linear activation function type for l=0 scalar features in input SphericalTensor.
            act_invariant: non-linear activation function type for higher-order tensor invariants. 
        """
        self._act = act_invariant 
        self.act = activation_getter(self._act) 
        
    
    def forward(self, x:SphericalTensor):

        x_inv = x.invariant() 
        gate_out = x.scalar_mul(self.act(x_inv))
    
        return gate_out





