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
            act_invariant: non-linear activation function type.
        """
        self._act = act_invariant 
        self.act = activation_getter(self._act) 
        
    
    def forward(self, x:SphericalTensor):

        x_inv = x.invariant() 
        gate_out = x.scalar_mul(self.act(x_inv))
    
        return gate_out


class Gate(nn.Module):
    def __init__(self, num_features:int, activation="silu"):
        """
        Args:
            num_features: number of invariant feature channels. 
            activation: non-linear activation type.
        """
        self.channels = num_features 
        self.act = activation_getter(activation) 
        self.lin = nn.Linear(num_features, num_features) 
    
    def forward(self, x:SphericalTensor):
        assert len(x.num_channels) == 1 
        assert x.num_channels[0] == self.channels 
        
        x_inv = x.invariant() 
        gate_out = x.scalar_mul(self.act(self.lin(x_inv))) 
        
        return gate_out 


