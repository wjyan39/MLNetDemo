import torch 
import torch.nn as nn 
from math import ceil 

from BasicUtility.BottomLayer import ResidualBlock 
from BasicUtility.ActivationFunc import activation_getter 
from PhysNet.Interaction import InteractionModule 

class _OutputLayer(nn.Module):
    """
    Output layer of PhysNet 
    """
    def __init__(self, num_features, output_dim, num_res, num_lin_out, activation, dropout=False):
        super().__init__() 
        self.num_res = num_res 
        self.num_lin_out = num_lin_out
        self.activation = activation_getter(activation) 

        for idx in range(num_res):
            self.add_module( 
                'res_layer' + str(idx), 
                ResidualBlock(num_features=num_features, activation=activation, dropout=dropout) 
            )
        
        last_dim = num_features 
        for idx in range(num_lin_out):
            this_dim = ceil(last_dim / 2)
            read_out_i = nn.Linear(last_dim, this_dim) 
            last_dim = this_dim 
            self.add_module('read_out{}'.format(idx), read_out_i) 
            # TODO Batch Norm 
        
        self.lin = nn.Linear(last_dim, output_dim, bias=False) 
        self.lin.weight.data.zero_() 

    def forward(self, x):
        tmp_res = x 
        # residual blocks 
        for i in range(self.num_res):
            tmp_res = self._modules['res_layer' + str(i)](tmp_res) 
        out = tmp_res 
        # muliti-linear output layers 
        for i in range(self.num_lin_out):
            a = self.activation(out) 
            out = self._modules['read_out{}'.format(i)](a) 
        # final output 
        out = self.activation(out) 
        out = self.lin(out) 
        return out 


class PhysModule(nn.Module):
    """ 
    PhysNet Main Module  
    """
    def __init__(self, num_featrues, num_kernel, dim_output, 
                num_res_atomic, num_res_interact, num_res_output, 
                num_lin_out, activation, dropout=False):
        super().__init__()
        # Interaction Block 
        self.interaction = InteractionModule(num_features=num_featrues, kernel_channels=num_kernel, num_res_block=num_res_interact,
                                    activation=activation, dropout=dropout) 
        # Atomic Residual layer 
        self.num_res_atomic = num_res_atomic 
        for idx in range(num_res_atomic):
            self.add_module(
                'res_layer' + str(idx), 
                ResidualBlock(num_features=num_featrues, activation=activation, dropout=dropout) 
            ) 
        # Output 
        self.output = _OutputLayer(num_features=num_featrues, output_dim=dim_output,
                            num_res=num_res_output, num_lin_out=num_lin_out, 
                            activation=activation, dropout=dropout)

    def forward(self, x, edge_index, edge_attr): 
        interacted_x, _ = self.interaction(x, edge_index, edge_attr)
        tmp_res = interacted_x.type(torch.float) 
        for i in range(self.num_res_atomic):
            tmp_res = self._modules['res_layer' + str(i)](tmp_res) 
        out_res = self.output(tmp_res) 
        # return updated x for next module if exists and out_res as output at this module 
        return tmp_res, out_res 


