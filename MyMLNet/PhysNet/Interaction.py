from turtle import forward
import torch 
from torch import nn as nn 
from torch.nn import functional as F 
from torch import Tensor 
import torch_geometric 
from torch_sparse import SparseTensor 

from BasicUtility.BottomLayer import ResidualBlock 
from BasicUtility.ActivationFunc import activation_getter 


class MessagePassingBlock(torch_geometric.nn.MessagePassing):
    """
    message passing in PhysNet 
    """
    def __init__(self, num_features, kernel_channels, activation, aggr, dropout=False):
        flow = 'source_to_target' 
        super().__init__(aggr=aggr, flow=flow) 
        
        self.lin_vertex = nn.Linear(num_features, num_features) 
        nn.init.xavier_uniform_(self.lin_vertex.weight.data)
        self.lin_vertex.bias.data.zero_() 
        
        self.lin_edge = nn.Linear(num_features, num_features) 
        nn.init.xavier_uniform_(self.lin_edge.weight.data) 
        self.lin_edge.bias.data.zero_() 

        self.G = nn.Linear(kernel_channels, num_features, bias=False) 
        self.G.weight.data.zero_() 

        self.activation = activation_getter(activation) 
    
    def message(self, x_j, edge_attr):
        msg = self.lin_edge(x_j) 
        msg = self.activation(msg) 
        masked_edge_attr = self.G(edge_attr) 
        msg = torch.mul(msg, masked_edge_attr) 
        return msg 
    
    def update(self, aggr_out, x):
        a = self.activation(self.lin_vertex(x)) 
        return a + aggr_out 
    
    def forward(self, x, edge_index, edge_attr):
        x = self.activation(x) 
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) 
    

class InteractionModule(nn.Module):
    """
    Interaction Module in PhysNet 
    """
    def __init__(self, num_features, kernel_channels, num_res_block, activation, dropout=False):
        super().__init__() 
        u = Tensor(1, num_features).type(torch.double).fill_(1.) 
        self.register_parameter('u', torch.nn.Parameter(u,requires_grad=True)) # make u tensor learnable 

        self.message_passing = MessagePassingBlock(num_features, kernel_channels, aggr='add', activation=activation) 

        # message passing layer
        self.message_passing_block = NotImplemented

        # N residual blocks 
        self.num_res_block = num_res_block 
        for idx in range(num_res_block):
            self.add_module('res_layer' + str(idx), ResidualBlock(num_features=num_features, activation=activation, dropout=dropout)) 
        
        # TODO
        # Batch Normalization 
        
        # interaction output layer 
        self.lin_output = nn.Linear(num_features, num_features) 
        nn.init.xavier_uniform_(self.lin_output.weight.data) 
        self.lin_output.bias.data.zero_() 
        
        self.activation = activation_getter(activation) 

    def forward(self, x, edge_index, edge_attr):
        msg_x = self.message_passing(x, edge_index, edge_attr) 
        tmp_res = msg_x 
        for idx in range(self.num_res_block):
            tmp_res = self._modules['res_layer' + str(idx)](tmp_res) 
        v = self.activation(tmp_res) 
        v = self.lin_output(tmp_res)
        return v + torch.mul(x, self.u), msg_x 
    

    