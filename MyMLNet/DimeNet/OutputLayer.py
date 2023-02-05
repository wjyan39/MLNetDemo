from turtle import forward
import torch 
from torch import nn 
from torch_geometric.nn import MessagePassing 
from torch_scatter import scatter 

from BasicUtility.ActivationFunc import activation_getter 

class _MPNScatter(MessagePassing):

    def __init__(self):
        super().__init__() 
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr) 
    
    def message(self, x_j, edge_attr):
        return edge_attr 
    
    def update(self, aggr_out, x):
        return aggr_out + x 

class OutputLayer(torch.nn.Module):
    
    def __init__(self, feature_dim, dim_rbf, n_output, num_lin, activation):
        super.__init__() 
        self.feature_dim = feature_dim 
        self.dim_rbf = dim_rbf 
        self.activation = activation_getter(activation) 
        
        self.num_lin = num_lin 
        for idx in range(num_lin):
            self.add_module('dense{}'.format(idx+1), nn.Linear(feature_dim, feature_dim)) 
        
        self.lin_rbf = nn.Linear(dim_rbf, feature_dim, bias=False)
        
        self.scatter_fn = _MPNScatter() 

        self.out_lin = nn.Linear(feature_dim, n_output, bias=False) 
        self.out_lin.weight.data.zero_() 

    def forward(self, m_ji, rbf_ji, atom_edge_index):
        e_ji = self.lin_rbf(rbf_ji) 
        message_ji = e_ji * m_ji  
        
        atomistic_i = scatter(reduce='add', scr=message_ji, index=atom_edge_index[1,:], dim=-2) 
        
        for idx in range(self.num_lin):
            atomistic_i = self._modules['dense{}'.format(idx+1)](atomistic_i) 
            atomistic_i = self.activation(atomistic_i) 

        out = self.out_lin(atomistic_i) 
        
        return out 

if __name__ == '__main__':
    from BasicUtility.UtilFunc import get_model_params
    model = OutputLayer(128, 6, 12, 3, 'ssp')
    print(get_model_params(model)) 
