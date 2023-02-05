import torch 
from torch import nn
from torch_geometric.nn import MessagePassing 

from DimeNet.OutputLayer import OutputLayer 
from BasicUtility.BottomLayer import ResidualBlock 
from BasicUtility.ActivationFunc import activation_getter 
from BasicUtility.UtilFunc import floating_type, get_model_params 

class DimeNetMPN(MessagePassing):
    def __init__(self, num_bilinear, dim_msg, num_rbf, num_sbf, activation):
        super().__init__() 
        self.num_bilinear = num_bilinear
        self.dim_msg = dim_msg 
        self.lin_s = nn.Linear(dim_msg, dim_msg) 
        self.lin_t = nn.Linear(dim_msg, dim_msg) 
        self.lin_rbf = nn.Linear(num_rbf, dim_msg, bias=False) 
        self.lin_sbf = nn.Linear(num_sbf, num_bilinear, bias=False) 
        # bi-linear layer 
        W_bi_linear = torch.zeros(dim_msg, dim_msg, num_bilinear).type(floating_type).uniform_(-1/dim_msg, 1/dim_msg) 
        self.register_parameter('W_bi_linear', torch.nn.Parameter(W_bi_linear, requires_grad=True))
        
        self.activation = activation_getter(activation) 

    def message(self, x_j, rbf_j, edge_attr):
        
        x_j = self.activation(self.lin_s(x_j))
        rbf = self.lin_rbf(rbf_j) 

        tmp_msg = rbf * x_j 

        sbf = self.lin_sbf(edge_attr) 

        msg = torch.einsum("wi, ijl, wl -> wj", tmp_msg, self.W_bi_linear, sbf) 
        
        return msg 
    
    def update(self, aggr_out, x):
        x = self.activation(self.lin_t(x)) 
        return x + aggr_out 
    
    def forward(self, x, edge_index, rbf, sbf):
        return self.propagate(edge_index, x=x, rbf=rbf, edge_attr=sbf) 


class DimeModule(nn.Module):
    
    def __init__(self, num_rbf, num_sbf, dim_msg, num_output, num_res_inter, num_res_msg, num_lin_out, dim_bi_linear, activation):
        
        super().__init__() 
        
        self.activation = activation_getter(activation) 
        msg_gate = torch.zeros(1, dim_msg).fill_(1.).type(floating_type) 
        self.register_parameter('gate', nn.Parameter(msg_gate, requires_grad=True)) 

        self.message_passing = DimeNetMPN(num_bilinear=dim_bi_linear, dim_msg=dim_msg, num_rbf=num_rbf, num_sbf=num_sbf, activation=activation) 

        self.num_res_inter = num_res_inter 
        for i in range(num_res_inter):
            self.add_module('res_interaction{}'.format(i+1), ResidualBlock(num_features=dim_msg, activation=activation)) 
        
        self.lin_interact_msg = nn.Linear(dim_msg, dim_msg) 

        self.num_res_msg = num_res_msg
        for i in range(num_res_msg):
            self.add_module('res_msg{}'.format(i+1), ResidualBlock(num_features=dim_msg, activation=activation)) 
        
        self.output_layer = OutputLayer(feature_dim=dim_msg, dim_rbf=num_rbf, n_output=num_output, num_lin=num_lin_out, activation=activation) 

    def forward(self, msg_ji, rbf_ji, sbf_kji, msg_egde_index, atom_egde_index):
        
        reserved_msg_ji = self.gate * msg_ji 
        
        mji = self.message_passing(msg_ji, msg_egde_index, rbf_ji, sbf_kji) 

        for i in range(self.num_res_inter):
            mji = self._modules['res_interaction{}'.format(i+1)](mji) 
        
        mji = self.activation(self.lin_interact_msg(mji))

        mji = mji + reserved_msg_ji  

        for i in range(self.num_res_msg):
            mji = self._modules['res_msg{}'.format(i+1)](mji) 
        
        out = self.output_layer(mji, rbf_ji, atom_egde_index) 

        return mji, out 
    
if __name__ == '__main__':
    model = DimeModule(6, 36, 128, 12, 1, 2, 3, 8, 'swish')
    print(get_model_params(model) - get_model_params(model.output_layer))
