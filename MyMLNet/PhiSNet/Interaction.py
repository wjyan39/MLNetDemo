import torch 
import torch.nn as nn 

from BasicUtility.O3.O3Tensor import SphericalTensor, O3Tensor 
from BasicUtility.O3.SphericalHamornics import RSHxyz
from PhiSNet.Mixing import PairMixing
from PhiSNet.SphLinear import SphLinear, ScaLinear 
import torch_geometric

class MessagePassingBlock(torch_geometric.nn.MessagePassing):
    def __init__(self, metadata:torch.LongTensor, envelop, num_features:int, aggr, group="so3"):
        flow = 'source_to_target' 
        super().__init__(aggr=aggr, flow=flow) 
        order_out = metadata.shape[0]
        self.pairmix = PairMixing(envelop=envelop, filter_channels=num_features, metadata_1=metadata, metadata_2=metadata, order_out=order_out, group=group)
        self.envelop = envelop
        rshs_in = torch.ones(order_out).long()
        self.sphlin = SphLinear(rshs_in, metadata, order_out, group=group) 
        self.scalin = ScaLinear(num_features=num_features, target_metadata=metadata, group=group)
        self.num_features = num_features

    def message(self, x_j, edge_attr):
        geom_filter = self.scalin(self.envelop(edge_attr[..., 1]))
        broadcast_rsh = self.sphlin(edge_attr[..., 0])
        msg_a_ten =  x_j.ten * geom_filter.ten * broadcast_rsh.ten
        msg_b = self.pairmix(x_j, broadcast_rsh, edge_attr[..., 0]) 
        msg_ten = msg_a_ten + msg_b.ten   
        return x_j.self_like(msg_ten)
    
    def update(self, aggr_out, x):
        return x.self_like(x.ten + aggr_out.ten)
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
         

class InteractBlock(nn.Module):
    """
    The Interaction Block in PhiSNet with little modification
    """
    def __init__(self, metadata_in:torch.LongTensor, envelop, num_features, group="so3") -> None:
        super().__init__()
        self.message_passing = MessagePassingBlock(metadata=metadata_in, )
    
    def forward(self, x, edge_index, edge_attr):
        msg_x = self.message_passing(x, edge_index, edge_attr) 
        return msg_x 
