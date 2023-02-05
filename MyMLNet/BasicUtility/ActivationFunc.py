from keyword import softkwlist
from turtle import forward
import torch 
from torch import nn as nn 
from torch.nn import functional as F 
import numpy as np 
import math 


def _shifted_soft_plus(x:torch.tensor):
    """
    activation function employed in PhysNet 
    sigma(x) = log(exp(x)+1) - log(2)
    :param x 
    :return nn.functional 
    """
    return F.softplus(x) - torch.Tensor([np.log(2)]).type(x.type()) 


class ShiftedSoftPlus(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return _shifted_soft_plus(x) 


class Swish(nn.Module):
    """
    Swish activation 
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x) 


class ReLU(nn.Module):
    """
    ReLU activation 
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.relu(x)  


activation_func_config = {
    'swish': Swish(),
    'ssp': ShiftedSoftPlus(),
    'relu': ReLU(),
    'silu': Swish(),
}


def activation_getter(act_required):
    if act_required.lower() in activation_func_config.keys():
        return activation_func_config[act_required.lower()] 
    else:
        raise NotImplementedError("Required activation function not yet implemented.")

