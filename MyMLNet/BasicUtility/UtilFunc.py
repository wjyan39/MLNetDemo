import math 
import re 

import numpy as np 
import torch 
from torch_scatter import scatter 

# Constant Global 
# device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
cpu_device = torch.device('cpu') 
# constants 
hartree2ev = 27.2114 
kcal2ev = 1/23.06035
floating_type = torch.float32

atom_U0_ref = {1:-0.500273, 6:-37.846772, 7:-54.583861, 8:-75.064579, 9:-99.718730} 

mae_fn = torch.nn.L1Loss(reduction='mean')
mse_fn = torch.nn.MSELoss(reduction='mean') 


def softplus_inverse(x):
    return torch.log(-torch.expm1(-x)) + x 

def _cutoff_fn(R_ij:torch.tensor, cutoff):
    x = R_ij / cutoff 
    x3 = x ** 3 
    x4 = x3 * x 
    x5 = x4 * x 
    res = 1 - 6 * x5 + 15 * x4 - 10 * x3
    return res 

def _correct_q(q_pred:torch.Tensor, N, atom_to_mol_batch, q_ref): 
    """]
    calculate corrected tilte_q in PhysNet 
    :param q_pred: partial charge predicted by PhysNet
    :return: corrected q, shape(-1, 1) 
    """
    Q_pred = scatter(redude='add', scr=q_pred, index=atom_to_mol_batch, dim=0)
    correct_q = (Q_pred - q_ref) / (N.type(torch.float32).to(device)) 
    broadcast_q = correct_q.take(atom_to_mol_batch) 
    return q_pred - broadcast_q 

def _chi_ij(R_ij, cutoff): 
    res = _cutoff_fn(2*R_ij, cutoff) / torch.sqrt(torch.mul(R_ij, R_ij)+1) + \
        (1 - _cutoff_fn(2*R_ij, cutoff)) / R_ij 
    return torch.where(R_ij != -1, res, torch.zeros_like(R_ij)) 
  
def cal_coulomb_E(qi:torch.Tensor, edge_dist, edge_index, cutoff, q_ref, N, atom_mol_batch):
    cutoff = cutoff.to(device) 
    if q_ref is not None:
        assert N is not None 
        assert atom_mol_batch is not None 
        qi = _correct_q(qi, N, atom_mol_batch, q_ref) 
    q_one = qi.take(edge_index[0, :]).view(-1, 1) 
    q_two = qi.take(edge_index[0, :]).view(-1, 1) 
    dist = _chi_ij(R_ij=edge_dist, cutoff=cutoff) 
    coulomb_E_atom = q_one * dist * q_two 
    coulomb_E = scatter(reduce='add', scr=coulomb_E_atom.view(-1), index=edge_index[0,:], dim_size=qi.shape[0], dim=0) 
    return (coulomb_E / 2).to(device) 

def get_model_params(model:torch.nn.Module, logger=None):
    res = ''
    for name, param in model.named_parameters():
        if logger is not None:
            logger.info('{}: {}'.format(name, param.data.shape)) 
        res = res + '{}: {}\n'.format(name, param.data.shape) 
    return sum([x.nelement() for x in model.parameters()]), res 

matrix_to_index_map = {}

def _get_index_from_matrix(num, prev_num):
    """
    get the edge index compatible with torch_geometric message passing 
    """
    if num in matrix_to_index_map:
        return matrix_to_index_map[num] + prev_num 
    else:
        index = torch.LongTensor(2, num*num).to(device)
        index[0, :] = torch.cat([torch.zeros(num, device=device).long().fill_(i) for i in range(num)], dim=0) 
        index[1, :] = torch.cat([torch.arange(num, device=device).long() for i in range(num)], dim=0) 
        mask = (index[0,:] != index[1,:]) 
        matrix_to_index_map[num] = index[:, mask] 
        return matrix_to_index_map[num] + prev_num 

def cal_edge(R:torch.Tensor, N:torch.Tensor, prev_N:torch.Tensor, edge_index, cal_coulomb=False):
    """
    calculate edge distance from edge_index
    param R: atom coordinates  
    """
    if cal_coulomb:
        coulomb_index = torch.cat(
            [_get_index_from_matrix(num.item(), prev_num) for num, prev_num in zip(N, prev_N)], dim=-1
        )
        points_i = R[coulomb_index[0,:], :]
        points_j = R[coulomb_index[1,:], :] 
        coulomb_dist = torch.sum((points_i - points_j)**2, keepdim=True, dim=-1) 
        coulomb_dist = torch.sqrt(coulomb_dist) 
    else:
        coulomb_dist = coulomb_index = None 
    
    short_range_index = edge_index 
    points_i = R[edge_index[0,:], :] 
    points_j = R[edge_index[1,:], :] 
    short_dist = torch.sum((points_i - points_j)**2, keepdim=True, dim=-1) 
    short_dist = torch.sqrt(short_dist) 
    return coulomb_dist, coulomb_index, short_dist, short_range_index  

batch_pattern = {}

def _get_batch_pattern(batch_size:int, max_num:int):
    if batch_size in batch_pattern.keys():
        return batch_pattern[batch_size] 
    else:
        pattern = [ i // max_num for i in range(batch_size*max_num)] 
        batch_pattern[batch_size] = pattern 
        return pattern 

def get_batch(atom_map, max_num):
    batch_size = atom_map.shape[0] // max_num 
    pattern = _get_batch_pattern(batch_size, max_num) 
    return torch.LongTensor(pattern)[atom_map] 

import gc 
def get_tensors():
    result = {"set_init"} 
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tup = (obj.__hash__(), obj.size()) 
                result.add(tup) 
        except:
            pass 
    print("*" * 30) 
    return result  

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']