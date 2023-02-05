import torch 
import torch.nn as nn 
from torch.nn import functional as F 
from torch_scatter import scatter 

import math 

from PhysNet.PhysModule import PhysModule 
from BasicUtility.BottomLayer import EmbeddingLayer 
from BasicUtility.BesselExpansion import rbf_expansion 
from BasicUtility.UtilFunc import softplus_inverse 

class PhysNetDemo(nn.Module):
    """
    Simple Demo of PhysNet (O. Unke, M. Meuwly J. Chem. Theory Comput. 2019, 15, 3678) 
    using Pytorch and Pytorch_geometric (PyG) APIs.
    The Pytorch implementation of sPhysNet (Lu et.al. J. Chem. Inf. Model 2021, 61, 1095) 
    is acknowleged. 
    """
    
    def __init__(self, n_atom_embed, num_features, num_output, cut_off_radius, num_kernel, 
        num_modules, num_res_atom, num_res_interact, num_res_out, num_lin_out, activation, 
        dropout=False, debug_mode=False):
        """
        :param n_atom_embed: number of embedding atoms 
        :param num_features: input feature size of the node 
        :param num_output: size of output prediction 
        :param num_kernel: number of channels used in gaussian expansion of distance matrix 
        :param cut_off_radius 
        :param num_modules: number of PhysModule in PhysNet architecture  
        :param num_res_atom: number of residual layers in PhysModule 
        :param num_res_interact: number of residual layers in interaction block of each PhysModule 
        :param num_res_out: number of residual layers in Output Layer of each PhysModule 
        :param num_lin_out: number of linear layers in Output layer of each PhysModule 
        :activation: String specifies activation function
        :param dropout: Boolean for whether dropout operation would be taken at every linear transform  
        :param debug_mode: Boolean 
        """
        super().__init__() 
        self.num_output = num_output 
        self.num_modules = num_modules
        self.num_res_atom = num_res_atom 
        self.num_res_interact = num_res_interact 
        self.num_res_out = num_res_out 
        self.cut_off_radius = cut_off_radius 
        self.debug_mode = debug_mode
        # self.cut_off = True if self.cut_off_radius > 0.0 else False 
        
        # registering params for distance expansion 
        # cut off radius 
        cut_off_radius = torch.as_tensor(cut_off_radius).type(torch.float32) 
        self.register_parameter('cutoff', nn.Parameter(cut_off_radius, requires_grad=False)) 
        # centers in gaussian expansion, learnable 
        centers = softplus_inverse(torch.linspace(1.0, math.exp(-cut_off_radius), num_kernel)) 
        centers = F.softplus(centers) 
        self.register_parameter('centers', nn.Parameter(centers, requires_grad=True)) 
        # widths, learnable 
        widths = [softplus_inverse((0.5 / ((1.0 - torch.exp(-cut_off_radius)) / num_kernel))**2)] * num_kernel 
        widths = torch.as_tensor(widths).type(torch.float32) 
        self.register_parameter('widths', nn.Parameter(widths, requires_grad=True)) 

        self.dist_calculator = nn.PairwiseDistance(keepdim=1) 
        
        # PhysNet Building Blocks 
        # Embedding layer
        self.embedding_layer = EmbeddingLayer(num_embeddings=n_atom_embed, out_features=num_features) 
        # registering main modules 
        for idx in range(self.num_modules):
            this_module = PhysModule(num_featrues = num_features, num_kernel = num_kernel, 
                            dim_output = num_output, num_res_atomic = num_res_atom, 
                            num_res_interact = num_res_interact, num_res_output = num_res_out,
                            num_lin_out = num_lin_out, activation = activation, dropout = dropout) 
            self.add_module('PhysModule{}'.format(idx+1), this_module) 
        # final atom-wise read-out 
        shift_matrix = torch.zeros(n_atom_embed, num_output).type(torch.float32) 
        scale_matrix = torch.zeros(n_atom_embed, num_output).type(torch.float32).fill_(1.0) 
        self.register_parameter('scale', nn.Parameter(scale_matrix, requires_grad=True)) 
        self.register_parameter('shift', nn.Parameter(shift_matrix, requires_grad=True)) 

    def forward(self, data):
        # data is object hiechracy from torch_geometric.data 
        N = data.N # number of atoms in the molecule 
        R = data.R # molecular coordinates (N, 3) 
        Z = data.Z # atom charge used for embedding 
        Q = data.Q # additional charge 

        # assigning atom to molecule 
        atom_mol_batch = data.atom_mol_batch  
        edge_index = data.edge_index 

        # dist matrix and rbf expansion 
        distance = self.dist_calculator(R[edge_index[0,:],:], R[edge_index[1,:],:]) 
        expansions = rbf_expansion(D=distance, centers=getattr(self,'centers'), widths=getattr(self,'widths'), cutoff=getattr(self,'cutoff')) 

        # Step 1         
        vi = self.embedding_layer(Z) 
        # Step 2 main modules  
        seperated_out_sum = 0.
        for idx in range(self.num_modules):
            this_module = self._modules['PhysModule{}'.format(idx+1)] 
            vi, cur_out = this_module(vi, edge_index, expansions) 
            seperated_out_sum += cur_out 
        # Step 3 atom-wise read out 
        atom_out = getattr(self,'scale')[Z,:] * seperated_out_sum + getattr(self, 'shift')[Z,:] 
        mol_pred_properties = scatter(reduce='add', scr=atom_out, index=atom_mol_batch, dim=0) 

        Q_pred, D_pred, F_pred = 0., 0., 0. 
        if self.num_output > 1: # require predicting properties other than energy 
            # atomic charge distribution 
            Q_pred = mol_pred_properties[:, -1] 
            Q_atom = atom_out[:, -1] 
            D_atom = Q_atom.view(-1, 1) * R 
            D_pred = scatter(reduce='add', scr=D_atom, index=atom_mol_batch, dim=0) 
        # molecular energy 
        E_pred = mol_pred_properties[:, 0]
        output = (E_pred, F_pred, Q_pred, D_pred) 
        if self.debug_mode:
            output = (*output, vi, atom_mol_batch, Z) 
        return output 


        


        




