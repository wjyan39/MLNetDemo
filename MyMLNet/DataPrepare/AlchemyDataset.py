import torch 
from torch_geometric.data import Data, InMemoryDataset 

import numpy as np 
import pandas as pd 
import pickle 
from pathlib import Path 

from BasicUtility.UtilFunc import cal_edge, remove_bonding_edge, floating_type


class AlchemyData(InMemoryDataset):

    def __init__(self, mode="dev", net_version="atom", root=".", train_csv_path=None, transform=None, pre_transform=None, cutoff=None, atom_bond_sep=True):
        self.mode = mode 
        self.net = net_version
        self.root_path = Path(root) 
        self.train_csv = train_csv_path 
        self.cutoff = cutoff 
        self.atom_bond_sep = atom_bond_sep
        self.target = NotImplemented 

        super().__init__(root=root, transform=transform, pre_transform=pre_transform) 
        self.data, self.slices = torch.load(self.processed_paths[0]) 
    
    @property 
    def raw_file_names(self):
        return self.mode + ".dat" 
    
    @property 
    def processed_file_names(self):
        return self.net + "-" + self.mode + ".pt" 
    
    def download(self):
        pass 
    
    def process(self):
        
        if self.train_csv is not None:
            # self.target = pd.read_csv(self.train_csv, index_col=0, usecols=['gdb_idx',] + ['property_{}'.format(x) for x in range(12)]) 
            # self.target = self.target[['property_{}'.format(x) for x in range(12)]] 
       		# currently, only take the first property label, energy  
            self.target = pd.read_csv(self.train_csv, index_col=0, usecols=['gdb_idx', 'property_0']) 
            self.target = self.target[['property_0']] 
		
        dat_file = self.raw_paths[0] 
        with open(dat_file, "rb") as dat:
            dat_dict = pickle.load(dat) 

		# list of data: torch_geometric.data.Data 
        data_list = [] 
		
        for entry in dat_dict:
			# construct the Data object 
            _tmp_Data = Data()
			# from dictionary  			
            dat_atom = dat_dict[entry]
			# target
            target = torch.as_tensor(self.target.loc[entry].tolist(), dtype=floating_type) if self.target is not NotImplemented else torch.as_tensor([entry], dtype=floating_type)
 
            _tmp_Data.N = torch.LongTensor(dat_atom["natm"]).view(-1)
            _tmp_Data.R = torch.DoubleTensor(dat_atom["atm_coords"]).view(-1,3) 
            _tmp_Data.Z = torch.DoubleTensor(dat_atom["atm_charges"]).view(-1) 
            _tmp_Data.E = target.view(-1) 

            if self.atom_bond_sep:
                _tmp_Data.bond_edge_index = torch.LongTensor(dat_atom["bonding_edges"])

            # pre-transform, generate edge_index accordingly 

            _tmp_Data = self.pre_transform(data=_tmp_Data, cutoff=self.cutoff, atom_bond_sep=self.atom_bond_sep)
 
            data_list.append(_tmp_Data)	

        print("collating...")
        data, slices = self.collate(data_list)
        print("saving...")
        torch.save((data, slices), self.processed_paths[0]) 


def custom_pre_transform(self, data, cutoff, atom_bond_sep):
    edge_index = torch.zeros(2, 0).long() 
    dist, full_edge, _, _, = cal_edge(R=data.R, N=data.N, prev_N=torch.Tensor([0]), edge_index=edge_index, cal_coulomb=True)
    dist, full_edge = dist.cpu(), full_edge.cpu() 

    if cutoff is not None:
        data.bn_edge_index = full_edge[:, (dist < cutoff).view(-1)] 
    else:
        data.bn_edge_index = full_edge 

    if atom_bond_sep:
        data.nonbond_edge_index = remove_bonding_edge(data.bn_edge_index, data.bond_edge_index) 
    
    for bond_type in ['bn', 'bond', 'nonbond']:
        _edge_index = getattr(data, bond_type + '_edge_index', False)
        if _edge_index is not False:
            setattr(data, 'num_' + bond_type + '_edge', torch.zeors(1).long() + _edge_index.shape[-1]) 

    return data  		





