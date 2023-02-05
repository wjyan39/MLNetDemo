import numpy as np 
import pandas as pd 
import torch 
import torch_geometric

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset 

import os 
from typing import List 
from tqdm import tqdm  


def scale_R(R):
    abs_min = torch.abs(R).min() 
    while abs_min < 1e-3:
        R = R - 1
        abs_min = torch.abs(R).min() 
    return R 

def sort_edge(edge_index):
    arg_sort = torch.argsort(edge_index[1, :]) 
    return edge_index[:, arg_sort]

def remove_bonding_edge(full_edge_index, bond_edge_index):
    """
    remove bonding index from full_edge_index
    """ 
    mask = torch.zeros(full_edge_index.shape[-1]).bool().fill_(False) 
    len_bonding = bond_edge_index.shape[-1] 
    for i in range(len_bonding):
        overlap_edge = (full_edge_index == bond_edge_index[:, i].view(-1,1)) 
        mask += (overlap_edge[0] & overlap_edge[1]) 
    other_bond = ~mask 
    return full_edge_index[:, other_bond] 


def sdf_mol_to_edge_index(mol):
    """
    calculate edge_index from rdkit.mol 
    """
    bonds = mol.GetBonds() 
    num_bonds = len(bonds) 
    _edge_index = torch.zeros(2, num_bonds).long() 
    for bond_id, bond in enumerate(bonds):
        _edge_index[0, bond_id] = bond.GetBeginAtomIdx() 
        _edge_index[1, bond_id] = bond.GetEndAtomIdx() 
    _edge_index_inv = _edge_index[[1,0], :] 
    _edge_index = torch.cat([_edge_index, _edge_index_inv], dim=-1) 
    return _edge_index.tolist()  

def concat_im_datasets(root: str, datasets: List[str], out_name: str):
    data_list = []
    for dataset in datasets:
        dummy_dataset = DummyIMDataset(root, dataset)
        for i in tqdm(range(len(dummy_dataset)), dataset):
            data_list.append(dummy_dataset[i])
    print("saving... it is recommended to have 32GB memory")
    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list),
               os.path.join(root, "processed", out_name)) 


class DummyIMDataset(InMemoryDataset):
    
    def __init__(self, root, dataset_name, split=None, **kwargs):
        self.dataset_name = dataset_name 
        self.split = split 
        super().__init__(root, None, None) 
        self.data, self.slice = torch.load(self.processed_paths[0]) 
        self.train_index, self.val_index, self.test_index = None, None, None 
        if split is not None:
            split_data = torch.load(self.processed_paths[1]) 
            train_index = split_data["train_index"] 
            perm_matrix = torch.randperm(len(train_index)) 
            self.train_index = train_index[perm_matrix[:-1000]] 
            self.val_index = train_index[perm_matrix[-1000:]] 
            self.test_index = split_data["test_index"] 
    
    @property 
    def raw_file_names(self):
        return ["dummy"] 
    
    @property 
    def processed_file_names(self):
        return [self.dataset_name, self.split] if self.split is not None else [self.dataset_name] 
    
    def download(self):
        pass 
    
    def process(self):
        pass 


def name_extender(name:str, edge_type=None, sort_edge=False, cutoff=None, geometry="QM", extra=None):
    
    if edge_type == 'cutoff':
        if cutoff is None:
            print('cutoff canot be None when edge version == cutoff, exiting...')
            exit(-1)
        name += '-cutoff-{:.2f}'.format(cutoff) 
    else:
        name += '-nocut' 

    if sort_edge:
        name += '-sorted' 
    
    name += geometry 

    return name 
