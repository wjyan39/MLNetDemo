"""
Basic datastructure for tracking relational data in model.
"""
from dataclasses import dataclass 
from typing import Sequence 

import torch 
from BasicUtility.O3.O3Tensor import SphericalTensor
from BasicUtility.O3.O3Tensor import to_numpy as to_np
from BasicUtility.O3.O3Tensor import from_numpy as from_np 

@dataclass 
class VerletList:
    """
    Padded Verlet List class.
    """
    def __init__(self):
        self.neighbor_idx = None 
        self.ndata = {}
        self.edata = {}
        self.edge_mask = None 
        self.n_nodes = None 
        self.PADSIZE = None 
        self.batch_num_nodes = None 
        self._dst_edim_locators = None 
    
    def from_mask(self, verlet_mask:torch.BoolTensor, padding_size, num_nodes, one_body_data, two_body_data):
        self.PADSIZE = padding_size 
        self.n_nodes = num_nodes 
        self.batch_num_nodes = torch.LongTensor([self.n_nodes]) 
        # get in_degrees of each node, verlet_mask is of shape (n_nodes, n_nodes) with 0 or 1 vals, i.e. the adajacent matrix.
        in_degrees = verlet_mask.long().sum(dim=1)
        assert(in_degrees.max().item() <= self.PADSIZE), \
            f"The max node degree must be smaller than the padding size, while got max in-degrees = {in_degrees.max().item()} and PADSIZE = {self.PADSIZE}."
        # source neighboring list 
        src_raw = (
            torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).expand(num_nodes, num_nodes)[verlet_mask]
        )
        # scatter source node ids (N*N) to padded table (N*PADSIZE) 
        src_loactors_1d = torch.LongTensor(
            [
                dstid * self.PADSIZE + src_location 
                for dstid in range(self.n_nodes) 
                for src_location in range(in_degrees[dstid])
            ]
        )
        ## flattened neighboring edge index for each node in Padding table 
        src_edim_locators_flattened = torch.LongTensor(
            [
                src_location 
                for dstid in range(self.n_nodes)
                for src_location in range(in_degrees[dstid])
            ]
        )
        ## copy to 2d (N*N) form 
        src_edim_locators = torch.zeros(num_nodes, num_nodes, dtype=torch.long) 
        src_edim_locators[verlet_mask] = src_edim_locators_flattened 
        # final scatter 
        self.neighbor_idx = (
            torch.zeros(self.n_nodes * self.PADSIZE, dtype=torch.long)
            .scatter_(dim=0, index=src_loactors_1d, src=src_raw)
            .view(self.n_nodes, self.PADSIZE)
        )
        self.edge_mask = (
            torch.zeros(self.n_nodes * self.PADSIZE, dtype=torch.bool)
            .scatter_(dim=0, index=src_loactors_1d, src=torch.ones(src_raw.size(0), dtype=torch.bool))
            .view(self.n_nodes, self.PADSIZE) 
        )
        self._dst_edim_locators = (
            torch.zeros(self.n_nodes, self.PADSIZE, dtype=torch.long)
            .masked_scatter_(
                mask=self.edge_mask,
                source=src_edim_locators.t()[verlet_mask]
            )
            .view(self.n_nodes, self.PADSIZE) 
        )
        # node feature 
        self.ndata = one_body_data 
        # edge feature scattered to padded table, (n_node, PADSIZE, efeat_channels)
        self.edata = {
            k : self._scatter_efeat(edata, verlet_mask, src_raw)
            for k, edata in two_body_data.items()
        }
        return self 
    
    def _scatter_efeat(self, edata, verlet_mask, src_raw):
        if isinstance(edata, torch.Tensor):
            if not verlet_mask.any():
                return torch.zeros(
                    self.n_nodes, self.PADSIZE, *edata.shape[2:], dtype=edata.dtype 
                )
            return (
                torch.zeros(
                    self.n_nodes, self.PADSIZE, *edata.shape[2:], dtype=edata.dtype
                )
                .view(self.n_nodes, self.PADSIZE, -1)
                .masked_scatter_(
                    mask=self.edge_mask.unsqueeze(-1),
                    source=edata[verlet_mask, ...].view(src_raw.shape[0], -1)
                )
                .view(self.n_nodes, self.PADSIZE, *edata.shape[2:]) 
            )
        elif isinstance(edata, SphericalTensor):
            if not verlet_mask.any():
                return edata.self_like(
                    torch.zeros(self.n_nodes, self.PADSIZE, *edata.shape[2:], dtype=edata.ten.dtype)
                )
            out_ten = (
                torch.zeros(self.n_nodes, self.PADSIZE, *edata.shape[2:], dtype=edata.ten.dtype)
                .view(self.n_nodes, self.PADSIZE, -1)
                .masked_scatter_(
                    mask=self.edge_mask.unsqueeze(-1),
                    source=edata.ten[verlet_mask, ...].view(src_raw.shape[0], -1),
                )
                .view(self.n_nodes, self.PADSIZE, *edata.shape[2:])
            )
            return edata.self_like(out_ten) 
        else:
            raise TypeError 
    
    def query_src(self, src_feat):
        """
        Return src-node data scattered into neighbor-list frame.
        """
        flattened_neighbor_idx = self.neighbor_idx.view(-1) 
        if isinstance(src_feat, torch.Tensor):
            flattened_2d_src = src_feat.view(src_feat.shape[0], -1) 
            flattened_out = (
                flattened_2d_src[flattened_neighbor_idx, ...]
                .view(*self.neighbor_idx.shape, flattened_2d_src.shape[1])
                .mul_(self.edge_mask.unsqueeze(2))
            )
            return flattened_out.view(*self.neighbor_idx.shape, *src_feat.shape[1:]) 
        elif isinstance(src_feat, SphericalTensor):
            flattened_2d_src = src_feat.ten.view(src_feat.ten.shape[0], -1) 
            flattened_out_ten = (
                flattened_2d_src[flattened_neighbor_idx, ...]
                .view(*self.neighbor_idx.shape, flattened_2d_src.shape[1])
                .mul_(self.edge_mask.unsqueeze(2))
            )
            return src_feat.__class__(
                flattened_out_ten.view(*self.neighbor_idx.shape, *src_feat.shape[1:]),
                rep_dims=tuple(dim+1 for dim in src_feat.rep_dims),
                metadata=src_feat.metadata,
                rep_layout=src_feat.rep_layout,
                num_channels=src_feat.num_channels
            )
        else:
            raise TypeError 
    
    def to_src_first_view(self, data):
        """
        Flipping src / dst node indexing. For dense matrix (Completely Connected), reduces to transpose.
        Args:
            data :: torch.Tensor || SphericalTensor :: Data to be transposed, must be contiguos.
        """
        scatter_ten = (self.neighbor_idx * self.PADSIZE + self._dst_edim_locators)[self.edge_mask]
        if isinstance(data, torch.Tensor):
            out_ten = torch.zeros_like(data).view(self.n_nodes * self.PADSIZE, -1) 
            out_ten[scatter_ten, :] = data.view(self.n_nodes, self.PADSIZE, -1)[self.edge_mask, :] 
            return out_ten.view_as(data) 
        elif isinstance(data, SphericalTensor):
            out_ten = torch.zeros_like(data.ten).view(self.n_nodes * self.PADSIZE, -1)
            out_ten[scatter_ten, :] = data.ten.view(self.n_nodes, self.PADSIZE, -1)[self.edge_mask, :]
            return data.self_like(out_ten.view_as(data.ten)) 
        else:
            raise TypeError 
    
    def to(self, device):
        self.neighbor_idx = self.neighbor_idx.to(device) 
        self.ndata = {
            k : v.to(device) if v is not None else None for k, v in self.ndata.items()
        }
        self.edata = {
            k : v.to(device) if v is not None else None for k, v in self.edata.items()
        }
        self.edge_mask = self.edge_mask.to(device)
        self._dst_edim_locators = self._dst_edim_locators.to(device) 
        self.batch_num_nodes = self.batch_num_nodes.to(device) 
        return self 

    def to_numpy_dict(self):
        return {
            "PADSIZE": self.PADSIZE,
            "n_nodes": self.n_nodes,
            "neighbor_idx": self.neighbor_idx.numpy(),
            "edge_mask": self.edge_mask.numpy(),
            "_dst_edim_locators": self._dst_edim_locators.numpy(),
            "batch_num_nodes": self.batch_num_nodes.numpy(),
            "ndata": {nk: to_np(nv) for nk, nv in self.ndata.items()},
            "edata": {ek: to_np(ev) for ek, ev in self.edata.items()},
        }
 
    def from_numpy_dict(self, src_dict):
        self.PADSIZE = src_dict["PADSIZE"]
        self.n_nodes = src_dict["n_nodes"]
        self.neighbor_idx = torch.from_numpy(src_dict["neighbor_idx"])
        self.edge_mask = torch.from_numpy(src_dict["edge_mask"])
        self._dst_edim_locators = torch.from_numpy(src_dict["_dst_edim_locators"])
        self.batch_num_nodes = torch.from_numpy(src_dict["batch_num_nodes"])
        self.ndata = {
             nk: from_np(nv) for nk, nv in src_dict["ndata"].items()
        }
        self.edata = {
            ek: from_np(ev) for ek, ev in src_dict["edata"].items()
        }
        return self
 
    @staticmethod
    def batch(vls: Sequence["VerletList"]):
        """
        Batching a list of VerletLists.
        """
        batched_vl = VerletList() 
        batched_vl.PADSIZE = vls[0].PADSIZE 
        batched_vl.batch_num_nodes = torch.cat([vl.batch_num_nodes for vl in vls]) 
        batched_vl.n_nodes = torch.sum(batched_vl.batch_num_nodes) 
        bnn_offsets = torch.repeat_interleave(
            torch.cat([torch.LongTensor([0]), torch.cumsum(batched_vl.batch_num_nodes, dim=0)[:-1]]),
            batched_vl.batch_num_nodes
        )
        batched_vl.neighbor_idx = torch.cat(
            [vl.neighbor_idx for vl in vls], dim=0
        ) + bnn_offsets.unsqueeze(1) 
        batched_vl.edge_mask = torch.cat([vl.edge_mask for vl in vls], dim=0) 
        batched_vl.ndata = {}
        for nk, nfeat in vls[0].ndata.items():
            if isinstance(nfeat, torch.Tensor):
                batched_vl.ndata[nk] = torch.cat([vl.ndata[nk] for vl in vls], dim=0)
            elif isinstance(nfeat, SphericalTensor):
                batched_vl.ndata[nk] = nfeat.self_like(
                    torch.cat([vl.ndata[nk].ten for vl in vls], dim=0)
                )
            elif nfeat is None:
                batched_vl.ndata[nk] = None
            else:
                raise TypeError 
        batched_vl.edata = {} 
        for ek, efeat in vls[0].edata.items():
            if isinstance(efeat, torch.Tensor):
                batched_vl.edata[ek] = torch.cat([vl.edata[ek] for vl in vls], dim=0)
            elif isinstance(efeat, SphericalTensor):
                batched_vl.edata[ek] = efeat.self_like(
                    torch.cat([vl.edata[ek].ten for vl in vls], dim=0)
                )
            elif efeat is None:
                batched_vl.edata[ek] = None 
            else:
                raise TypeError 
        
        batched_vl._dst_edim_locators = torch.cat(
            [vl._dst_edim_locators for vl in vls], dim=0
        )
        return batched_vl


