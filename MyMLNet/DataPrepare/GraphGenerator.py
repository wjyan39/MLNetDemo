import torch 
import torch.nn as nn 

from BasicUtility.VerletList import VerletList 
from BasicUtility.O3.O3Tensor import O3Tensor 
from OrbNet2.MolFeature import MolFeature, Molecule
from OrbNet2.QCInterface import OneBodyToSpherical, TwoBodyToSpherical, OneBodyReduction
from Config import AOConfig2


EDGE_PADDING_SIZE = 32 

class VanilaLoader(nn.Module):
    """
    Fills in metadata only to the VerletList, no cutoffs in edges.
    """
    def __init__(self):
        super.__init__()
    
    def forward(self, mol_feature):
        two_body_data, one_body_data = {}, {} 
        num_nodes = len(mol_feature.molecule.atomic_numbers) 
        # Discard digonal blocks 
        verlet_mask = ~torch.eye(num_nodes, dtype=torch.bool) 
        mol_feature.lr_vl = VerletList().from_mask(verlet_mask, num_nodes, num_nodes, one_body_data, two_body_data)


class GraphLoader(nn.Module):
    """
    Equivariant and differentiable graph generation for AO features.
    """
    def __init__(self, config:AOConfig2, load_onebody=True):
        super.__init__()
        self.load_onebody = load_onebody 
        self.nfeats = config.one_body_features 
        self.efeats = config.two_body_features 
        self.shell_map = config.shells_map 
        # torch-gauge interface
        self.onebody2sp = OneBodyToSpherical(config.metadata_in_1body_so3, rep_dim=1)
        self.twobody2sp = TwoBodyToSpherical(config.metadata_2body_so3, rep_dims=(0, 1), basis_layout=self.shells_map)
        if not load_onebody:
            self.onebody_reduction = OneBodyReduction(config.metadata_2body_so3, basis_layout=self.shells_map)
        self._edge_dim = len(config.two_body_features)
        self.register_buffer("_ecut", torch.tensor([config.edge_cutoffs[feat] for feat in config.two_body_features]))
    
    def forward(self, mol_feature:MolFeature, padding_size=EDGE_PADDING_SIZE, low_memory=False, density_kernel=False):
        X = mol_feature.feat_2body.type(self._ecut.dtype) 

        two_body_data, one_body_data = {}, {}
        num_nodes = len(mol_feature.molecule.atomic_numbers)

        if density_kernel:
            occ_freq = torch.pow(4, torch.arange(1, 5))
            occs = torch.zeros_like(mol_feature.eorbs)
            occs[: len(mol_feature.occs)] = mol_feature.occs
            occ_kernel = torch.exp(
                (mol_feature.homo - mol_feature.eorbs).unsqueeze(1).mul(occ_freq.unsqueeze(0)).abs().neg()
            ).mul(occs.unsqueeze(1))
            vir_freq = torch.pow(4, torch.arange(1, 5))
            vir_kernel = torch.exp(
                (mol_feature.lumo - mol_feature.eorbs).unsqueeze(1).mul(vir_freq.unsqueeze(0)).abs().neg()
            ).mul((2 - occs).unsqueeze(1))
            mo_kernel = torch.cat([occ_kernel, vir_kernel], dim=1).t()
            kernel_densities = (
                mol_feature.orbitals.t()
                .unsqueeze(0)
                .mul(mo_kernel.unsqueeze(1))
                .matmul(mol_feature.orbitals)
                .permute(1, 2, 0)
                .type(self._ecut.dtype)
            )
            X = torch.cat([X, kernel_densities], dim=-1) 
        
        folded_X = self.twobody2sp(mol_feature.molecule.symbols, X) 
        if self.load_onebody:
            H_a = torch.stack(
                [mol_feature.feat_1body[feat] for feat in self.efeats],
                dim=-1,
            ).type(self._ecut.dtype) 
        else:
            H_a = self.onebody_reduction(
                mol_feature.molecule.atomic_numbers, folded_X.ten[torch.eye(num_nodes, dtype=torch.bool)]
            )
        one_body_data["atom_f"] = O3Tensor.from_so3(self.onebody2sp(H_a).unfold(update_self=True), parity=1)

        folded_X = O3Tensor.from_so3(folded_X, parity=1)
        ao_norm_trans = -torch.log(folded_X.ten.pow(2).sum((2,3)).sqrt() + 1e-6) 

        coord_xyz: torch.Tensor = mol_feature.geometry.type(self._ecut.dtype) 
        one_body_data["coordinates_xyz"] = coord_xyz 
        # pair distance and unit vec 
        xyzdiff = coord_xyz.unsqueeze(0) - coord_xyz.unsqueeze(1) 
        two_body_data["distances"] = torch.norm(xyzdiff, dim=2, keepdim=True) 
        # edge mask 
        feat_mask = ao_norm_trans < 12 
        verlet_mask = feat_mask.any(-1) 
        verlet_mask = torch.logical_and(~torch.eye(num_nodes, dtype=torch.bool), verlet_mask)
        if low_memory:
            ao_in_mask_ten = torch.logical_and(
                ~(folded_X.ten == 0).any(-1),
                verlet_mask.unsqueeze(2).unsqueeze(3),
            ).bool() 
            two_body_data["ao_in_mask"] = folded_X.self_like(ao_in_mask_ten)
            one_body_data["ao_in_content"] = folded_X.ten[ao_in_mask_ten, :]
            two_body_data["ao_in"] = None 
        else:
            two_body_data["ao_in"] = folded_X 
        two_body_data["unit_vec"] = torch.zeros_like(xyzdiff)
        two_body_data["unit_vec"][verlet_mask, :] = xyzdiff[verlet_mask, :] / two_body_data["distances"][verlet_mask, :]
        one_body_data["atomic_numbers"] = torch.LongTensor(mol_feature.molecule.atomic_numbers) 
        mol_feature.lr_vl = VerletList().from_mask(verlet_mask, padding_size, num_nodes, one_body_data, two_body_data)

        if low_memory:
            del mol_feature.feat_2body 
            mol_feature.feat_2body = None 
            del one_body_data, two_body_data, X, folded_X 
        
        return mol_feature


class BatchTransform(nn.Module):
    def __init__(self, config:AOConfig2):
        super.__init__() 
        self.nfeats = config.one_body_features 
        self.efeats = config.two_body_features 
        self.shells_map = config.shells_map 
        self.onebody2sp = OneBodyToSpherical(config.metadata_1body_o3, rep_dim=1) 
        self.twobody2sp = TwoBodyToSpherical(config.metadata_2body_so3, rep_dims=(1, 2), basis_layout=self.shells_map) 
        self.onebody_reduction = OneBodyReduction(config.metadata_2body_so3, basis_layout=self.shells_map) 
        self._edge_dim = len(config.two_body_features) 
        self.register_buffer("_ecut", torch.tensor([config.edge_cutoffs[feat] for feat in config.two_body_features])) 
    
    def forward(self, mol_feature:MolFeature):
        num_nodes = len(mol_feature.molecule[0].atomic_numbers) 
        X = mol_feature.feat_2body.type(self._ecut.dtype) 
        folded_X = self.twobody2sp(mol_feature.molecule[0].symbols, X) 
        H_a = self.onebody_reduction(
            mol_feature.atomic_numbers.flatten.long(),
            torch.diagnal(folded_X.ten, dim1=1, dim2=2).permute(0, 4, 1, 2, 3).flatten(0, 1) 
        )
        mol_feature.lr_vl.ndata["atom_f"] = O3Tensor.from_so3(self.onebody2sp(H_a).unfold(update_self=True), parity=1)

        folded_X = O3Tensor.from_so3(folded_X, parity=1)
        diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=X.device) 
        coord_xyz: torch.Tensor = mol_feature.geometry.type(self._ecut.dtype)
        mol_feature.lr_vl.ndata["coordinates_xyz"] = coord_xyz.flatten(0, 1)
        xyzdiff = coord_xyz.unsqueeze(1) - coord_xyz.unsqueeze(2)
        distances = torch.zeros(*xyzdiff.shape[:3], 1, dtype=xyzdiff.dtype, device=xyzdiff.device)
        distances[:, diag_mask] = torch.norm(xyzdiff[:, diag_mask], dim=2, keepdim=True)
        mol_feature.lr_vl.edata["distances"] = distances.flatten(0, 1) 
        # Discard diagonal blocks
        mol_feature.lr_vl.edata["ao_in"] = O3Tensor(
            folded_X.ten.flatten(0, 1),
            (2, 3),
            metadata=folded_X.metadata,
            num_channels=folded_X.num_channels, 
            rep_layout=folded_X.rep_layout
        )
        univecs = torch.zeros_like(xyzdiff) 
        univecs[:, diag_mask, :] = xyzdiff[:, diag_mask, :] / distances[:, diag_mask, :]
        mol_feature.lr_vl.edata["unit_vec"] = univecs.flatten(0, 1)
        mol_feature.lr_vl.ndata["atomic_numbers"] = mol_feature.atomic_numbers.flatten().long() 
        
        return mol_feature  

