import torch 
import numpy as np 
import h5py
import os 
from torch.utils.data import Dataset 
from tqdm import tqdm
from typing import List

from BasicUtility.VerletList import VerletList 
from BasicUtility.O3.O3Tensor import O3Tensor 
from OrbNet2.MolFeature import MolFeature, Molecule
from OrbNet2.QCInterface import OneBodyToSpherical, TwoBodyToSpherical, OneBodyReduction
from Config import AOConfig2

EDGE_PADDING_SIZE = 32

class MolDataset(Dataset):
    """
    AO Features from QC Calculations, load from HDF5 and process to needed form.
    """
    def __init__(self, datafiles:List[str], label_name:str, config:AOConfig2, mode:str="valid", metadata_only=False, maxmol=160000):
        self.config = config 
        self.label_trans = self.config.label_transform 
        self.mode = mode 
        self.metadata_only = metadata_only 
        self.label = label_name 
        self.MAXMOL:int = maxmol 
        self.base_label = config.base_label_name 
        if self.base_label:
            self.base_label_trans = self.config.base_label_transform 
        
        self.data_h5:List[h5py.filters] = []
        for datafile in datafiles:
            self.data_h5.append(h5py.File(datafile, "r")) 
        self.data_names:List[str] = [os.path.basename(r).replace(".hdf5", "") for r in datafiles] 
        dataset_names = [name.split("_")[0] for name in self.data_names] 
        self.name = "_".join(dataset_names) if len(dataset_names) > 1 else self.data_names[-1] 
        self._process()
        for f in self.data_h5:
            f.close()
    
    def _process(self):
        graphs, labels, IDs = [], [], []
        self.features:List[MolFeature] = []
        for i, data in enumerate(self.data_h5):
            if self.mode not in data.keys():
                continue 
            samples = list(data[self.mode].keys()) 
            # np.random.RandomState(seed=42).shuffle(samples) 
            # read in molecule data 
            ct = 0 
            for mol in tqdm(samples):
                for geo_id in data[self.mode][mol].keys():
                    if ct >= self.MAXMOL:
                        print("Reach the Max Capacity of model read-in.")
                        break 
                    try:
                        feature_group = data[self.mode][mol][geo_id] 
                        identifier = {"MolID":mol, "GeoID":geo_id}
                        IDs.append(identifier) 
                        mol_feature = MolFeature(
                            identifier, 
                            torch.stack(
                                [
                                    torch.from_numpy(
                                        feature_group[f"2body/{feat}"][()].copy()
                                    ) for feat in self.config.two_body_features
                                ]
                            ),
                            None,
                            Molecule(
                                atomic_numbers = feature_group["atomic_numbers"][()],
                                geometry = feature_group["geometry_bohr"][()]
                            ),
                            label_force = torch.from_numpy(
                                self.config.force_transform(feature_group[self.config.force_label][()])
                            ) if self.config.forces else None, 
                            base_grads = torch.from_numpy(
                                self.config.base_force_transform(feature_group[self.config.base_force_label][()])
                            ) if self.config.forces else None, 
                            feat_grads = torch.stack(
                                [
                                    torch.from_numpy(
                                        feature_group[f"2body_grad/{feat}"][()]
                                    ) for feat in self.config.two_body_features
                                ],
                                dim=-1
                            ) if self.config.forces else None, 
                        )
                        l_raw = feature_group[self.label][()].copy()
                        if isinstance(l_raw, np.float64):
                            l_raw = np.asarray([[l_raw]]) 
                        elif l_raw.ndim == 1:
                            l_raw = np.asarray([l_raw]) 
                        l = torch.from_numpy(self.label_trans(l_raw)) 
                        mol_feature.label = l 
                        if self.base_label:
                            lb_raw = feature_group[self.base_label][()].copy()
                            if isinstance(lb_raw, np.float64):
                                lb_raw = np.asarray([[lb_raw]])
                            elif lb_raw.ndim == 1:
                                lb_raw = np.asarray([lb_raw])
                            mol_feature.base_label = torch.from_numpy(self.base_label_trans(lb_raw))
                    except KeyError:
                        print("Bad geometry: ", mol, geo_id) 
                        raise 
                        continue 
                    labels.append(l) 
                    self.features.append(mol_feature)
                    ct += 1
            data.close()
            return labels, IDs 

    def __len__(self):
        return len(self.features) 
        
    def __getitem__(self, idx):
        f = self.features[idx] 
        return f 
        
    def batcher(self, batch):
        features = batch 
        batch_feature_sets = MolFeature.batch(features)
        return batch_feature_sets


class VanilaLoader(torch.nn.Module):
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


class GraphLoader(torch.nn.Module):
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

def pre_transform_resolver(config:AOConfig2):
    if config.forces:
        return VanilaLoader()
    else:
        return GraphLoader(config, load_onebody=False) 
