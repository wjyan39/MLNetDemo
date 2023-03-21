import torch 
import numpy as np 
import h5py
import os 
from torch.utils.data import Dataset 
from tqdm import tqdm
from typing import List

from OrbNet2.MolFeature import MolFeature, Molecule
from DataPrepare.GraphGenerator import VanilaLoader, GraphLoader 
from Config import AOConfig2


def pre_transform_resolver(config:AOConfig2):
    if config.forces:
        return VanilaLoader()
    else:
        return GraphLoader(config, load_onebody=False) 



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
        self.transform = pre_transform_resolver(config)
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
        feat = self.transform(f)
        return feat 
        
    def batcher(self, batch):
        features = batch 
        batch_feature_sets = MolFeature.batch(features)
        return batch_feature_sets


