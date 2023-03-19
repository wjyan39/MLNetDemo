from dataclasses import dataclass 
from typing import List

import torch 
import numpy as np 

from BasicUtility.VerletList import VerletList

atomic_number_to_symbol_dict = {
    1: "H",     2: "He",    3: "Li",    4: "Be",
    5: "B",     6: "C",     7: "N",     8: "O",
    9: "F",     10: "Ne",   11: "Na",   12: "Mg",
    17: "Cl", 
}

@dataclass 
class Molecule:
    def __init__(self, atomic_numbers, geometry, charge=0):
        self.atomic_numbers = atomic_numbers 
        self.geometry = geometry 
        self.charge = charge 
        self.symbols = [atomic_number_to_symbol_dict[atm] for atm in self.atomic_numbers] 


class MolFeature:
    """
    Bagged matrices (T[\Psi]) and atomic metadata.
    """
    def __init__(
        self, 
        identifier, 
        feat_2body=None, 
        feat_1body=None,
        molecule=None,
        lr_vl=None,
        sr_vl=None,
        label=None,
        base_label=None,
        label_force=None,
        base_grads=None,
        feat_grads=None,
    ):
        self.id = identifier
        self.feat_2body = feat_2body
        self.feat_1body = feat_1body 
        self.lr_vl = lr_vl
        self.sr_vl = sr_vl 
        self.label = label 
        self.base_label = base_label
        self.molecule: Molecule = molecule
        self.geometry = None
        self.label_force = label_force
        self.base_grads = base_grads
        self.feat_grad = feat_grads 
        if self.molecule is not None:
            self.geometry = torch.from_numpy(self.molecule.geometry) 
            self.charge = torch.DoubleTensor([[self.molecule.charge]])
            self.atomic_numbers = torch.from_numpy(self.molecule.atomic_numbers)
    
    def to(self, device):
        dev_f = MolFeature(
            self.id, 
            lr_vl = self.lr_vl.to(device) if self.lr_vl else None,
            sr_vl = self.sr_vl.to(device) if self.sr_vl else None,
            label = self.label.to(device), 
            base_label=self.base_label.to(device) if self.base_label is not None else None,
            label_force=self.label_force.to(device) if self.label_force is not None else None,
            base_grads=self.base_grads.to(device) if self.base_grads is not None else None,
            feat_grads=self.feat_grads.to(device, non_blocking=True) if self.feat_grads is not None else None,
        )
        dev_f.molecule = self.molecule 
        if self.feat_2body is not None:
            dev_f.feat_2body = self.feat_2body.to(device)
        if self.geometry is not None:
            dev_f.geometry = self.geometry.to(device)
            dev_f.atomic_numbers = self.atomic_numbers.to(device) 
            dev_f.charge = self.charge.to(device)
        return dev_f

    @staticmethod
    def batch(features: List["MolFeature"]):
        batchedfeat = MolFeature(
            [f.id for f in features],
            lr_vl = VerletList.batch([f.lr_vl for f in features]),
            label=torch.cat([f.label for f in features], dim=0)
        )
        if features[0].base_label is not None:
            batchedfeat.base_label = torch.cat([f.base_label for f in features], dim=0) 
        if features[0].label_force is not None:
            batchedfeat.molecule = [f.molecule for f in features]
            batchedfeat.geometry = torch.stack([f.geometry for f in features], dim=0) 
            batchedfeat.atomic_numbers = torch.stack([f.atomic_numbers for f in features], dim=0) 
            batchedfeat.label_force = torch.stack([f.label_force for f in features], dim=0) 
            batchedfeat.base_grads = torch.stack([f.base_grads for f in features], dim=0) 
            batchedfeat.feat_grads = torch.stack([f.feat_grads for f in features], dim=0) 
            batchedfeat.feat_2body = torch.stack([f.feat_2body for f in features], dim=0)
        if features[0].charge is not None:
            batchedfeat.charge = torch.cat([f.charge for f in features], dim=0)
        batchedfeat.batch = True 
        return batchedfeat
