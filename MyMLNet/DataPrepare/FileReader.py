import numpy as np 
import scipy
from rdkit import Chem 
from rdkit.Chem import rdmolfiles 
from pyscf import gto, lib

from DataPrepare.DataUtility import sdf_mol_to_edge_index

def sdf_to_dict(sdf_path:str, mode="atom"):
    with open(sdf_path, 'r') as f:
        sdf_string = f.read() 
    mol_sdf = rdmolfiles.MolFromMolBlock(sdf_string, removeHs=False)

    # --Atomistic Feature-- 
    #  read from a single sdf file via rdkit  
    mol_dict = {}
    atm_coords = mol_sdf.GetConformers()[0].GetPositions() / lib.param.BOHR 
    atm_charges = np.array([atm.GetAtomicNum() for atm in mol_sdf.GetAtoms()]) 
    atm_symbols = np.array([atm.GetSymBol() for atm in mol_sdf.GetAtoms()]) 
    bonding_edges = sdf_mol_to_edge_index(mol_sdf) 

    natm = mol_sdf.GetNumAtoms()
    natm_heavy = natm - (atm_charges == 1).sum() 

    mol_dict["natm"] = natm                     # Number of atoms in the mole 
    mol_dict["atm_coords"] = atm_coords         # Atomic positions, unit Bohr, (natm, 3)
    mol_dict["atm_charges"] = atm_charges       # Atomic charges, unit a.u., (natm,) 
    mol_dict["bonding_edges"] = bonding_edges   # Molecule bonding adajacency, forms compact to torch_geometric edge_index (2, num_edge)

    if mode == "atom":
        return mol_dict, None 

    # --QM based Electronic Info-- 
    ao_dict = {} 
    ## running pyscf 
    mol_build = []
    for atm, coord in zip(atm_symbols, atm_coords):
        row = [atm] + [str(f) for f in coord.tolist()] 
        mol_build.append(" ".join(row)) 
    mole = gto.Mole() 
    mole.atom = "\n".join(mol_build) 
    mole.basis = "STO-3G" 
    mole.build() 
    assert (mole.nelec[0] == mole.nelec[1]) 
    nocc = mole.nelec[0]  
    ## Integrals 
    S = mole.intor("int1e_ovlp") 
    H = mole.intor("int1e_kin") + mole.intor("int1e_nuc") 
    H = (H + H.T) / 2  # symmetrize core Hamiltonian 
    
    ao_dict["ovlp"] = S        # ao basis Overlap S_uv
    ao_dict["coreH"] = H       # ao basis core Hamiltonian 

    return mol_dict, ao_dict 

if __name__ == '__main__':
    """
    Quantum chemistry process when reading original database
    Usage:
        sdf_reader.py [--task=<str>]
    Options:
        --task=<str>  Task (dev, valid_00, train, ...) [default: all]
    """
    import docopt
    from pathlib import Path 
    import pickle

    args = docopt(__doc__) 
    print(args) 
    TASK = args["--task"] 
    tasks = TASK 
    if TASK == "all":
        valids = args["--fold"]  # number of k-fold cross validation 
        assert (valids > 0 and valids <= 10)
        tasks = ["dev", "test"] + ["valid_0{}".format(i) for i in range(valids)]  

    reader = sdf_to_dict

    sdf, root = Path("raw-sdf"), Path("raw") 
    for t in tasks:
        print("currently processing {} files".format(t)) 
        d = {} 
        for f in (sdf / t).glob("**/*.sdf"):
            r = reader(f) 
            d[int(f.stem)] = r
        # dumping file to .dat
        with open(root / (t + ".dat"), "wb") as f_dat:
            pickle.dump(d, f_dat, pickle.HIGHEST_PROTOCOL) 
    





    

