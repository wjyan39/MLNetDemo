"""
Model related set-up.
Copied from OrbNet2.
"""
from typing import Callable, Dict, List 
from pydantic import BaseModel, Extra 

import torch 
import enum 

class ModelVersion(str, enum.Enum):
    OrbNet2 = "orbnet2" 

class NormType(str, enum.Enum):
    batch = "batch"
    layer = "layer"
    node = "node" 

class AOConfig(BaseModel):
    # Dataloader
    feature_grad: bool = False  # Require feature gradients when using the differentiable graph generator
    refit_shifts: bool = True   # Does not support targets other than global scalars, currently.

    # Feature pipeline 
    one_body_features: List[str] = ["F", "P", "S", "H"]
    two_body_features: List[str] = ["F", "P", "S", "H"] 
    n_marker: int = 118 
    # two body distances are computed on the fly 

    # Default model hyper-parameters 
    dim: int = 256       # Width of hidden message passing layers
    output_dim: int = 1  # Dimension of targets
    ndim: int = len(one_body_features)  # Input channels for nodes
    edim: int = len(two_body_features)  # Input channels for edges
    erbf_dim: int = 16  # Number of basis functions for the edge layer 
    n_conv: int = 4     # Number of message passing layers 
    n_decoding: int = 4 # Number of point-wise interaction layers
    n_aux_decoding: int = 3
    enc_norm: str = NormType.batch
    graph_norm: str = NormType.node
    dec_norm: str = NormType.batch
    attn_heads: int = 8  # Number of attention heads in multi-head attention layers
    activation: str = "swish"

    # Training set-up
    conformer_loss_weight: float = 8
    pair_training: bool = False
    refit_d3: bool = False
    distributed: bool = False

    # Data processing 
    MAXMOL: int = 160000
    ## Log-scale Cutoffs for pair in matrices 
    edge_cutoffs: Dict[str, float] = {
       "F": 12.0,
       "D": 12.0,
       "P": 12.0,
       "S": 12.0,
       "H": 12.0,
    } 

    # Training schedule 
    batch_size: int = 64
    warmup_epoch: int = 100
    max_epoch: int = 300
    max_lr: float = 5e-4
    optimizer: str = "torch_adam"
    lr_scheduler: str = "cosine_annealing"
    
    possible_atoms = list(range(1, 119)) 
    possible_shells = [
    "1s",
    "2p",
    "2s",
    "3d",
    "3p",
    "3s",
    "4d",
    "4p",
    "4s",
    "5d",
    "5p",
    "5s",
    "6d",
    "6p",
    "6s",
    "7d",
    ] 

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow 

def nlogabs(x):
    return -torch.log(torch.abs(x) + 1e-4)

def identity(x):
    return x 

class AOConfig2(AOConfig):
    model_version: str = ModelVersion.OrbNet2
    run_name: str = "inference"
    # spdfghi..., ranging from 0 to max_l
    # The number of channels per each angular momenta for one-body features
    metadata_in_1body_so3: List[int] = [16, 8, 4, 0, 0]
    # The number of channels per each angular momenta/parity for hidden one-body attributes
    metadata_1body_o3: List[int] = [128, 24, 48, 8, 24, 4, 12, 2, 6, 0]
    # The channel padding size per each angular momenta for two-body features
    metadata_2body_so3: List[int] = [2, 1, 0, 0, 0]

    # The atomic representation metadata of the target property
    metadata_out_1body_o3: List[int] = [13, 0, 11, 0, 9, 0, 4, 0, 1, 0]

    shells_map: Dict[str, List[str]] = {
        "H": ["1s", "2s"],
        "He": ["1s"],
        "Li": ["2s", "2p", "2p", "2p"],
        "Be": ["2s", "2p", "2p", "2p"],
        "B": ["2s", "2p", "2p", "2p"],
        "C": ["2s", "2p", "2p", "2p"],
        "N": ["2s", "2p", "2p", "2p"],
        "O": ["2s", "2p", "2p", "2p"],
        "F": ["2s", "2p", "2p", "2p"],
    }

    cg_precision: torch.dtype = torch.float
    attn_heads: int = 8
    block_dim: int = 8
    orbital_features: bool = False

    decoding_mode: str = "scalar"
    reduce_op: str = "sum"
    output_unit = "unitless"
    label_transform: Callable[[torch.Tensor], torch.Tensor] = staticmethod(identity)
    base_label_transform: Callable[[torch.Tensor], torch.Tensor] = staticmethod(identity)

    possible_elements: List[int] = [1, 6, 7, 8, 9]
    to_interleaved: bool = True
