import numpy as np 
import torch 

from scipy.special import binom, factorial 
from joblib import Memory 
import os 

memory = Memory(os.path.join(".", ".basic_cache"), verbose=0) 

def vm(m):
    return (1/2) * (m < 0).long() 

@memory.cache 
def get_c_lmtuv(l, m, t, u, v):
    c = (
        (-1) ** (t + v - vm(m)) 
        * (1/4) ** t 
        * binom(l, t) 
        * binom(l-t, torch.abs(m)+t) 
        * binom(t, u) 
        * binom(torch.abs(m), 2*v)
    )
    assert (c != 0).any()
    return c 

@memory.cache 
def get_ns_lm(l, m):
    return (1 / (2**torch.abs(m) * factorial(l))) * torch.sqrt(
        2 * factorial(l + torch.abs(m)) * factorial(l - torch.abs(m)) 
        / (2 ** (m ==0).long())
    ) 

def get_xyzcoeff_lm(l, m):
    ts, us, vs = [], [], [] 
    for t in torch.arange((l - torch.abs(m)) // 2 + 1):
        for u in torch.arange(t+1):
            for v in torch.arange(vm(m),torch.floor(torch.abs(m)/2 - vm(m)).long() + vm(m) + 1): 
                ts.append(t)
                us.append(u)
                vs.append(v) 
    ts, us, vs = torch.stack(ts), torch.stack(us), torch.stack(vs) 
    xpows_lm = 2*ts + torch.abs(m) - 2*(us + ts) 
    ypows_lm = 2 * (us + vs) 
    zpows_lm = l - 2*ts - torch.abs(m) 
    xyzpows_lm = torch.stack([xpows_lm, ypows_lm, zpows_lm], dim=0) 
    clm_tuv = get_c_lmtuv(l, m, ts, us, vs) 
    return clm_tuv, xyzpows_lm 

class RSHxyz(torch.nn.Module):
    """
    Generating Real Spherical Harmonics up to given order from xyz coordinates. 
    For ith (x,y,z) point in the batch, pre-generated coefficients are stored in the order:
            a_{i, tuv}^{lm}: l --> m --> tuv
    
    Args:
        max_l: int: The maximum order l of the desired spherical harmonics. 
    """

    def __init__(self, max_l:int):
        super().__init__() 
        self.max_l = max_l 
        self._init_coeffs() 
    
    def _init_coeffs(self):
        dst_pointers, xyzpows, ns_lms, clmtuvs = [], [], [], [] 
        for l in torch.arange(self.max_l + 1, dtype=torch.long):
            for m in torch.arange(-l, l+1, dtype=torch.long):
                ns_lm = get_ns_lm(l, m) 
                clm_tuv, xyzpow_lm = get_xyzcoeff_lm(l, m)  
                dst_pointer = torch.ones_like(clm_tuv) * (l*(l+1) + m) 
                dst_pointers.append(dst_pointer) 
                xyzpows.append(xyzpow_lm) 
                ns_lms.append(ns_lm) 
                clmtuvs.append(clm_tuv) 
        self.register_buffer("dst_pointers", torch.cat(dst_pointers).long()) 
        self.register_buffer("clm_tuvs", torch.cat(clmtuvs, dim=0)) 
        self.register_buffer("xyzpows", torch.cat(xyzpows, dim=1).long()) 
        self.register_buffer("ns_lms", torch.stack(ns_lms, dim=0)) 
        self.out_metadata = torch.ones((1,self.max_l+1), dtype=torch.long) 
        


