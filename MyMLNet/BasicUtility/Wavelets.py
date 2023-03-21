import torch
import torch.nn as nn 
import math 

class WaveLet(nn.Module):
    def __init__(self, in_features, out_features, eps=0.4):
        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features 
        frequencies = torch.rand(in_features, out_features) * 8 * math.pi 
        phases = torch.rand(in_features, out_features) * 2 * math.pi 
        scales = torch.rand(in_features, out_features) * 2 
        self.frequencies = nn.Parameter(frequencies.unsqueeze(0) / 4, requires_grad=True) 
        self.phases = nn.Parameter(phases.unsqueeze(0) / 4, requires_grad=True)
        self.scales = nn.Parameter(scales.unsqueeze(0), requires_grad=True) 
        self.eps = eps  

    def basis_func(self, x , d):
        """
        Args:
            x: modulated features 
            d: scaled features 
        """
        phi = torch.sin(4 * (x * self.frequencies + self.phases)) * torch.exp(-(((torch.abs(self.scales) + self.eps) * x) ** 2))
        return phi 
    
    def forward(self, raw_inp):
        size = (*raw_inp.shape[:-1], self.in_features, self.out_features) 
        x = raw_inp.unsqueeze(-1).expand(size) 
        rbfs = self.basis_func(torch.exp(-x), x) 
        return rbfs.view(*size[:-2], size[-1] * size[-2]) 
    

class GTO(nn.Module):
    """
    Gaussian-type wavelet basis functions.
    """
    def __init__(self, in_features, out_features, eps=0.08, alpha=1.2):
        super().__init__()
        self._eps = eps 
        self.in_features = in_features
        self.out_features = out_features 
        freqs = self._eps * torch.pow(alpha, torch.arange(self.out_features)) 
        self.freqs = nn.Parameter(freqs.unsqueeze(0).expand(in_features, -1).clone(), requires_grad=True) 

    def basis_func(self, f):
        """
        Args:
            f: feats in log scale 
        """    
        freqs = self.freqs.abs() 
        phi = torch.exp(-f.pow(2) * freqs) * torch.cos(f * freqs * math.pi)
        return phi 
    
    def forward(self, feat):
        size = (*feat.shape[:-1], self.in_features, self.out_features)
        f = feat.unsqueeze(-1).expand(size)
        rbfs = self.basis_func(f) 
        return rbfs.view(*size) 
    

class Chebyshev(nn.Module):
    """
    Chebyshev polynomials up to stated order.
    """
    def __init__(self, order:int):
        self.order = order 
        orders = torch.arange(order+1, dtype=torch.long) 
        self.register_buffer("orders", orders) 
        self._init_coefficients() 
    
    def _init_coefficients(self):
        """
        Coeffients via recursion.
        """
        coeff = torch.zeros(self.order + 1, self.order + 1)
        coeff[0, 0] = 1 
        coeff[1, 1] = 1 
        # expansion coeffs in columns
        for idx in range(2, self.order + 1):
            coeff[1:, idx] = 2 * coeff[:-1, idx-1] 
            coeff[:, idx] -= coeff[:, idx - 2]
        self.register_buffer("cheby_coeff", coeff[:, 1:]) 
    
    def forward(self, feat):
        size = (*feat.shape, self.order+1) 
        feat = feat.unsqueeze(-1).expand(size).pow(self.orders) 
        out = torch.matmul(feat, self.cheb_coeff)
        return out 





