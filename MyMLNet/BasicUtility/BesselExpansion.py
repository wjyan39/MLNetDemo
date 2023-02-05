from matplotlib.widgets import EllipseSelector
from pandas import cut
import torch 
import torch.nn 
# from UtilFunc import _cutoff_fn 
import numpy as np 
import sympy as sym 
from scipy import special as sp 
from scipy.optimize import brentq 

import os 
import math 

from BasicUtility.UtilFunc import _cutoff_fn, device 

# def _cutoff_fn(D:torch.tensor, cutoff):
#    x = D / cutoff 
#    x3 = x ** 3 
#    x4 = x3 * x 
#    x5 = x4 * x 
#    res = 1 - 6 * x5 + 15 * x4 - 10 * x3
#    return res 

def rbf_expansion(D, centers, widths, cutoff):
    """
    Input D: distance matrix beteen atoms
          centers, widths: params in tensor (both size of K channels) that determine the rbf  
    Output rbf: expanded rbfs, size in (D.shape, K) 
    """
    rbf = _cutoff_fn(D=D, cutoff=cutoff) * torch.exp(-widths * (torch.exp(-D) - centers) ** 2) 
    return rbf 


def Jn(r, n):
    """
    spherical bessel funtions of the first type of order n 
    """
    return np.sqrt(np.pi/(2*r)) * sp.jv(n+0.5, r) 


def Jn_zero(n, k):
    zerosj = np.zeros((n+1,k), dtype="float32") 
    zerosj[0] = np.arange(1, k+1) * np.pi 
    points = np.arange(1, k+n+1) * np.pi 
    racines = np.zeros(k+n, dtype="float32") 
    for i in range(1, n+1):
        for j in range(k+n-i):
            foo = brentq(Jn, points[j], points[j+1], (i,))
            racines[j] = foo 
        points = racines 
        zerosj[i][:k] = racines[:k] 
    return zerosj 


def spherical_bessel_formulas(n):

    x = sym.symbols('x') 
    f = [sym.sin(x)/x] 
    a = sym.sin(x)/x 
    for i in range(n):
        b = sym.diff(a, x) / x 
        f += [sym.simplify(b*(-x)**(i+1))] 
        a = sym.simplify(b) 
    return f 


def bessel_basis(n, k):
    zeros = Jn_zero(n, k) 
    norms = [] 
    for order in range(n+1):
        tmp_norm = [] 
        for i in range(k):
            tmp_norm += [0.5 * Jn(zeros[order, i], order+1)**2] 
        tmp_norm = 1 / np.array(tmp_norm)**0.5 
        norms += [tmp_norm] 
    
    f = spherical_bessel_formulas(n) 
    x = sym.symbols('x') 
    bess_basis = [] 
    for order in range(n+1):
        tmp_bess_basis = []
        for i in range(k):
            tmp_bess_basis += [sym.simplify(norms[order][i]*f[order].subs(x, zeros[order, i]))] 
        bess_basis += [tmp_bess_basis] 
    return bess_basis 


def sph_harm_prefactor(l, m):
    return ((2*l+1) * np.math.factorial(l-abs(m)) / (4 * np.pi * np.math.factorial(l+abs(m)))) ** 0.5 


def associated_legendre_polynomials(l, zero_m_only=True):

    z = sym.symbols('z') 
    P_l_m = [[0]*(j+1) for j in range(l+1)] 
    P_l_m[0][0] = 1 
    if l > 0:
        P_l_m[1][0] = z 
        for j in range(2, l+1):
            P_l_m[j][0] = sym.simplify( 
                ((2*j-1)*z*P_l_m[j-1][0] - (j-1)*P_l_m[j-2][0]) / j 
            )
        if not zero_m_only:
            for i in range(1, l):
                P_l_m[i][i] = sym.simplify((l-2*i)*P_l_m[i-1][i-1]) 
                P_l_m[i+1][i] = sym.simplify((2*i+1)*z*P_l_m[i][i]) 
                for j in range(i+2, l+1):
                    P_l_m[j][i] = sym.simplify(
                        ((2*j-1)*z*P_l_m[j-1][i] - (i+j-1)*P_l_m[j-2][i]) / (j-i)  
                    )
            
            P_l_m[l][l] = sym.simplify((1-2*l)*P_l_m[l-1][l-1]) 
    
    return P_l_m 


def real_sph_harm(l, zero_m_only=True, spherical_coordinates=True):
    if not zero_m_only:
        S_m = [0] 
        C_m = [1] 
        for i in range(1, l+1):
            x = sym.symbols('x') 
            y = sym.symbols('y') 
            S_m += [x*S_m[i-1] + y*C_m[i-1]] 
            C_m += [x*C_m[i-1] - y*S_m[i-1]] 
    P_l_m = associated_legendre_polynomials(l, zero_m_only) 
    if spherical_coordinates:
        theta = sym.symbols('theta') 
        z = sym.symbols('z') 
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta)) 
        if not zero_m_only:
            phi = sym.symbols('phi') 
            for i in range(len(S_m)):
                S_m[i] = S_m[i].subs(x, sym.sin(theta) * sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi)) 
            for i in range(len(C_m)):
                C_m[i] = C_m[i].subs(x, sym.sin(theta) * sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))
     
    Y_l_m = [['0']*(2*j+1) for j in range(l+1)] 
    for i in range(l+1):
        Y_l_m[i][0] = sym.simplify(sph_harm_prefactor(i,0)*P_l_m[i][0]) 
    if not zero_m_only:
        for i in range(1, l+1):
            for j in range(1, i+1):
                Y_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i,j) * C_m[j] * P_l_m[i][j]
                )
        for i in range(1, l+1):
            for j in range(1, i+1):
                Y_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i,-j) * S_m[j] * P_l_m[i][j]
                )
    
    return Y_l_m


def Y_l_zero(l):
    """
    compute s(l=0) type spherical harmonics only 
    """
    P_l_m = associated_legendre_polynomials(l, zero_m_only=True) 
    Y_l = [['0'] * (2*j+1) for j in range(l+1)] 
    for i in range(l+1):
        Y_l[i][0] = sym.simplify(
            sph_harm_prefactor(i,0) * P_l_m[i][0] 
        )
    return Y_l 


def bessel_expansion_raw(D:torch.Tensor, num:int, cutoff):
    n_rbf = torch.arange(1, num+1).view(1,-1).type(D.type()).to(device) 
    return math.sqrt(2.0/cutoff) * torch.sin(n_rbf*D*math.pi/cutoff) / D 


def _cutoff_fn_bessel(d_expanded, cutoff, p):
    p = torch.Tensor([p]).type(d_expanded.type()) 
    return 1 - (p+1)*d_expanded.pow(p) + p*d_expanded.pow(p+1.0) 


def bessel_expansion_continuous(D:torch.Tensor, num:int, cutoff, p=6):
    cutoff = torch.Tensor([cutoff]).type(D.type()) 
    continuous_cutoff = _cutoff_fn_bessel(D/cutoff, cutoff, p) 
    return bessel_expansion_raw(D, num, cutoff) * continuous_cutoff


class BesselCalculator:

    def __init__(self, n_srbf, n_shbf, envelop_p, cos_theta=True):
        
        self.envelop_p = envelop_p
        self.n_srbf = n_srbf
        self.n_shbf = n_shbf
        self.dim_sbf = n_srbf * (n_shbf+1)
        
        self.z_ln = torch.as_tensor(Jn_zero(n_shbf, n_srbf)).type(torch.float32) 
        self.normalizer = self.get_norm() 
        

        x = sym.symbols('x') 
        j_l = spherical_bessel_formulas(n_shbf)  
        self.j_l = [sym.lambdify([x], f, modules=torch) for f in j_l] 
        if cos_theta:
            Y_l = Y_l_zero(l=n_shbf)
            angle_input = sym.symbols('theta') 
        else:
            Y_l = real_sph_harm(l=n_shbf) 
            angle_input = sym.symbols('theta') 
        self.Y_l = [sym.lambdify([angle_input], f[0], modules=torch) for f in Y_l] 
        self.Y_l[0] = lambda _theta: torch.zeros_like(_theta).fill_(float(Y_l[0][0])) 
        self.to(device)

    def get_norm(self):
        norms = torch.zeros_like(self.z_ln)
        for l in range(self.n_shbf + 1):
            for n in range(self.n_srbf):
                norms[l][n] = torch.sqrt(2/(Jn(self.z_ln[l][n], l+1))**2) 
        return norms 

    def to(self, _device):
        self.z_ln = self.z_ln.to(_device) 
        self.normalizer = self.normalizer.to(_device) 
    
    def cal_sbf(self, D, angle, feature_dist):
        scaled_D = (D / feature_dist).view(-1, 1, 1) 
        expanded_D = scaled_D * self.z_ln 
        radius_part = torch.cat([f(expanded_D[:,[l],:]) for l, f in enumerate(self.j_l)], dim=1) 
        angle_part = torch.cat([f(angle).view(-1,1) for f in self.Y_l], dim=-1) 
        res = self.normalizer.unsqueeze(0) * radius_part * angle_part.unsqueeze(-1) 
        return res.view(-1, self.dim_sbf) 
    
    def cal_rbf(self, D, feature_dist, n_rbf):
        if self.envelop_p > 0:
            return bessel_expansion_continuous(D, num=n_rbf, cutoff=feature_dist, p=self.envelop_p) 
        else:
            return bessel_expansion_raw(D, num=n_rbf, cutoff=feature_dist) 
    


