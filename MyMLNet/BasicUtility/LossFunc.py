import torch 
from copy import deepcopy 
import torch_geometric 

from BasicUtility.UtilFunc import device, mae_fn, mse_fn, kcal2ev 

class LossFn:
    def __init__(self, w_e, w_f, w_q, w_p, action="E", target_names=None):
        self.target_names = target_names 
        self.action = deepcopy(action) 
        self.w_e = w_e # weight of Energy pred.
        self.w_f = w_f # weight of Force pred. 
        self.w_q = w_q # weight of Charge pred.  
        self.w_p = w_p # weight of Dipole pred. 
    
    def __call__(self, E_pred, F_pred, Q_pred, D_pred, data, require_detail=False):
        if self.action in ["E", "QD"]:
            # energy prediction
            mae_loss = torch.mean(torch.abs(E_pred - data.E), dim=0, keepdim=True) 
            rmse_loss = torch.sqrt(torch.mean((E_pred - data.E)**2), dim=0, keepdim=True) 
            total_loss = mae_loss.sum() 
            if require_detail:
                detail = {"MAE_{}".format(name):mae_loss[:,i].item() for i, name in enumerate(self.target_names)} 
                for i, name in enumerate(self.target_names):
                    detail["RMSE_{}".format(name)] = rmse_loss[:, i].item() 
            else:
                detail = None 

            if self.action == "QD":
                q_mae = torch.mean(torch.abs(Q_pred - data.Q))
                d_mae = torch.mean(torch.abs(D_pred - data.D)) 
                total_loss = total_loss + self.w_q * q_mae + self.w_p * d_mae 
                if require_detail:
                    detail["MAE_Q"] = q_mae.item()
                    detail["MAE_D"] = d_mae.item()  

            return total_loss, detail 
        elif self.action == "all":
            # default physnet 
            E_loss, F_loss, Q_loss, D_loss = 0, 0, 0, 0
            E_loss = self.w_e * torch.mean(torch.abs(E_pred - data.E)) 
             # if 'F' in data.keys():
            #     F_loss_batch = torch_geometric.utils.scatter_('mean', torch.abs(F_pred - data['F'].to(device)),
            #                                                   data['atom_to_mol_batch'].to(device))
            #     F_loss = self.w_f * torch.sum(F_loss_batch) / 3

            Q_loss = self.w_q * torch.mean(torch.abs(Q_pred - data.Q))

            D_loss = self.w_d * torch.mean(torch.abs(D_pred - data.D))

            if require_detail:
                return E_loss + F_loss + Q_loss + D_loss, {"MAE_E": E_loss.item(), "MAE_F": F_loss,
                                                           "MAE_Q": Q_loss.item(), "MAE_D": D_loss.item()}
            else:
                return E_loss + F_loss + Q_loss + D_loss, None 
        else:
            raise ValueError("Invalid action: {}".format(self.action))          

