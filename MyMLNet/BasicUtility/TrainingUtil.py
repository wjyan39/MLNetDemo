from re import I
from unicodedata import name
from xml.etree.ElementInclude import default_loader
from copy import copy 
import torch 
import argparse 

from DataPrepare.DataUtility import DummyIMDataset
from DataPrepare.AlchemyDataset import AlchemyData 

def add_parser_args(parser:argparse.ArgumentParser):
    # model architecture 
    parser.add_argument('--debug_mode', type=str) 
    parser.add_argument('--num_modules', type=int, help="number of Module block applied in the model") 
    parser.add_argument('--bonding_type', type=str, help="eg. BN B B N, B for bonding-edge, N for non-bonding edge, and BN for both") 
    parser.add_argument('--cutoff', type=float, help="cutoff radius, in anstrom") 
    parser.add_argument('--num_embedding', type=int, default=95)
    parser.add_argument('--num_feature', type=int) 
    parser.add_argument('--num_lin_out', type=int, help="number of linear layers employed in Output block") 
    parser.add_argument('--num_lin_res_atom', type=int)
    parser.add_argument('--num_lin_res_interact', type=int)
    parser.add_argument('--num_lin_res_output', type=int)
    parser.add_argument("--num_output", type=int, default=1, help="number of outputs, defaults to 1 for energy and 2 to include charge.")
    # training strategy 
    parser.add_argument('--num_ephoc', type=int) 
    parser.add_argument('--learning_rate', type=float) 
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--valid_num', type=int, default=5, help="require k-fold validation") 
    parser.add_argument('--valid_batch_size', type=int) 
    parser.add_argument('--max_norm', type=float, help="used in parameter climb")
    parser.add_argument('--early_stop', type=int, default=-1, help="early stopping, set to -1 to disable")
    parser.add_argument('--optimizer', type=str, default='emaAms_0.999', help="Adam_${ema} | SGD")
    parser.add_argument('--decay_steps', type=int)
    parser.add_argument('--use_trained_model', type=str, default="False")  
    # --loss function set up--
    parser.add_argument('--l2lambda', type=float)
    parser.add_argument('--nh_lambda', type=float)
    parser.add_argument('--decay_rate', type=float)
    parser.add_argument('--force_weight', type=float)
    parser.add_argument('--charge_weight', type=float)
    parser.add_argument('--dipole_weight', type=float)
    parser.add_argument('--action', type=str, default="E")
    parser.add_argument('--target_name', type=str, default="E") 
    # files
    parser.add_argument('--log_file_name', type=str)
    parser.add_argument('--folder_prefix', type=str)
    parser.add_argument('--config_name', type=str, default='config.txt')
    parser.add_argument('--data_provider', type=str, help="Data provider arguments: qm9 | alchemy") 

    return parser 


class MyRandomSampler(torch.utils.data.RamdomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None):
        self.random_seed = None 
        super(MyRandomSampler, self).__init__(data_source, replacement, num_samples) 
    
    def __iter__(self):
        torch.manual_seed(self.random_seed)
        return super(MyRandomSampler, self).__iter__() 


def info_resolver(s:str):
    """
    resolve the way of rbf expansion
    eg. gaussian_64_10.0 means gaussian expansion, num_rbf=64 and cutoff_radius=10.0 
    """
    info = s.split('_') 
    res = {'name': info[0], 'dist':float(info[-1])} 
    if info[0] == 'bessel' or info[0] == 'gaussian':
        res['num'] = int(info[1])  
    elif info[0] == 'Dime': # the expansion in DimeNet 
        res['num'] = int(info[1]) 
        res['n_srbf'] = int(info[2]) 
        res['n_shbf'] = int(info[3]) 
    elif info[0] == 'none':
        pass 
    return res 


def option_solver(option_txt) -> dict():
    if len(option_txt.split('[')) == 1:
        return {}
    else:
        # option txt should be like '[n_read_out=2,other_opt=value]'
        option_pair = option_txt.split('[')[1]
        option_pair = option_pair[:-1] 
        return { argument.split('=')[0] : argument.split('=')[1] for argument in option_pair.split(',') }
    

def _add_arg_from_config(_kwargs, config_args):
    for args_name in ['cutoff']:
        _kwargs[args_name] = getattr(config_args, args_name)
    return _kwargs 


def print_val_results(dataset_name, loss, emae, ermse, qmae=None, qrmse=None, pmae=None, prmse=None):
    log_info = 'Validating {}: '.format(dataset_name)
    log_info += (' loss: {:.6f} '.format(loss))
    log_info += ('emae: {:.6f} '.format(emae))
    log_info += ('ermse: {:.6f} '.format(ermse)) 
    if qmae and qrmse:
        log_info += ('qmae: {:.6f} '.format(qmae))
        log_info += ('qrmse: {:.6f} '.format(qrmse))
    if pmae and prmse:
        log_info += ('pmae: {:.6f} '.format(pmae))
        log_info += ('prmse: {:.6f} '.format(prmse))
    return log_info


def train_step(model, _optimizer, data_batch, loss_fn, max_norm):
    # t0 = time.time()
    with torch.autograd.set_detect_anomaly(True):
        model.train()
        _optimizer.zero_grad()

        E_pred, F_pred, Q_pred, p_pred, loss_nh = model(data_batch)

        # t0 = record_data('forward', t0, True)

        loss = loss_fn(E_pred, F_pred, Q_pred, p_pred, data_batch) + loss_nh

        # print("Training b4 backward: {:.2f} MB".format(torch.cuda.memory_allocated(device) * 1e-6))

        loss.backward()

    # print("Training after backward: {:.2f} MB".format(torch.cuda.memory_allocated(device) * 1e-6))

    # t0 = record_data('backward', t0, True)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    _optimizer.step()

    torch.cuda.empty_cache()
    # t0 = record_data('step', t0, True)
    # print_function_runtime()

    result_loss = loss.data[0]

    return result_loss

def valid_step(model, _data_loader, loss_fn):
    model.eval() 
    valid_size, loss, detail = 0, 0.0, None 
    with torch.no_grad: 
        for val_data in _data_loader:
            _batch_size = len(val_data.E) 
            E_pred, F_pred, Q_pred, D_pred, loss_nh = model(val_data) 
            aggr_loss, loss_detail = loss_fn(E_pred, F_pred, Q_pred, D_pred, val_data, require_detail=True) 
            loss += aggr_loss.item()* _batch_size 
            valid_size += _batch_size 

            if detail is None:
                detail = copy(loss_detail) 
                for key in detail:
                    detail[key] = 0. 
        
            for key in detail:
                detail[key] += loss_detail[key] * _batch_size 
    
    loss /= valid_size 
    for key in detail:
        detail[key] /= valid_size 
    detail["loss"] = loss 
    return detail 



def data_provider_solver(name_full, _kw_args):
    additional_kwargs = option_solver(name_full) 
    for key in additional_kwargs.keys():
        if additional_kwargs[key] in ["True", "False"]:
            additional_kwargs[key] = (additional_kwargs[key] == "True") 
        else:
            try:
                additional_kwargs[key] = float(additional_kwargs[key]) 
            except ValueError:
                pass 
    base_name:str = name_full.split('[')[0] 
    for key in additional_kwargs.keys():
        _kw_args[key] = additional_kwargs[key] 
    
    if base_name.lower() == 'qm9':
        return DummyIMDataset, _kw_args
    elif base_name.lower() == 'alchemy':
        return AlchemyData, _kw_args 
    else:
        raise ValueError('Unrecognized dataset name: {} .'.format(base_name)) 


def kwargs_solver(args):
    debug_mode = (args.debug_mode.lower() == 'true')
    NetKwargs = { 
        'num_embedding': args.num_embedding,
        'num_feature' : args.num_feature, 
        'num_output': args.num_output,
        'num_lin_out': args.num_lin_out,
        'num_lin_res_atom': args.num_lin_res_atom, 
        'num_lin_res_interact': args.num_lin_res_interact, 
        'num_lin_res_output': args.num_lin_res_output, 
        'activations': args.activations,
        'debug_mode': debug_mode,
        'num_modules': args.num_modules,
        'action': args.action,
        'target_name': args.target_name
    }
    return NetKwargs
