from __future__ import print_function
from codecs import ignore_errors
from genericpath import isfile
import os, sys, shutil, time, glob
import os.path as osp 
import argparse, logging 
from datetime import datetime 
from copy import copy
from sympy import remove_handler 

import torch
import torch.cuda, torch.utils.data  
from tqdm import tqdm 

from BasicUtility.TrainingUtil import add_parser_args, data_provider_solver, _add_arg_from_config, kwargs_solver, \
    MyRandomSampler, train_step, valid_step 
from BasicUtility.UtilFunc import device, floating_type, get_model_params, get_lr 
from BasicUtility.LossFunc import LossFn 
from PhysNet import PhysNetDemo

default_kwargs = {'root': '.', 'net_version': 'atom' ,'pre_transform': 'custom_pre_transform'}

def train(config_args, data_provider, dataset_setup, validing):
    net_kwargs = kwargs_solver(config_args) 
    config_dict = vars(net_kwargs) 
    # -------- set up running directory -------- # 
    while True:
        current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S') 
        run_directory = config_dict["folder_prefix"] + '_run_' + current_time 
        if not osp.exists(run_directory):
            os.mkdir(run_directory) 
            break 
        else:
            time.sleep(10)
    
    shutil.copyfile(config_dict["config_name"], osp.join(run_directory, config_dict['config_name'])) 

    # -------------- Logger setup -------------- #
    logging.basicConfig( 
        filename=osp.join(run_directory, config_dict['config_name']), 
        format = '%(asctime)s %(message)s', filemode = 'w'
    )
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG) 

    # ---------- Meta data file setup ---------- # 
    meta_data_name = osp.join(run_directory, 'meta.txt') 
    
    # ----------- Dataset Preparation ---------- #
    valid_dataset_list = [] 
    for valid_idx in range(config_dict["valid_num"]):
        valid_dataset_list.append(data_provider(mode='valid_{:02d}'.format(valid_idx), train_csv_path='./raw/train.csv', **dataset_setup)) 
    dev_dataset = data_provider(mode='dev', train_csv_path='./raw/train.csv', **dataset_setup) 
    test_dataset = data_provider(mode='test', **dataset_setup) 

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config_dict['batch_size']) 

    train_datasets, valid_dataset = [dev_dataset], NotImplemented 
    for idx in range(config_dict["valid_num"]):
        if idx == validing:
            valid_dataset = valid_dataset_list[idx] 
        else:
            train_datasets.append(valid_dataset_list[idx]) 
    train_dataset = torch.utils.data.ConcatDataset(train_datasets) 

    train_size, valid_size = len(train_dataset.data.E), len(valid_dataset.data.E) 
    logger.info('train size: {} \nvalid size: {} \n'.format(train_size, valid_size)) 
    num_train_batches = train_size // config_dict['batch_size'] + 1  


    train_sampler = MyRandomSampler(train_dataset) 

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config_dict['batch_size'], sampler=train_sampler) 
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config_dict['valid_batch_size']) 

    # ------------ Loss Function ------------- # 
    w_e, w_f, w_q, w_p = 1., config_dict['force_weight'], config_dict['charge_weight'], config_dict['dipole_weight']
    loss_fn = LossFn(w_e=w_e, w_f=w_f, w_q=w_q, w_p=w_p, action=config_dict['action'], target_names=config_dict['target_names']) 

    # ------------- Model Set up ------------- # 
    net = PhysNetDemo(**net_kwargs)
    net = net.to(device).type(floating_type) 
    shadow_net = PhysNetDemo(**net_kwargs) 
    shadow_net = shadow_net.to(device).type(floating_type) 

    # use pre-trained model
    if config_dict["use_trained_model"]:
        trained_modeldir = glob.glob(config_dict["use_trained_model"])[0] 
        logger.info('using trained model: {}'.format(config_dict['use_trained_model'])) 
        if osp.exists(osp.join(trained_modeldir, 'trained_model.pt')):
            net.load_state_dict(torch.load(osp.join(trained_modeldir, 'trained_model.pt'), map_location=device), strict=False) 
        else:
            net.load_state_dict(torch.load(osp.join(trained_modeldir, 'best_model.pt'), map_location=device), strict=False) 
        incompatible_keys = shadow_net.load_state_dict(
            torch.load(osp.join(trained_modeldir, 'best_model.pt'), map_location=device), 
            strict=False) 
        logger.info("-------- incompatible keys: ---------") 
        logger.info(str(incompatible_keys)) 
    else:
        trained_modeldir = None 
    
    # optimizer 
    if config_dict['optimizer'].split('_')[0] == 'Adam':
        optimizer = torch.optim.Adam(
            net.parameters(), lr=config_dict["learning_rate"], ema=float(config_dict["optimizer"].split('_')[1]), 
            betas=(0.9, 0.99), weight_decay=0, amsgrad=True 
        )
    elif config_dict['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config_dict['learning_rate']) 
    else:
        raise ValueError('Unrecognized optimizer: {}'.format(config_dict['optimizer'])) 
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config_dict['decay_steps'], gamma=0.1) 

    # ------------ Printing meta data ------------- #
    if torch.cuda.is_available():
        logger.info('device name: ' + torch.cuda.get_device_name(device))
        logger.info("Cuda mem allocated: {:.2f} MB".format(torch.cuda.memory_allocated(device) * 1e-6))

    with open(meta_data_name, 'w+') as f:
        n_parm, model_structure = get_model_params(net, None)
        logger.info('model params: {}'.format(n_parm))
        f.write('*' * 20 + '\n')
        f.write(model_structure)
        f.write('*' * 20 + '\n')
       
        for _key in net_kwargs.keys():
            f.write("{} = {}\n".format(_key, net_kwargs[_key]))

    # ---------------- training ----------------- # 
    logger.info('start training...') 

    t0 = time.time() 
    val_loss = valid_step(net, valid_loader, loss_fn) 

    with open(osp.join(run_directory, "loss_data.csv"), 'a') as f:
        f.write(" ephoc \t train_loss \t valid_loss \t delta_time \t")
        for key in val_loss.keys():
            if key != "loss":
                f.write(",{}\t".format(val_loss[key]))
        f.write("\n") 
        
        f.write("0, \t -1, \t {},\t {}\t".format(val_loss["loss"], time.time-t0)) 
        for key in val_loss.keys():
            if key != "loss":
                f.write(",{}\t".format(val_loss[key])) 
        f.write("\n") 
    
    best_loss = val_loss["loss"] 

    logger.info('Init learning rate = {}'.format(get_lr))
    loss_data = [] 
    early_stop_count = 0 
    
    for epoch in range(config_dict['num_epochs']):
        train_loss = 0. 
        loader = tqdm(train_loader, "epoch: {}".format(epoch)) 
        for batch_num, data in loader:
            cur_size = data.E.shape[0] 
            train_loss += train_step(
                model=net, _optimizer=optimizer, data_batch=data, 
                loss_fn=loss_fn, max_norm=config_dict['max_norm'] 
            ) 
            if config_dict['debug_mode'] & ((batch_num+1) % 300 == 0):
                logger.info("Batch num: {}/{}, training loss:{}".format(batch_num, num_train_batches, train_loss )) 
        logger.info('epoch {} ends, learning rate: {}'.format(epoch, get_lr(optimizer))) 
        val_loss = valid_step(model=net, _data_loader=valid_loader, loss_fn=loss_fn) 

        _loss_data_this_epoch = {'epoch':epoch, 'train_loss':train_loss, 'valid_loss':val_loss['loss'], 'time':time.time()} 
        _loss_data_this_epoch.update(val_loss) 
        loss_data.append(_loss_data_this_epoch) 
        torch.save(loss_data, osp.join(run_directory, 'loss_data.pt')) 
        with open(osp.join(run_directory, "loss_data.csv"), "a") as f:
            f.write("{},{},{},{}".format(epoch, train_loss, val_loss["loss"], time.time()-t0))
            t0 = time.time()
            for key in val_loss.keys():
                if key != "loss":
                    f.write(",{}".format(val_loss[key]))
            f.write("\n") 
        # record best model and early stop  
        if val_loss['loss'] < best_loss:
            best_loss =val_loss['loss'] 
            torch.save(net.state_dict(), osp.join(run_directory, 'best_model.pt')) 
            torch.save(optimizer.state_dict(), osp.join(run_directory, 'best_model_optim.pt'))
            early_stop_count = 0 
        else:
            early_stop_count += 1 
            if early_stop_count == config_dict['early_stop']:
                logger.info('early stop at epoch {}.'.format(epoch)) 
                break 

    remove_handler(logger)  


def main(default_kwargs=default_kwargs, validing:int=0):

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@') 
    parser = add_parser_args(parser) 

    # parse the config file 
    config_name = 'config.txt' 
    if len(sys.argv) == 1:
        if osp.isfile(config_name):
            args, unknown = parser.parse_known_args(["@" + config_name]) 
        else:
            raise Exception('couldn\'t find file \"config.txt\".') 
    else:
        args = parser.parse_args() 
        config_name = args.config_name 
        args, unknown = parser.parse_known_args(["@" + config_name]) 
    args.config_name = config_name 

    data_provider_class, _kwargs = data_provider_solver(args.data_provider, default_kwargs) 
    _kwargs = _add_arg_from_config(_kwargs, args) 
    # data_provider = data_provider_class(**_kwargs) 
    train(config_args=args, data_provider=data_provider_class, dataset_setup=_kwargs, validing=validing) 

    print("finished!") 

if __name__ == '__main__':
    main()


