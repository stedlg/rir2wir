#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 2024

@author: stedlg

---

Froom Room Impulse Responses to Wall Impulse Responses (RIR2WIR)

This script implements an alternating optimization algorithm to estimate Wall Impulse Responses (WIRs)
from Room Impulse Responses (RIRs). It handles geometrical uncertainties, spatial frequency dependent responses of devices,
and background noise. The system supports multi-GPU execution
and tracks performance metrics (MAE, MSE, CE) across batches.

Key Components:
- Data preparation with noise and geometric uncertainty handling
- Alternating optimization: delay estimation (gradient descent) and WIR estimation (Conjugate Gradient Method)
- Constraint projection for physical realizability
- Comprehensive evaluation and result aggregation
- Save results (format: {dataset}_{sigma_geo}_nq{noise_q}_{mode}_lr{lr}_n_obs{n_obs}_blur_std{blur_std}_psnr{psnr}.pt)
"""

#%% Paths

path = "your_path/" # path to the folder

dataset_name = "hybrid" # "hybrid" or "ideal"
path_data = path+"data/"+dataset_name+"/"
 
path_results = path+"results/"

#%% Modules 

import sys 

sys.path.append(path+"code/")

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import gc

import functions as f
import alternating_optimization as g

#%% Parameters 

# Global parameters 

fs = 16000
sound_speed = 343
simulator_offset = 40
dtype=torch.float32 

# Setting parameters 

n_obs = 16 # number of RIRs used for the estimaiton 
mode = "standard" # if "oracle", the WIRs are known for the optimization of the delays
mode_alternance = "projection" # if "standard", constraint projection is not implemented 
blur_std = 1
psnr = 50 
scaling = True # scaling the RIRs (not required)
noise_q = False # geometrical uncertainy affects the positions of the walls and the devices (True) or only the devices (False)
sigma_geo = 0.02 # std of the geometrical uncertainty (in cm)`
batch_size_room = 10

# Optimization parameters

max_iter_alternate = 3 # maximum number of iterations for alternating optimization

# Delays 
max_iter =  500 # maximum number of iterations for delay optimization
previous = 50 # reference for early stoping is taken previous iterations before
patience = 60 # minimum number of iterations
threshold = 1e-2 # the optimization stops if the current loss is larger (1-threshold)
lr = 0.01 # learning rate for delay optimization 

# WIRs
n_iter_cgm = 1000 # number of iterations for WIRs optimization
L = 32  

MSE_loss = nn.MSELoss()    
MAE_loss = nn.L1Loss() 

global_params = {'fs' : fs,'sound_speed' : sound_speed,'simulator_offset' : simulator_offset,'dtype':dtype}

setting_params = {'dataset_name' : dataset_name,'mode' : mode,'mode_alternance' : mode_alternance,'psnr' : psnr,'blur_std' : blur_std,'n_obs' : n_obs,'noise_q' : noise_q,'sigma_geo' :sigma_geo}

optim_params = {'L' : L,'lr' : lr,'previous' : previous,'max_iter' :  max_iter,'patience' : patience,'threshold' : threshold, 'MSE_loss':MSE_loss,'MAE_loss':MAE_loss,
'max_iter_alternate' : max_iter_alternate ,'n_iter_cgm' :  n_iter_cgm}

#%% Data preparation

X_dataset,H_dataset,V_dataset,q_dataset,mask_dataset,real_toa_dataset,measured_toa_dataset,n_reflections_walls_dataset,dT,delay_max = f.data_preparation(path_data,n_obs,psnr,noise_q,sigma_geo,sound_speed,dtype,fs,simulator_offset,scaling)                

#%% Devices and module objects (parallelization on GPUs)

device_ids = [0,1] # gpu ids : if not found, the program will run on cpu
print("devices : ", device_ids)
main_device = device_ids[0]
device = torch.device("cuda:"+str(main_device) if torch.cuda.is_available() else "cpu")

print("device : ", device)

module_optim = f.ModuleOptim()
module_cgm = f.ModuleCGM()
module_correction = f.ModuleCorrection()
module_reconstruction = f.ModuleReconstruction()
module_conv1d = f.ModuleConv1d()

if torch.cuda.is_available():
    
    module_optim = nn.DataParallel(module_optim, device_ids=device_ids)
    module_cgm = nn.DataParallel(module_cgm, device_ids=device_ids)
    module_correction = nn.DataParallel(module_correction, device_ids=device_ids)
    module_reconstruction = nn.DataParallel(module_reconstruction, device_ids=device_ids)
    module_conv1d = nn.DataParallel(module_conv1d, device_ids=device_ids)

#%% Alternating optimization algorithm 

n_batches = X_dataset.shape[0]//batch_size_room # number of room bacths            

print(" \n Beginning of the experiment")

V_walls_estimate__ = [] # store the estimated WIRs over room batches

for n_batch in tqdm(range(n_batches)):
    
    ### Batch selection and processing
    
    x,H_noise,v,q,mask,real_toa,measured_toa,n_reflections_walls,dT = X_dataset[n_batch*batch_size_room:(n_batch+1)*batch_size_room,...],H_dataset[n_batch*batch_size_room:(n_batch+1)*batch_size_room,...],V_dataset[n_batch*batch_size_room:(n_batch+1)*batch_size_room,...],q_dataset[n_batch*batch_size_room:(n_batch+1)*batch_size_room,...],mask_dataset[n_batch*batch_size_room:(n_batch+1)*batch_size_room,...],real_toa_dataset[n_batch*batch_size_room:(n_batch+1)*batch_size_room,...].nan_to_num(),measured_toa_dataset[n_batch*batch_size_room:(n_batch+1)*batch_size_room,...].nan_to_num(),n_reflections_walls_dataset[n_batch*batch_size_room:(n_batch+1)*batch_size_room,...],dT
    
    q[mask.sum(axis = 1)==0] = torch.nan
    v[mask.sum(axis = 1)==0,:] = 0
    delta = measured_toa - real_toa
    Q_max = q.nan_to_num().max().int().item()
    
    #### Gaussian filtering
    
    if not(blur_std == 0):
        
         x_blurred,H_noise_blurred = f.gaussian_blurring(x,H_noise,blur_std,fs,dtype,device,simulator_offset)
        
    else : 
        
        H_noise_blurred = H_noise
        x_blurred = x
    
    x_blurred,H_noise_blurred,v,q,mask,real_toa,measured_toa = x_blurred.to(device),H_noise_blurred.to(device),v.to(device),q.to(device),mask.to(device),real_toa.to(device),measured_toa.to(device)
 
    ### Alternating optimization algorithm
    
    V_walls_estimate_,residual_0_,residual_init_,residual_end_,mae_target_0_,mae_target_init_,mae_target_end_,ce_target_0_,ce_target_init_,ce_target_end_,mae_0_,mae_init_,mae_end_,ce_0_,ce_init_,ce_end_,mse_v_t_target_0_,mse_v_t_target_init_,mse_v_t_target_end_,mse_v_t_0_,mse_v_t_init_,mse_v_t_end_,mae_target_f_0_,mae_target_f_init_,mae_target_f_end_,ce_target_f_0_,ce_target_f_init_,ce_target_f_end_,mae_f_0_,mae_f_init_,mae_f_end_,ce_f_0_,ce_f_init_,ce_f_end_,losses_,losses_delta_mae_,losses_delta_ce_,loss_0_ = g.alternating_optimization(x_blurred,H_noise_blurred,measured_toa,delta,module_reconstruction,module_correction,module_cgm,batch_size_room,delay_max,x,H_noise,Q_max,mask,v,q,L,n_reflections_walls,module_conv1d,device,global_params,setting_params,optim_params)
 
    ### Aggregating metrics over each batches
    
    if n_batch == 0 :
        
        losses__ = np.array(losses_)
        loss_0__ = np.array(loss_0_)
        losses_delta_mae__ = np.array(losses_delta_mae_)
        losses_delta_ce__ = np.array(losses_delta_ce_)
    
        residual_0__ = np.array(residual_0_)
        
        residual_init__ = np.array(residual_init_)
        residual_end__ = np.array(residual_end_)
        
        mae_target_0__ = np.array(mae_target_0_)
        mae_target_init__ = np.array(mae_target_init_)
        mae_target_end__ = np.array(mae_target_end_)
        
        ce_target_0__ = np.array(ce_target_0_)
        ce_target_init__ = np.array(ce_target_init_)
        ce_target_end__ = np.array(ce_target_end_)
        mae_0__ = np.array(mae_0_)
        mae_init__ = np.array(mae_init_)
        mae_end__ = np.array(mae_end_)
        ce_0__ = np.array(ce_0_)
        ce_init__ = np.array(ce_init_)
        ce_end__ = np.array(ce_end_) 
        mse_v_t_target_0__ = np.array(mse_v_t_target_0_) 
        mse_v_t_target_init__ = np.array(mse_v_t_target_init_) 
        mse_v_t_target_end__ = np.array(mse_v_t_target_end_) 
        mse_v_t_0__ = np.array(mse_v_t_0_) 
        mse_v_t_init__ = np.array(mse_v_t_init_)
        mse_v_t_end__ = np.array(mse_v_t_end_)
        
        mae_target_f_0__ = np.array(mae_target_f_0_)
        mae_target_f_init__ = np.array(mae_target_f_init_)
        mae_target_f_end__ = np.array(mae_target_f_end_)
        ce_target_f_0__ = np.array(ce_target_f_0_)
        ce_target_f_init__ = np.array(ce_target_f_init_)
        ce_target_f_end__ = np.array(ce_target_f_end_)
        mae_f_0__ = np.array(mae_f_0_)
        mae_f_init__ = np.array(mae_f_init_)
        mae_f_end__ = np.array(mae_f_end_)
        ce_f_0__ = np.array(ce_f_0_)
        ce_f_init__ = np.array(ce_f_init_)
        ce_f_end__ = np.array(ce_f_end_) 

    else : 
        
        losses__ += np.array(losses_)
        loss_0__ += np.array(loss_0_)
        losses_delta_mae__ += np.array(losses_delta_mae_)
        losses_delta_ce__ += np.array(losses_delta_ce_)
        
        residual_0__ += np.array(residual_0_)
        residual_init__ += np.array(residual_init_)
        residual_end__ += np.array(residual_end_)
        
        mae_target_0__ += np.array(mae_target_0_)
        mae_target_init__ += np.array(mae_target_init_)
        mae_target_end__ += np.array(mae_target_end_)
        
        ce_target_0__ += np.array(ce_target_0_)
        ce_target_init__ += np.array(ce_target_init_)
        ce_target_end__ += np.array(ce_target_end_)
        mae_0__ += np.array(mae_0_)
        mae_init__ += np.array(mae_init_)
        mae_end__ += np.array(mae_end_)
        ce_0__ += np.array(ce_0_)
        ce_init__ += np.array(ce_init_)
        ce_end__ += np.array(ce_end_) 
        mse_v_t_target_0__ += np.array(mse_v_t_target_0_) 
        mse_v_t_target_init__ += np.array(mse_v_t_target_init_)
        mse_v_t_target_end__ += np.array(mse_v_t_target_end_) 
        mse_v_t_0__ += np.array(mse_v_t_0_) 
        mse_v_t_init__ += np.array(mse_v_t_init_) 
        mse_v_t_end__ += np.array(mse_v_t_end_)
        
        mae_target_f_0__ += np.array(mae_target_f_0_)
        mae_target_f_init__ += np.array(mae_target_f_init_)
        mae_target_f_end__ += np.array(mae_target_f_end_)
        ce_target_f_0__ += np.array(ce_target_f_0_)
        ce_target_f_init__ += np.array(ce_target_f_init_)
        ce_target_f_end__ += np.array(ce_target_f_end_)
        mae_f_0__ += np.array(mae_f_0_)
        mae_f_init__ += np.array(mae_f_init_)
        mae_f_end__ += np.array(mae_f_end_)
        ce_f_0__ += np.array(ce_f_0_)
        ce_f_init__ += np.array(ce_f_init_)
        ce_f_end__ += np.array(ce_f_end_)
     
    V_walls_estimate__.append(np.array(V_walls_estimate_))
  
print(" \n End of the experiment")

#%% Avergaging metrics over batches
    
losses__ = losses__ / n_batches
loss_0__ = loss_0__ / n_batches
losses_delta_mae__ = losses_delta_mae__ / n_batches
losses_delta_ce__ = losses_delta_ce__ / n_batches

residual_0__ = residual_0__ / n_batches
residual_init__ = residual_init__ / n_batches
residual_end__ = residual_end__ / n_batches
mae_target_0__ = mae_target_0__ / n_batches
mae_target_init__ = mae_target_init__ / n_batches
mae_target_end__ = mae_target_end__ / n_batches
ce_target_0__ = ce_target_0__ / n_batches
ce_target_init__ = ce_target_init__ / n_batches
ce_target_end__ = ce_target_end__ / n_batches

mae_0__ = mae_0__ / n_batches
mae_init__ = mae_init__ / n_batches
mae_end__ = mae_end__ / n_batches
ce_0__ = ce_0__ / n_batches
ce_init__ = ce_init__ / n_batches
ce_end__ = ce_end__ / n_batches 
mse_v_t_target_0__ = mse_v_t_target_0__ / n_batches 
mse_v_t_target_init__ = mse_v_t_target_init__ / n_batches 
mse_v_t_target_end__ = mse_v_t_target_end__ / n_batches 
mse_v_t_0__ = mse_v_t_0__ / n_batches 
mse_v_t_init__ = mse_v_t_init__ / n_batches 
mse_v_t_end__ = mse_v_t_end__ / n_batches

mae_target_f_0__ = mae_target_f_0__ / n_batches
mae_target_f_init__ = mae_target_f_init__ / n_batches
mae_target_f_end__ = mae_target_f_end__ / n_batches
ce_target_f_0__ = ce_target_f_0__ / n_batches
ce_target_f_init__ = ce_target_f_init__ / n_batches
ce_target_f_end__ = ce_target_f_end__ / n_batches
mae_f_0__ = mae_f_0__ / n_batches
mae_f_init__ = mae_f_init__ / n_batches
mae_f_end__ = mae_f_end__ / n_batches
ce_f_0__ = ce_f_0__ / n_batches
ce_f_init__ = ce_f_init__ / n_batches
ce_f_end__ = ce_f_end__ / n_batches
                                          
V_walls_estimate__ = np.array(V_walls_estimate__)

#%% Saving results

torch.save({"n_iter_cgm" : n_iter_cgm,
            "lr":lr,
            "mode":mode,
            "n_obs":n_obs,
            "dataset_name":dataset_name,
            "blur_std":blur_std,
            "noise_q":noise_q,
            "sigma_geo":sigma_geo,
            "L":L,
            "n_obs":n_obs,
            'psnr':psnr,
            'max_iter': max_iter,
            'max_iter_alternate': max_iter_alternate,
            "n_batches":n_batches,
            "n_rooms":X_dataset.shape[0],
            
            "loss_0__":loss_0__,
            'losses__':losses__,
            'losses_delta_mae__': losses_delta_mae__,
            'losses_delta_ce__': losses_delta_ce__,
            "mae_target_0__":mae_target_0__,
            "mae_target_init__":mae_target_init__,
            "mae_target_end__":mae_target_end__,
            "ce_target_0__":ce_target_0__,
            "ce_target_init__":ce_target_init__,
            "ce_target_end__":ce_target_end__,
            "mae_0__":mae_0__,
            "mae_init__":mae_init__,
            "mae_end__":mae_end__,
            "ce_0__":ce_0__,
            "ce_init__":ce_init__,
            "ce_end__":ce_end__,
            "residual_0__":residual_0__,
            "residual_init__":residual_init__,
            "residual_end__":residual_end__,
            "mse_v_t_target_0__":mse_v_t_target_0__,
            "mse_v_t_target_init__":mse_v_t_target_init__,
            "mse_v_t_target_end__":mse_v_t_target_end__,
            "mse_v_t_0__":mse_v_t_0__,
            "mse_v_t_init__":mse_v_t_init__,
            "mse_v_t_end__":mse_v_t_end__,
            "mae_target_f_0__":mae_target_f_0__,
            "mae_target_f_init__":mae_target_f_init__,
            "mae_target_f_end__":mae_target_f_end__,
            "ce_target_f_0__":ce_target_f_0__,
            "ce_target_f_init__":ce_target_f_init__,
            "ce_target_f_end__":ce_target_f_end__,
            "mae_f_0__":mae_f_0__,
            "mae_f_init__":mae_f_init__,
            "mae_f_end__":mae_f_end__,
            "ce_f_0__":ce_f_0__,
            "ce_f_init__":ce_f_init__,
            "ce_f_end__":ce_f_end__,
            
            "V_walls_truth":V_dataset[q_dataset==1].reshape(X_dataset.shape[0],-1,V_dataset.shape[-1]).cpu().numpy(),
            "V_walls_cgm":V_walls_estimate__
             },
           
            path_results+dataset_name[:3]+'_std'+str(sigma_geo)+'_nq'+str(int(noise_q))+'_'+mode+'_lr'+str(lr)+'_n_obs'+str(n_obs)+'_blur_std'+str(blur_std)+'_psnr'+str(psnr)+'.pt')