#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 2024

@author: stedlg

--- 

RIR2WIR Toolkit

This module provides functions and classes for  
fractional delay filtering, and forward operator correction, 
data preparation, Gaussian blurring, and optimization modules
for delay estimation and signal reconstruction.

Key functionalities:
- Windowed sinc-based fractional delay filters
- Forward operator correction with Gaussian kernel support
- Additive white Gaussian noise addition
- Dataset preparation for optimization
- Neural network modules for delay estimation and signal reconstruction
"""

#%% Modules 

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc 
import alternating_optimization as g

#%% Functions 
  
def windowed_sinc(T,simulator_offset,blur_std = 0):
    
    """
    Windowed sinc function. 
    (from the article "PYROOMACOUSTICS: A PYTHON PACKAGE FOR AUDIO ROOM SIMULATION AND ARRAY PROCESSING ALGORITHMS")
    
    Parameters
    ----------
    
    T: tensor, time axis ("fractional samples")
    
    simulator_offset: int, Constant simulator offset (in pyroomacoustics : 40 samples)
    
    blur_std: float, If zero, the kernel is a sinc. Else, standard deviation of the gaussian kernel (in samples). 
    
    Outputs
    ----------
    kernel : tensor, sinc on gaussian kernel
   
    """
    
    if (blur_std == 0) :
        return (1/2)*(1+torch.cos(2*torch.pi*T/(2*simulator_offset)))*torch.sinc(T)
    else : 
        kernel = torch.exp((-1)*(T**2) / (2 * blur_std**2))
        return kernel #equivalent to kernel / kernel.sum() 
        
def windowed_sinc_delay_filters(old_toa,delta,fs,delay_max,simulator_offset,flip_negative_delays=True,device = "cpu",finite_support = True,blur_std = 0):
    
    """
    Fractional delay filters. 
    
    Parameters
    ----------

    old_toa: tensor, Current parameters of the forward operator (here old times of arrival of the image source (old_toas))
    new_toa: tensor, Updated parameters for the forward operator (here new times of arrival of the image source (new_toas))
    fs: int, Sampling frequency
    delay_max: float, Upper bound for the absolute delay to apply on the components of the forward operator.
    simulator_offset: int, Constant simulator offset (in pyroomacoustics : 40 samples)
    flip_negative_delays : boolean, flip rows corresponding to negative delays
    device: string, Device which runs the computation 
    finite_support: boolean, Is the support of the filter finite? 
    blur_std: float, If zero, the kernel is a sinc. Else, standard deviation of the gaussian kernel(in samples). 

    Outputs
    ----------
    win_sinc : tensor, Fractional delay filters 
    """
    
    B,M,K = 1,old_toa.shape[-2],old_toa.shape[-1]
    if len(old_toa.shape) == 3:
        
        # batch size (number of rooms)
        
        B = old_toa.shape[-3]

    fractional_delays = (delta*fs).reshape(B*M*K)
    
    T = torch.arange(-simulator_offset,int(delay_max*fs)+simulator_offset,device=device).repeat(B*M*K,1)-torch.abs(fractional_delays).unsqueeze(1)
    
    new_toa = delta+old_toa 
    
    new_toa[new_toa==0] = 1
    
    win_sinc = (old_toa/(new_toa)).reshape(B*M*K).unsqueeze(1)*windowed_sinc(T,simulator_offset,blur_std)
    
    if finite_support: 
        
        idx = torch.arange(0,win_sinc.shape[-1],device=device).repeat(B*M*K,1)
        mask_to_zero = (idx<(win_sinc.argmax(dim=-1).unsqueeze(-1)-simulator_offset)) | (idx.squeeze()>(win_sinc.argmax(dim=-1).unsqueeze(-1)+simulator_offset))
        win_sinc[mask_to_zero] = 0
        
    if flip_negative_delays:
        win_sinc[fractional_delays<0,:] = win_sinc[fractional_delays<0,:].flip(1)

    return win_sinc

def forward_operator_correction(H,old_toa,delta,fs,delay_max,simulator_offset,q,v_truth,targets_at_top=True,device = "cpu",finite_support = True,blur_std = 0,n_reflections_walls = None):
    
    """
    Correction of the forward operator. 
    
    It applies fractional delays to the echoes. 
    It also allows to filter by a gaussian kernel (blur_std != 0)
    Plus, it allows to place the first-order image sources at the top of the Tensor : it is required for estimation but it is already done on the attached data.
    
    The Pytorch implementation allows to backpropagate gradients for optimization 
    and to split computation on gpus with convinience.
    
    Parameters
    ----------
    
    H: tensor, Forward operator (here geometry and device responses / (dims : M,K,T or B,M,K,T))
    old_toa: tensor, Current parameters of the forward operator (here old times of arrival of the image source (old_toas)) / (dims : M,K or B,M,K))
    delta: tensor, Updated parameters for the forward operator (here the difference betwen new and old times of arrival of the image sources (delta)) / (dims : M,K or B,M,K))
    fs: int, Sampling frequency
    delay_max: float, Upper bound for the absolute delay to apply on the components of the forward operator
    simulator_offset: int, Constant simulator offset (in pyroomacoustics : 40 samples)
    q: tensor, Reflection order of the image sources (dims : K or B,K))
    targets_at_top: boolean, Placing zero and first-order image sources at the top of the tensors to simplify training
    device: string, Device which runs computation 
    finite_support: boolean, Is the support of the fractional delay filter finite? 
    blur_std: float, If zero, the kernel is a sinc. Else, standard deviation of the gaussian kernel (in samples). 
    n_reflections_walls: tensor, number of reflections on each wall for each image source
    
    Outputs
    ----------
    H_tilde : tensor, Corrected forward operator 
    v_truth: tensor, ground truth of the wall impulse reponses  
    q: tensor, reflection order of the image sources
    old_toa: tensor, intital toas of the echoes
    delta: tensor, time delays to apply to the echoes 
    Phi: fractional tensor, delay filters which are applied to the echoes
    n_reflections_walls : tensor, number of reflections on each wall for each image source
        
    """
     
    B,M,K,P = 1,H.shape[-3],H.shape[-2],H.shape[-1]
    if len(H.shape) == 4:
        # batch size (number of rooms)
        B = H.shape[-4]
   
    H = H.reshape(B*M*K,-1)
    old_toa = old_toa.reshape(B,M,K)
    delta = delta.reshape(B,M,K)
    
    if n_reflections_walls is not None : 
        S = n_reflections_walls.shape[0]
        n_reflections_walls = n_reflections_walls.reshape(B,-1,K)
    
    if q is not None : 
        q = q.reshape(B,K)

    # delays are reshaped to B*M*K inside this functions
   
    Phi = windowed_sinc_delay_filters(old_toa,delta,fs,delay_max,simulator_offset,flip_negative_delays=True,device = device,finite_support = finite_support,blur_std = blur_std)
    
    # groups = M*K --> Each input channel is convolved with the corresponding output channel only, then convolutions are concatenated
    
    H_tilde = F.conv1d(H.unsqueeze(0),Phi.flip(1).unsqueeze(1),padding = Phi.shape[-1]-1,groups = B*M*K).squeeze()
 
    # crop the result of the convolution according to the sign of the delay 
    
    fractional_delays = (delta*fs).reshape(B*M*K)
    
    if len(H_tilde.shape) == 1:
        H_tilde = H_tilde.unsqueeze(0)
 
    mask_sign = torch.ones(H_tilde.shape,device ='cpu')
    
    mask_sign[(fractional_delays>=0),:simulator_offset] = 0

    mask_sign[(fractional_delays>=0),-(mask_sign.shape[-1]-(P+simulator_offset)):] = 0
    
    mask_sign[(fractional_delays<0),:(mask_sign.shape[-1]-(P+simulator_offset))] = 0

    mask_sign[(fractional_delays<0),-simulator_offset:] = 0
    
    H_tilde,mask_sign,Phi = H_tilde.cpu(),mask_sign.cpu(),Phi.cpu()
    H_tilde = H_tilde[mask_sign == 1].reshape(H_tilde.shape[0],P)
    H_tilde = H_tilde.reshape(B,M,K,P)
    Phi = Phi.reshape(B,M,K,-1)
   
    D  = Phi.shape[-1] 
        
    if targets_at_top:
        
        H_tilde = torch.cat((H_tilde.transpose(-3,-2)[q<=1,:].reshape(B,-1,M,P).transpose(-3,-2),H_tilde.transpose(-3,-2)[(q>1)|(q.isnan()),:].reshape(B,-1,M,P).transpose(-3,-2)),dim = -2)
       
        old_toa = torch.cat((old_toa.transpose(-2,-1)[q<=1,...].reshape(B,-1,M).transpose(-2,-1),old_toa.transpose(-2,-1)[(q>1)|(q.isnan()),...].reshape(B,-1,M).transpose(-2,-1)),dim = -1)
        delta = torch.cat((delta.transpose(-2,-1)[q<=1,...].reshape(B,-1,M).transpose(-2,-1),delta.transpose(-2,-1)[(q>1)|(q.isnan()),...].reshape(B,-1,M).transpose(-2,-1)),dim = -1)
        
        if n_reflections_walls is not None : 
            n_reflections_walls = torch.cat((n_reflections_walls.transpose(-2,-1)[q<=1,...].reshape(B,-1,S).transpose(-2,-1),n_reflections_walls.transpose(-2,-1)[(q>1)|(q.isnan()),...].reshape(B,-1,S).transpose(-2,-1)),dim = -1)
        
        Phi = torch.cat((Phi.transpose(-3,-2)[q<=1,:].reshape(B,-1,M,D).transpose(-3,-2),Phi.transpose(-3,-2)[(q>1)|(q.isnan()),:].reshape(B,-1,M,D).transpose(-3,-2)),dim = -2)
        
        if v_truth is not None : 
    
           v_truth = v_truth.reshape(B,K,-1)
           L = v_truth.shape[-1]
           v_truth = torch.cat((v_truth[q<=1,:].reshape(B,-1,L),v_truth[(q>1)|(q.isnan()),:].reshape(B,-1,L)),dim = -2)
        
        if q is not None : 
            q = torch.cat((q[q<=1].reshape(B,-1),q[(q>1)|(q.isnan())].reshape(B,-1)),dim=-1)
            
    H_tilde,Phi = H_tilde.to(device),Phi.to(device)
    return [H_tilde,v_truth,q,old_toa,delta,Phi,n_reflections_walls]

def add_awgn_with_psnr(X, psnr=None,return_noise = False):
    
    """Adding additive white gaussian noise 
    to the room impulse responses
    
    Parameters
    ----------
    
    X: tensor, Room impulse responses
    psnr: float, Peak signal-to-noise ratio (in decibels)
    return_noise: boolean, Is the applied noise needed? 
    
    Outputs
    ----------
    X_noise : tensor, Noisy room impulse responses
    noisy : tensor, Noise applied to RIRs """
    
    if psnr is not None:
        # Calculate the standard deviation of the noise based on the given PSNR
        std = X.amax(axis=-1) * np.sqrt(10 ** (-psnr / 10))
        
        # Generate noise with the same shape as X from a Gaussian distribution
        noise = torch.randn(X.size())
        
        # Scale the noise using the calculated standard deviation
        noise *= std.unsqueeze(-1)
        
        # Add the noise to the original signal
        X_noisy = X + noise
        
    if return_noise :
        return [X_noisy,noise]
    else : 
        return X_noisy

def data_preparation(path_data,n_obs,psnr,noise_q,sigma_geo,sound_speed,dtype,fs,simulator_offset,scaling):
    
    """
    Loads and prepares dataset for optimization.
    
    Parameters
    ----------
    path_data: str
        Path to data files
    n_obs: int
        Number of observations to use
    psnr: float
        PSNR for noise addition
    noise_q: bool
        Whether to apply order-dependent noise
    sigma_geo: float
        Geometric uncertainty parameter
    sound_speed: float
        Speed of sound
    dtype: torch.dtype
        Data type
    fs: int
        Sampling frequency
    simulator_offset: int
        Simulator time offset
    scaling: bool
        Whether to normalize data
        
    Returns
    -------
    list
        Prepared dataset and parameters
    """
    
    flag = False
    
    files = os.listdir(path_data)
    
    for file in files:
        if os.path.isdir(path_data+file) or file.startswith('.')  : 
            continue
        
        data = torch.load(path_data+file)
        
        x,H,v,q,mask,real_toa,measured_toa,n_reflections_walls,dT = data["X_dataset"],data["H_dataset"],data["V_dataset"],data["q_dataset"],data["mask_dataset"],data["real_toa_dataset"],data["measured_toa_dataset"],data["n_reflections_walls_dataset"],data["dT"]
        
        indices = list(range(n_obs))
        indices_obs = torch.tensor(indices)
        
        if psnr is not None:
               
            x = add_awgn_with_psnr(x,psnr)
    
        if len(H.shape) == 4:
            B,M,K,P = H.shape
    
        #### geometrical uncertainty 
        
        # maximum absolute delay authorized during optimization 
        
        if not(noise_q) :  
            
            delay_max = 2 * (sigma_geo*(2)/sound_speed) # 2 : intervalle 95% / 3 : intervalle 99%
            
        else :  
            
            delay_max = 2 * (sigma_geo*(2+2)/sound_speed) # intervalle 95% les ordres 2.
             
        if (sigma_geo != 0): 
            
            if os.path.isfile(path_data+"geometrical_uncertainty/"+file[:-3]+"_nq"+str(int(noise_q))+".pt"):
                
                geometrical_uncertainty = torch.load(path_data+"geometrical_uncertainty/"+file[:-3]+"_nq"+str(int(noise_q))+".pt")
                measured_toa = geometrical_uncertainty["measured_toa"]
                print("load and apply " + file[:-3]+"_nq"+str(int(noise_q))+".pt")
            
            else : 
                
                measured_toa = real_toa + ((sigma_geo*(noise_q*q.unsqueeze(1)+2)/sound_speed)* torch.randn(real_toa.shape,dtype = dtype)) 
                torch.save({'measured_toa': measured_toa}, path_data+"geometrical_uncertainty/"+file[:-3]+"_nq"+str(int(noise_q))+".pt")
                print("save " + file[:-3]+"_nq"+str(int(noise_q))+".pt")
     
            measured_toa = measured_toa.to(dtype).nan_to_num()
            real_toa = real_toa.to(dtype).nan_to_num()
            
            data = {"X_dataset":x[:,indices_obs,:].cpu(), 
                    "H_dataset":H[:,indices_obs,...].cpu(),
                    "V_dataset":v,
                    "q_dataset":q,
                    "mask_dataset": mask[:,indices_obs,:],
                    "real_toa_dataset": real_toa[:,indices_obs,:],
                    "measured_toa_dataset": measured_toa[:,indices_obs,:],
                    'n_reflections_walls_dataset' : n_reflections_walls,
                    "dT":dT}
            
            del x,H,v,q,mask,real_toa,measured_toa,n_reflections_walls
            gc.collect()
            
            measured_toa,real_toa,q = data["measured_toa_dataset"],data["real_toa_dataset"],data["q_dataset"]
            
            delta = (measured_toa - real_toa)
            delta = delta.nan_to_num() 
            
            delay_max_noise = delta.abs().max()
            
            H_real = data["H_dataset"]
            H_noisy,_,_,_,_,*_ = forward_operator_correction(H_real.to(dtype).cpu(),real_toa.to(dtype).cpu(),delta.to(dtype).cpu(),fs,delay_max_noise.item(),simulator_offset,q=None,v_truth=None,targets_at_top=False,device=torch.device("cpu")) 
            
            H = H_noisy
    
        else : 
            
            data = {"X_dataset":x[:,indices_obs,:].cpu(), 
                    "H_dataset":H[:,indices_obs,...].cpu(),
                    "V_dataset":v,
                    "q_dataset":q,
                    "mask_dataset": mask[:,indices_obs,:],
                    "real_toa_dataset": real_toa[:,indices_obs,:],
                    "measured_toa_dataset": measured_toa[:,indices_obs,:],
                    'n_reflections_walls_dataset' : n_reflections_walls,
                    "dT":dT}
            
            del x,H,v,q,mask,real_toa,measured_toa,n_reflections_walls
            gc.collect()
            
            H = data["H_dataset"]
            measured_toa = data["real_toa_dataset"]
      
        if not(flag):
            
            X_shape = data["X_dataset"].shape[-1]
            X_dataset,H_dataset,V_dataset,q_dataset,mask_dataset,real_toa_dataset,measured_toa_dataset,n_reflections_walls_dataset,dT = data["X_dataset"],H,data["V_dataset"],data["q_dataset"],data["mask_dataset"],data["real_toa_dataset"],measured_toa,data["n_reflections_walls_dataset"],data["dT"]
            flag = True 
            
        else : 
            
            dU = data["X_dataset"].shape[-1]-X_dataset.shape[-1]
            dP = H.shape[-1]-H_dataset.shape[-1]
            dK = H.shape[-2]-H_dataset.shape[-2]
            
            X_dataset = torch.cat([F.pad(X_dataset,(0,max(0,dU))),F.pad(data["X_dataset"],(0,max(0,-dU)))],dim = 0)
            H_dataset = torch.cat([F.pad(H_dataset,(0,max(0,dP),0,max([0,dK]))),F.pad(H,(0, max(0,-dP),0,max([0,-dK])))],dim = 0)
            V_dataset = torch.cat([F.pad(V_dataset,(0,0,0,max([0,dK]))),F.pad(data["V_dataset"],(0,0,0,max([0,-dK])))],dim = 0) 
            
            val = torch.nan 
            q_dataset = torch.cat([F.pad(q_dataset,(0,max([0,dK])),value=val),F.pad(data["q_dataset"],(0,max([0,-dK])),value=val)],dim = 0)           
            n_reflections_walls_dataset = torch.cat([F.pad(n_reflections_walls_dataset,(0,max([0,dK])),value=val),F.pad(data["n_reflections_walls_dataset"],(0,max([0,-dK])),value=val)],dim = 0)   #.unsqueeze(0)    
    
            real_toa_dataset = torch.cat([F.pad(real_toa_dataset,(0,max([0,dK])),value=val),F.pad(data["real_toa_dataset"],(0,max([0,-dK])),value=val)],dim = 0) 
            measured_toa_dataset = torch.cat([F.pad(measured_toa_dataset,(0,max([0,dK])),value=val),F.pad(measured_toa,(0,max([0,-dK])),value=val)],dim = 0) 
            
            mask = torch.ones(measured_toa.shape,dtype=dtype)
            mask[H.sum(dim = -1) == 0] = 0
            mask_dataset = torch.cat([F.pad(mask_dataset,(0,max([0,dK]))),F.pad(mask,(0,max([0,-dK])))],dim = 0)     
    
    print('Data Loaded, Size training set : ' , X_dataset.shape[0])
    
    real_toa_dataset = real_toa_dataset.nan_to_num()
    measured_toa_dataset = measured_toa_dataset.nan_to_num()
    
    del H,real_toa,q,measured_toa
    del indices,indices_obs
    
    if (sigma_geo != 0):
        
        del H_real,H_noisy,delta
        
    gc.collect()
    
    if scaling :    
        print('scale data, max_rir')
        max_rirs = X_dataset.amax(dim = -1)
        X_dataset = X_dataset/max_rirs.unsqueeze(-1)
        H_dataset = H_dataset/max_rirs.unsqueeze(-1).unsqueeze(-1)
    
    return [X_dataset,H_dataset,V_dataset,q_dataset,mask_dataset,real_toa_dataset,measured_toa_dataset,n_reflections_walls_dataset,dT,delay_max]

def gaussian_blurring(x,H_noise,blur_std,fs,dtype,device,simulator_offset):
    
    """
    Applies Gaussian blur to signals and forward operator.
    
    Parameters
    ----------
    x: tensor
        Input signals
    H_noise: tensor
        Noisy forward operator
    blur_std: float
        Standard deviation for blur
    fs: int
        Sampling frequency
    dtype: torch.dtype
        Data type
    device: str
        Computation device
    simulator_offset: int
        Simulator time offset
        
    Returns
    -------
    list
        Blurred signals and operator
    """
    
    kernel_size = (4*blur_std)/fs
    delta_blur_x = torch.zeros((x.size(0),x.size(1),1),dtype = dtype,device = x.device) # pas de décalage
    real_toa_blur_x = torch.ones((x.size(0),x.size(1),1),dtype = dtype,device = x.device) # pas de normalisation
    delta_blur_H = torch.zeros((H_noise.size(0),H_noise.size(1),H_noise.size(2)),dtype = dtype,device = H_noise.device)
    real_toa_blur_H = torch.ones((H_noise.size(0),H_noise.size(1),H_noise.size(2)),dtype = dtype,device = H_noise.device)
    
    H_noise_blurred,_,_,_,_,*_ = forward_operator_correction(H_noise.to(dtype),real_toa_blur_H,delta_blur_H,fs,kernel_size,simulator_offset,q=None,v_truth=None,targets_at_top=False,device = "cpu",finite_support = False,blur_std = blur_std)
    x_blurred,_,_,_,_,*_ = forward_operator_correction(x.unsqueeze(2).to(dtype),real_toa_blur_x,delta_blur_x,fs,kernel_size,simulator_offset,q=None,v_truth=None,targets_at_top=False,device = "cpu",finite_support = False,blur_std = blur_std)
    
    x_blurred = x_blurred.squeeze()

    del delta_blur_x,delta_blur_H,real_toa_blur_x,real_toa_blur_H
    gc.collect()
    
    return [x_blurred,H_noise_blurred]

"""Modules allows to split computation on gpus 
(like neural networks)""" 

class ModuleOptim(nn.Module):
    
    """
    Optimization module for delay estimation.
    
    Performs forward pass of delay optimization using:
    - Fractional delay correction
    - Forward operator reconstruction
    """
    
    def __init__(self):
        super(ModuleOptim, self).__init__()  
    def forward(self,U,H_noise, v_known, measured_toa,simulator_offset,fs,delay_max,device):
        
        delta_estimate = delay_max*torch.tanh(U).nan_to_num()
        H_estimate,_,_,_,_,*_= forward_operator_correction(H_noise,measured_toa,delta_estimate,fs,delay_max,simulator_offset,q=None,v_truth=None,targets_at_top=False,device = H_noise.device)

        x_estimate = F.conv2d(F.pad(H_estimate.transpose(0,1),(v_known.shape[-1]-1,v_known.shape[-1]-1)),v_known.flip(-1).unsqueeze(1),groups = v_known.shape[0]).squeeze().transpose(0,1)
        
        return [x_estimate,H_estimate]
 
class ModuleCGM(nn.Module):
    
    """
    Conjugate Gradient Method module.
    
    Estimates wall impulse responses using Conjugate Gradient Method.
    """
    
    def __init__(self):
        super(ModuleCGM, self).__init__()
        
    def forward(self,n_iter_cgm,x,H_estimate,v_truth,q_truth,L,dtype,lambda_tikhonov = 0):
        
        v_cgm,residual,mae_target,ce_target,mae,ce,mse_v_t_target,mse_v_t,mae_target_f,ce_target_f,mae_f,ce_f = g.torch_cgm(n_iter_cgm,x,H_estimate,L,v_init=None,v_truth=v_truth,q = q_truth,device=H_estimate.device,dtype = dtype,lambda_tikhonov=lambda_tikhonov)  
        return [v_cgm,residual,mae_target,ce_target,mae,ce,mse_v_t_target,mse_v_t,mae_target_f,ce_target_f,mae_f,ce_f]

class ModuleCorrection(nn.Module): 
    
    """
    Forward operator correction module.
    
    Applies estimated delays to the forward operator.
    """
    
    def __init__(self):
        super(ModuleCorrection, self).__init__()   
    def forward(self,delta_estimate,H_noise,measured_toa,simulator_offset,fs,delay_max,device,blur_std = 0):
        
        finite_support = (blur_std == 0)
        H_estimate,_,_,_,_,*_= forward_operator_correction(H_noise,measured_toa,delta_estimate,fs,delay_max,simulator_offset,q=None,v_truth=None,targets_at_top=False,device = H_noise.device,finite_support = finite_support,blur_std = blur_std)
        return H_estimate
    
class ModuleReconstruction(nn.Module):
    
    """
    Signal reconstruction module.
    
    Reconstructs signals from corrected forward operator.
    """
    
    def __init__(self):
        super(ModuleReconstruction, self).__init__()   
    def forward(self,H_estimate,v_known):
        
        x_estimate = F.conv2d(F.pad(H_estimate.transpose(0,1),(v_known.shape[-1]-1,v_known.shape[-1]-1)),v_known.flip(-1).unsqueeze(1),groups = v_known.shape[0]).squeeze().transpose(0,1)#.transpose(0,1)#.squeeze().transpose(0,1)
        if len(x_estimate.shape ) == 2:
            x_estimate = x_estimate.transpose(0,1)
        
        return x_estimate
     
class ModuleConv1d(nn.Module): 
    
    """
    1D convolution module.
    
    Performs grouped 1D convolutions for efficient computation.
    """

    def __init__(self):
        super(ModuleConv1d, self).__init__()   
    def forward(self,H,v_):
        
        B,M,K,P = H.shape
        Hv = F.conv1d(H.reshape(B*M*K,-1).unsqueeze(0),v_.reshape(B*M*K,-1).flip(1).unsqueeze(1),padding = v_.shape[-1]-1,groups = B*M*K).squeeze()[:,:P]
        return Hv