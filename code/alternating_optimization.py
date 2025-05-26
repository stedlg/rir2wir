#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on April 2024

@author: stedlg

---

Room Acoustics Parameter Estimation Module

This module implements optimization algorithms for joint estimation of time delays
and wall impulse responses in room acoustic environments from noisy audio measurements.
It provides functions for alternating optimization, delay estimation, conjugate gradient
method for wall impulse response estimation, and constraint projection to ensure
physical realizability of parameters.

Main Functions:
- alternating_optimization(): Alternates between delay and wall response estimation
- delay_optimization(): Estimates sound propagation delays using gradient descent
- torch_cgm(): Conjugate Gradient Method for wall impulse response estimation
- constraint_projection(): Ensures physical realizability of estimated parameters

Typical use case: Analyze room impulse response recordings to determine acoustic
properties of walls and precise timing of sound reflections.
"""

#%% Modules 

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

#%% Functions

def alternating_optimization(x_blurred,H_noise_blurred,measured_toa,delta,module_reconstruction,module_correction,module_cgm,batch_size_room,delay_max,x,H_noise,Q_max,mask,v,q,L,n_reflections_walls,module_conv1d,device,global_params,setting_params,optim_params):
    
    """
    Alternating optimization algorithm for joint estimation of delays and wall impulse responses.
    
    The algorithm alternates between:
    1. Delay optimization using gradient descent
    2. Wall impulse response estimation using Conjugate Gradient Method
    3. Projection of estimated responses onto physical constraints
    
    Parameters
    ----------
    x_blurred: tensor, Blurred observations (room impulse responses)
    H_noise_blurred: tensor, Noisy and blurred forward operator
    measured_toa: tensor, Measured time-of-arrival values
    delta: tensor, Ground truth delay values
    module_reconstruction: callable, Reconstruction module
    module_correction: callable, Delay correction module
    module_cgm: callable, Conjugate Gradient Method module
    batch_size_room: int, Number of rooms in batch
    delay_max: float, Maximum possible delay value
    x: tensor, Ground truth observations
    H_noise: tensor, Noisy forward operator
    Q_max: int, Maximum reflection order
    mask: tensor, Mask indicating valid measurements
    v: tensor, Ground truth wall impulse responses
    q: tensor, Reflection orders of image sources
    L: int, Length of wall impulse responses
    n_reflections_walls: tensor, Wall reflection information
    module_conv1d: callable, 1D convolution module for constraint projection
    device: str, Computation device ('cpu' or 'cuda')
    global_params: dict, Global parameters 
    setting_params: dict, Setting parameters 
    optim_params: dict, Optimization parameters 
    
    Returns
    -------
    list: Contains all optimization metrics and results including:
        - V_walls_estimate_: Estimated wall impulse responses over iterations
        - Various residual and error metrics (residual_0_, mae_target_0_, etc.)
        - Delay optimization metrics (losses_, losses_delta_mae_, etc.)
        - Oracle comparison metrics (loss_0_)
    """
    
    # Parameters 
    
    fs,simulator_offset,dtype = global_params['fs'],global_params['simulator_offset'],global_params['dtype']
    mode,blur_std,mode_alternance,sigma_geo = setting_params['mode'],setting_params['blur_std'],setting_params['mode_alternance'],setting_params['sigma_geo']
    n_iter_cgm,max_iter_alternate,max_iter,lr,patience,previous,threshold, MAE_loss, MSE_loss = optim_params['n_iter_cgm'],optim_params['max_iter_alternate'],optim_params['max_iter'],optim_params['lr'],optim_params['patience'],optim_params['previous'],optim_params['threshold'],optim_params['MAE_loss'],optim_params['MSE_loss']
    
    # Metrics for delay optimization : see definition of delay_optimization() for description
    
    losses_ = [] # loss for the optimization of delays over iterations of the alternating algorithm
    loss_0_ = [] # oracle, value of the loss for perfect estimation of delays over iterations of the alternating algorithm
    losses_delta_mae_ = [] # mae for the estimated delays over iterations of the alternating algorithm
    losses_delta_ce_ = [] # ce for the estimated delays over iterations of the alternating algorithm 

    # Metrics for WIRs optimization : see definition of torch_cgm() for description
    
    residual_0_,mae_target_0_,ce_target_0_,mae_0_,ce_0_,mse_v_t_target_0_,mse_v_t_0_,mae_target_f_0_,ce_target_f_0_,mae_f_0_,ce_f_0_ = [],[],[],[],[],[],[],[],[],[],[]
    residual_init_,mae_target_init_,ce_target_init_,mae_init_,ce_init_,mse_v_t_target_init_,mse_v_t_init_,mae_target_f_init_,ce_target_f_init_,mae_f_init_,ce_f_init_ = [],[],[],[],[],[],[],[],[],[],[]
    residual_end_,mae_target_end_,ce_target_end_,mae_end_,ce_end_,mse_v_t_target_end_,mse_v_t_end_,mae_target_f_end_,ce_target_f_end_,mae_f_end_,ce_f_end_ = [],[],[],[],[],[],[],[],[],[],[]
    
    V_walls_estimate_ = [] # store the estimated WIRs over iterations of the alternating algorithm
 
    if not(mode == "oracle"):
        v_known = torch.zeros(v.shape,device=device) 
        v_known[...,0] = 1 
        v_known[(mask.sum(axis = 1)==0),:] = 0 # présent dans aucune des mesures pour une salle
    else :        
        v_known = v
    
    v_walls_estimate = v_known[q==1].reshape(v_known.shape[0],-1,v_known.shape[-1])  
    V_walls_estimate_.append(v_walls_estimate[...,:L].cpu().numpy())
    
    # Initial WIRs : reflective walls or ground truth (oracle)
    
    for iter_alternate in range(max_iter_alternate):
        
        #### Projection on constraints (after first iteration)
        
        S = n_reflections_walls.shape[1]
        n_id_walls = torch.argmax((n_reflections_walls.transpose(-2,-1)[q==1,...].reshape(batch_size_room,-1,S).transpose(-2,-1))[0], axis=0)
        
        if iter_alternate > 0 :
            
            # No need to alternate in these cases
            
            if (mode == "oracle") | (blur_std == 0) :
                break
            
            if not(mode_alternance == 'standard'):
                
                v_known = constraint_projection(Q_max,v_walls_estimate,n_reflections_walls,n_id_walls,v,mask,device,module_conv1d,batch_size_room)  
                v_known = v_known.to(device)
                
            else : #### Standard alternating optimization   
                
                v_known = F.pad(v_walls_estimate,(0,v.shape[-1]-v_walls_estimate.shape[-1]), "constant", 0).to(device)
            
        #### Optimization of delays
        
        [losses,losses_delta_mae,losses_delta_ce,delta_estimate,x_estimate,H_estimate] = delay_optimization(delay_max,max_iter,lr,x_blurred,H_noise_blurred,measured_toa,delta,mask,device,patience,previous,threshold,sigma_geo,module_reconstruction,module_correction,simulator_offset,fs,v_known,MSE_loss,MAE_loss)

        losses_.append(losses)
        losses_delta_mae_.append(losses_delta_mae)
        losses_delta_ce_.append(losses_delta_ce)
        
        with torch.no_grad():
            
            # Oracle case (perfect estimation of delays)
            
            x,H_noise = x.to(device),H_noise.to(device)

            H_estimate_not_blurred = module_correction(delta_estimate,H_noise,measured_toa,simulator_offset,fs,delay_max,device)
            
            H_real = module_correction((-1)*delta,H_noise,measured_toa,simulator_offset,fs,delta.abs().max().item(),device)
            H_real_blurred = module_correction((-1)*delta,H_noise_blurred,measured_toa,simulator_offset,fs,delta.abs().max().item(),device)
 
            x_0 = module_reconstruction(H_real_blurred,v_known)
            x_0 = x_0[...,:x.shape[-1]]
            
            loss_0 = []
            loss_0.append(MSE_loss(x_blurred,x_0).item())
            loss_0_.append(loss_0) 
            
            #### Optimization of WIRs with the Conjugate Gradient Method 
   
            _,residual_0,mae_target_0,ce_target_0,mae_0,ce_0,mse_v_t_target_0,mse_v_t_0,mae_target_f_0,ce_target_f_0,mae_f_0,ce_f_0 = module_cgm(n_iter_cgm,x,H_real,v,q,L,dtype,lambda_tikhonov=0) 
            _,residual_init,mae_target_init,ce_target_init,mae_init,ce_init,mse_v_t_target_init,mse_v_t_init,mae_target_f_init,ce_target_f_init,mae_f_init,ce_f_init = module_cgm(n_iter_cgm,x,H_noise,v,q,L,dtype,lambda_tikhonov=0)
            v_cgm_end,residual_end,mae_target_end,ce_target_end,mae_end,ce_end,mse_v_t_target_end,mse_v_t_end,mae_target_f_end,ce_target_f_end,mae_f_end,ce_f_end= module_cgm(n_iter_cgm,x,H_estimate_not_blurred,v,q,L,dtype,lambda_tikhonov=0)
            
            if not(mode_alternance == 'standard'):
                v_walls_estimate = v_cgm_end[q==1].reshape(v_known.shape[0],-1,L)
                V_walls_estimate_.append(v_walls_estimate.cpu().numpy())
                
            else : 
                
                v_walls_estimate = v_cgm_end
                V_walls_estimate_.append(v_walls_estimate[q==1].reshape(v_known.shape[0],-1,L).cpu().numpy())
            
            #### Calculation of metrics and storage 
            
            residual_0_.append(residual_0.cpu().numpy().mean(axis=0).tolist())
            residual_init_.append(np.array(residual_init.cpu().numpy()).mean(axis=0).tolist())
            residual_end_.append(np.array(residual_end.cpu().numpy()).mean(axis=0).tolist())
            

            mae_target_0_.append(mae_target_0)
            mae_target_init_.append(mae_target_init)
            mae_target_end_.append(mae_target_end)
            ce_target_0_.append(ce_target_0)
            ce_target_init_.append(ce_target_init)
            ce_target_end_.append(ce_target_end)
            mae_0_.append(mae_0)
            mae_init_.append(mae_init)
            mae_end_.append(mae_end)
            ce_0_.append(ce_0)
            ce_init_.append(ce_init)
            ce_end_.append(ce_end)  
            mse_v_t_target_0_.append(mse_v_t_target_0)
            mse_v_t_target_init_.append(mse_v_t_target_init)
            mse_v_t_target_end_.append(mse_v_t_target_end)
            mse_v_t_0_.append(mse_v_t_0)
            mse_v_t_init_.append(mse_v_t_init)
            mse_v_t_end_.append(mse_v_t_end)
            
            mae_target_f_0_.append(mae_target_f_0.cpu().numpy())
            mae_target_f_init_.append(mae_target_f_init.cpu().numpy())
            mae_target_f_end_.append(mae_target_f_end.cpu().numpy())
            ce_target_f_0_.append(ce_target_f_0.cpu().numpy())
            ce_target_f_init_.append(ce_target_f_init.cpu().numpy())
            ce_target_f_end_.append(ce_target_f_end.cpu().numpy())
            mae_f_0_.append(mae_f_0.cpu().numpy())
            mae_f_init_.append(mae_f_init.cpu().numpy())
            mae_f_end_.append(mae_f_end.cpu().numpy())
            ce_f_0_.append(ce_f_0.cpu().numpy())
            ce_f_init_.append(ce_f_init.cpu().numpy())
            ce_f_end_.append(ce_f_end.cpu().numpy()) 
    
    return [V_walls_estimate_,residual_0_,residual_init_,residual_end_,mae_target_0_
    ,mae_target_init_,mae_target_end_,ce_target_0_,ce_target_init_,ce_target_end_
    ,mae_0_,mae_init_,mae_end_,ce_0_,ce_init_,ce_end_,mse_v_t_target_0_,mse_v_t_target_init_
    ,mse_v_t_target_end_,mse_v_t_0_,mse_v_t_init_,mse_v_t_end_,mae_target_f_0_
    ,mae_target_f_init_,mae_target_f_end_,ce_target_f_0_,ce_target_f_init_
    ,ce_target_f_end_,mae_f_0_,mae_f_init_,mae_f_end_,ce_f_0_,ce_f_init_,ce_f_end_,losses_,losses_delta_mae_,losses_delta_ce_,loss_0_]

def delay_optimization(delay_max,max_iter,lr,x_blurred,H_noise_blurred,measured_toa,delta,mask,device,patience,previous,threshold,sigma_geo,module_reconstruction,module_correction,simulator_offset,fs,v_known,MSE_loss,MAE_loss):
    
    """
    Optimizes delays using gradient descent.
    
    Parameters
    ----------
    delay_max: float, Maximum possible delay value
    max_iter: int, Maximum number of iterations
    lr: float, Learning rate
    x_blurred: tensor, Blurred observations
    H_noise_blurred: tensor, Noisy and blurred forward operator
    measured_toa: tensor, Measured time-of-arrival values
    delta: tensor, Ground truth delays
    mask: tensor, Mask indicating valid measurements
    device: str, Computation device
    patience: int, Patience for early stopping
    previous: int, Number of previous losses to consider
    threshold: float, Relative threshold for early stopping
    sigma_geo: float, Geometry noise parameter
    module_reconstruction: callable, Reconstruction module
    module_correction: callable, Delay correction module
    simulator_offset: float, Simulator time offset
    fs: float, Sampling frequency
    v_known: tensor, Known wall impulse responses
    MSE_loss: callable, MSE loss function
    MAE_loss: callable, MAE loss function
    
    Returns
    -------
    list: Contains:
        - Loss values over iterations
        - Delay MAE and CE metrics
        - Estimated delays and reconstructed signals
    """
    
    
    losses = [] 
    
    losses_delta_mae = [] 
    losses_delta_ce = []
    
    U = torch.zeros(H_noise_blurred.shape[:-1],requires_grad = True, device=device)
    optimizer = optim.Adam([U],lr = lr)
    
    for iteration in range(max_iter-1):
    
        # Stopping criterion
        
        if len(losses)>patience: 
            if (losses[-1]/losses[-previous]) > (1-threshold) : 
                print("Stopping criterion for delay optimization :",iteration,"iterations")
                break 
            
        if (sigma_geo == 0) and (iteration >2) :
            break
  
        optimizer.zero_grad() 
              
        delta_estimate = delay_max*torch.tanh(U).nan_to_num()
        
        H_estimate = module_correction(delta_estimate,H_noise_blurred,measured_toa,simulator_offset,fs,delay_max,device)
        x_estimate = module_reconstruction(H_estimate,v_known)
        x_estimate = x_estimate[...,:x_blurred.shape[-1]]
        
        # loss 
        
        loss = MSE_loss(x_blurred,x_estimate)
        loss_delta_mae = MAE_loss(delta_estimate[mask==1],(-1)*delta[mask==1]).detach().item()
        loss_delta_ce = ((((((delta_estimate - ((-1)*delta))[mask==1])*fs).abs())<1).sum() / mask.sum()).detach().item()
        
        losses.append(loss.item())
        losses_delta_mae.append(loss_delta_mae)
        losses_delta_ce.append(loss_delta_ce)
        
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters U
        
    _ = [losses.append(losses[-1]) for fill in range(max_iter-len(losses))]
    _ = [losses_delta_mae.append(losses_delta_mae[-1]) for fill in range(max_iter-len(losses_delta_mae))]
    _ = [losses_delta_ce.append(losses_delta_ce[-1]) for fill in range(max_iter-len(losses_delta_ce))]

    return [losses,losses_delta_mae,losses_delta_ce,delta_estimate,x_estimate,H_estimate]

def torch_cgm(n_iter,x,H,L=None,v_init=None,v_truth=None,q = None,device = "cpu",dtype=torch.float64,frequency_error = True,lambda_tikhonov = 0,all_orders = False):
    
    """
    Conjugate Gradient Method (CGM) implementation in Pytorch.
    
    The Pytorch implementation allows to backpropagate gradients for optimization 
    and to split computation on gpus with convinience.
    
    Parameters
    ----------
    n_iter: int, Number of iterations in the CGM
    x: tensor, Observations (here room impulse responses (dim : (M,dT)))
    H: tensor, Forward operator (here geometry and device responses (dim : M,K,T))
    L: int, optional, Length of the model parameters (here the wall impulse repsonses)
    v_init: tensor, optional, Initial values model parameters (dim : (K,L))
    v_truth : tensor, optional, Ground truth model parameters (dim : (K,L1))
    q : tensor, optional, Reflection order of the image sources (dim : K)
    device: string, Device which runs the computation
    dtype: dtype, Float precision
    frequency_error: boolean, Are metrics over frequencies needed?
    lambda_tikhonov: float, Parameter which controls the importance of the regularization term
    all_orders: boolean, If the estimation is regularized, should it operate on all the image sources or just the first-order ones? 
     
    Outputs
    ----------
    v_cgm: tensor, Estimates of the model parameters (dim : (K,L)) 
    residual: tensor, Residuals over the iterations 
    mae_target: tensor, Mean absolute error for the estimation of the reflectivity/absorption profiles over iterations for the first-order image sources 
    ce_target: tensor, Proportion of correct estimates for the estimation of the reflectivity/absorption profiles over iterations for the first-order image sources 
    mae: tensor, Mean absolute error for the estimation of the reflectivity/absorption profiles over iterations for all the image sources
    ce: tensor, Proportion of correct estimates for the estimation of the reflectivity/absorption profiles over iterations for all the image sources 
    mse_v_t_target: tensor, Mean squared error for the estimation of the wall impulse responses over iterations for the first-order image sources 
    mse_v_t: tensor, Mean squared error for the estimation of the wall impulse responses over iterations for all the image sources 
    mae_target_f: tensor, Mean absolute error for the estimation of the reflectivity/absorption profiles over frequencies for the first-order image sources 
    ce_target_f: tensor, Proportion of correct estimates for the estimation of the reflectivity/absorption profiles over frequencies for the first-order image sources 
    mae_f: tensor, Mean absolute error for the estimation of the reflectivity/absorption profiles over frequencies for all the image sources
    ce_f: tensor, Proportion of correct estimates for the estimation of the reflectivity/absorption profiles over frequencies for all the image sources 
    """
    
    residual = []
    mae_target = []
    mae = []
    ce_target = []
    ce = []
    mse_v_t_target = []
    mse_v_t = []

    eps = torch.finfo(torch.float64).eps 
    
    B,M,K,P = 1,H.shape[-3],H.shape[-2],H.shape[-1]
    
    if x.shape[-1] > P :
        x = x[...,:P]
    
    if len(H.shape) == 4:
        B = H.shape[-4]
        # size : M,B,K,P
        
        H = H.transpose(0,1)
    else : 
        H = H.unsqueeze(1)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

    if v_init is None:
        
        v = torch.zeros(B,K,L,dtype = dtype,device = device)
    
    else : 
        v = v_init         

    Hv = F.conv2d(F.pad(H,(v.shape[-1]-1,v.shape[-1]-1)),v.flip(-1).unsqueeze(1),groups = B).squeeze()[...,:x.shape[-1]]  
    
    if B == 1 :
        Hv = Hv.unsqueeze(0)
        if M == 1 :
            Hv = Hv.unsqueeze(1)
    if M == 1 :
        Hv = Hv.unsqueeze(1)
    else : 
        Hv = Hv.transpose(0,1)   
    
    # To consider only if tikhonov regulrization is applied (lambda_tikhonov != 0) 
    # Rather  we want to regularize along all the image sources or just the 6 walls 
    
    if all_orders :
        
        mask_tikhonov = torch.ones(B,K,L,dtype = dtype,device = device)
       
    else : 
        
        mask_tikhonov = torch.ones(B,K,L,dtype = dtype,device = device)
        mask_tikhonov[q != 1,:] = 0
        
    r = F.conv2d(F.pad(H[:,:,:,:x.shape[-1]].transpose(0,2),(v.shape[-1]-1,0)),(x-Hv).unsqueeze(1),groups = B).squeeze().flip(-1)[...,-(v.shape[-1]):] - lambda_tikhonov*(torch.mul(v,mask_tikhonov)).transpose(0,1) 

    if B == 1 :
        r = r.unsqueeze(0) 
    else : 
        r = r.transpose(0,1)
        
    p = r
    
    for n in range(n_iter):
      
        Hp = F.conv2d(F.pad(H,(v.shape[-1]-1,v.shape[-1]-1)),p.flip(-1).unsqueeze(1),groups = B).squeeze()[...,:x.shape[-1]] 
        if B == 1 :
            Hp = Hp.unsqueeze(0)
            if M == 1 :
                Hp = Hp.unsqueeze(1)
        if M == 1 :
            Hp = Hp.unsqueeze(1)
        else : 
            Hp = Hp.transpose(0,1)
    
        alpha = (torch.square(torch.norm(r,dim=(-2,-1)))/((torch.square(torch.norm(Hp,dim=(-2,-1)))+lambda_tikhonov*torch.square(torch.norm((torch.mul(p,mask_tikhonov)),dim=(-2,-1))))+eps)).unsqueeze(1).unsqueeze(1)
        v = v + alpha*p
        
        HtHp = F.conv2d(F.pad(H[:,:,:,:x.shape[-1]].transpose(0,2),(v.shape[-1]-1,0)),Hp.unsqueeze(1),groups = B).squeeze().flip(-1)[...,-(v.shape[-1]):]
        
        if B == 1 :
            HtHp = HtHp.unsqueeze(0)
        else : 
            HtHp = HtHp.transpose(0,1)
       
        beta = (torch.square(torch.norm((r - alpha*HtHp - lambda_tikhonov*alpha*(torch.mul(p,mask_tikhonov))),dim=(-2,-1)))/(torch.square(torch.norm(r,dim=(-2,-1)))+eps)).unsqueeze(1).unsqueeze(1) 
        
        r = r - alpha*HtHp - lambda_tikhonov*alpha*(torch.mul(p,mask_tikhonov))
        
        residual.append((torch.abs(r).amax(dim=(-2,-1))))
        
        p = r + beta*p

        if not((v_truth is None) and (q is None)):
            
            if not(v[q==1,:].size(0) == 0) :
                
                omega_estimate_target = torch.square(torch.abs(torch.fft.rfft(v[q==1,:],axis = -1)))
                omega_estimate_target[omega_estimate_target>1]=1
                
                omega_truth_target = torch.square(torch.abs(torch.fft.rfft(v_truth[q==1,:][...,:v.shape[-1]],axis = -1)))
                
                mae_target.append(torch.abs(omega_estimate_target - omega_truth_target).mean())
                ce_target.append((torch.abs(omega_estimate_target - omega_truth_target)<(0.1)).to(torch.float).mean())
                mse_v_t_target.append(torch.square(v[q==1,:] - (v_truth[q==1,:][...,:v.shape[-1]])).mean())
            
            if not(v[q>1,:].size(0) == 0) : 
                
                omega_estimate = torch.square(torch.abs(torch.fft.rfft(v[q>1,:],axis = -1)))
                omega_estimate[omega_estimate>1]=1
                
                omega_truth = torch.square(torch.abs(torch.fft.rfft(v_truth[q>1,:][...,:v.shape[-1]],axis = -1)))
                
                mae.append(torch.abs(omega_estimate - omega_truth).mean())
                ce.append((torch.abs(omega_estimate - omega_truth)<(0.1)).to(torch.float).mean())
                mse_v_t.append(torch.square(v[q>1,:] - (v_truth[q>1,:][...,:v.shape[-1]])).mean())
                
    v_cgm = v
    
    residual = torch.stack(residual,dim = 1)
    
    if not((v_truth is None) and (q is None)) :  
        if frequency_error : 
            
            if not(v[q==1,:].size(0) == 0) : 
               
                omega_estimate_target = torch.square(torch.abs(torch.fft.rfft(v[q==1,:],axis = -1)))
                omega_estimate_target[omega_estimate_target>1]=1 # called practical coefficient in the litterature
                    
                omega_truth_target = torch.square(torch.abs(torch.fft.rfft(v_truth[q==1,:][...,:v.shape[-1]],axis = -1)))
                
                mae_target_f = torch.abs(omega_estimate_target - omega_truth_target).mean(axis = 0)
                ce_target_f = (torch.abs(omega_estimate_target - omega_truth_target)<(0.1)).to(torch.float).mean(axis = 0)
            
            else : 
                mae_target_f = None
                ce_target_f = None
            
            if not(v[q>1,:].size(0) == 0) : 
                omega_estimate = torch.square(torch.abs(torch.fft.rfft(v[q>1,:],axis = -1)))
                omega_estimate[omega_estimate>1]=1 # called practical coefficient the litterature
                
                omega_truth = torch.square(torch.abs(torch.fft.rfft(v_truth[q>1,:][...,:v.shape[-1]],axis = -1)))
                
                mae_f = torch.abs(omega_estimate - omega_truth).mean(axis = 0) 
                ce_f = (torch.abs(omega_estimate - omega_truth)<(0.1)).to(torch.float).mean(axis = 0)

            else : 
                mae_f = None
                ce_f = None
                
            return [v_cgm,residual,mae_target,ce_target,mae,ce,mse_v_t_target,mse_v_t,mae_target_f,ce_target_f,mae_f,ce_f]
        else :
            return [v_cgm,residual,mae_target,ce_target,mae,ce,mse_v_t_target,mse_v_t]
    else :
        
        mae_target = None 
        mae = None 
        ce_target = None
        ce = None
        mse_v_t_target = None
        mse_v_t = None
        
        return [v_cgm,residual,mae_target,ce_target,mae,ce,mse_v_t_target,mse_v_t]
    
def constraint_projection(Q_max,v_walls_estimate,n_reflections_walls,n_id_walls,v,mask,device,module_conv1d,batch_size_room):
    
    """
    Projects estimated wall impulse responses onto physical constraints.
    
    Reconstructs all cumulative wall impulse responses from the 6 estimated ones.
    
    Parameters
    ----------
    Q_max: int, Maximum reflection order
    v_walls_estimate: tensor, Estimated wall impulse responses
    n_reflections_walls: tensor, Wall reflection counts
    n_id_walls: tensor, Wall identifiers
    v: tensor, Ground truth wall impulse responses
    mask: tensor, Measurement mask
    device: str, Computation device
    module_conv1d: callable, 1D convolution module
    batch_size_room: int, Number of rooms in batch
    
    Returns
    -------
    tensor: Reconstructed wall impulse responses satisfying physical constraints
    """
    
    list_v_build = []
    
    for q_build in range(Q_max) :
        
        v_build = torch.zeros(v.shape,device=device) 
        v_build[...,0] = 1 
        v_build[(mask.sum(axis = 1)==0),:] = 0
        
        list_v_build.append(v_build)
    
    for n_room in range(v.shape[0]):
        for n_si in range(v.shape[1]):
            c = 0 
            
            for n_wall_coeff in range(n_reflections_walls.shape[1]): 
                
                n_hit_wall_coeff = n_reflections_walls[n_room,n_wall_coeff,n_si].int().item()
 
                if n_hit_wall_coeff > 0 :
                    n_id_wall = (n_id_walls == n_wall_coeff).nonzero(as_tuple=False).item()
                   
                    for n_hit in range(n_hit_wall_coeff):
                        
                        if c >= len(list_v_build):
                            continue
                        list_v_build[c][n_room,n_si,:v_walls_estimate.shape[-1]] = v_walls_estimate[n_room,n_id_wall,:]
                        c+=1
    
    # Cumulative convolutions
    
    recomputed_v = list_v_build[0]
    
    for v_temp in list_v_build[1:] :
        recomputed_v = module_conv1d(recomputed_v.unsqueeze(1),v_temp).reshape(batch_size_room,-1,recomputed_v.shape[-1])
                    
    return recomputed_v    