# rir2wir

Paper: From Room Impulse Responses to Wall Impulse Responses (to be published soon)

Authors: Stéphane Dilungana, Antoine Deleforge, Cédric Foy, Sylvain Faisan

This repository contains the Python implementation of the alternating optimization approach introduced in the upcoming paper From Room Impulse Responses to Wall Impulse Responses. The code allows reproduction of the core methodology and experiments presented in the article. It is fully implemented in PyTorch to enable efficient parallelization on GPUs.

Please note that detailed information about the data used, required dependencies,and environment setup, will be provided at the time of publication. 

The main script is main.py, which launches the full experiment. It relies on the following components:

- alternating_optimization.py: implementation of the alternating optimization procedure, including:
  
    * echoes delay optimization using Adam gradient descent, 
    * wall impulse response (WIR) estimation as a linear system inversion using the Conjugate Gradient method, 
    * projection onto non-linear constraints.
      
- functions.py: utility functions used throughout the pipeline.
  
Affiliations: Inria Nancy Grand Est (MULTISPEECH), Cerema (UMRAE), ICube (IMAGeS)
