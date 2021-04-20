#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 19:29:31 2021

@author: emilam
"""
import numpy as np              
import matplotlib.pyplot as plt 
from scipy.interpolate import make_interp_spline, BSpline
import math
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
from scipy.stats import norm

plt.style.use('seaborn-darkgrid')

truevalues = [0.005,0.02]  

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

### WHOLE HISTORY STUFF ####

estimates_optim = np.load('EstimatesWholeOptim1.npy')
estimates_random = np.load('EstimatesWholeRandom1.npy')
estimates_const = np.load('EstimatesWholeConst20_1.npy')
estimates_nf = np.load('EstimatesWholeNoFreq1.npy')

optimal_freqs = np.load('OptFrequenciesWholeOptim1.npy')
#optimal_const = np.load('OptFrequenciesWholeConst20_1.npy')

entropies_optim = np.load('EntropiesWholeOptim1.npy')
entropies_random = np.load('EntropiesWholeRandom1.npy')
entropies_const = np.load('EntropiesWholeConst20_1.npy')
entropies_nf = np.load('EntropiesWholeNoFreq1.npy')

mse_optim = []
mse_random = []
mse_const = []
mse_nf = []

for i in range(len(estimates_optim)):
    mse_optim.append(rmse(truevalues, estimates_optim[i]))
    mse_random.append(rmse(truevalues, estimates_random[i]))
    mse_const.append(rmse(truevalues, estimates_const[i]))
    mse_nf.append(rmse(truevalues,estimates_nf[i]))

plt.figure()
plt.title('Inference on whole spike history')
plt.xlabel('Trial')
plt.ylabel('RMSE')
x = np.linspace(1,13,13)
plt.plot(x,mse_optim,'rx-',label='Optimised Frequency')
plt.plot(x,mse_random,'bx-',label='Randomised Frequency')
plt.plot(x,mse_const,'gx-',label='Constant 20Hz')
plt.plot(x,mse_nf,'kx-',label='Baseline frequency')
plt.legend()
plt.show()

plt.figure()
plt.title('Entropies on whole spike history')
plt.xlabel('Trial')
plt.ylabel('RMSE')
x = np.linspace(1,13,13)
plt.plot(x,entropies_optim,'rx-',label='Optimised Frequency')
plt.plot(x,entropies_random,'bx-',label='Randomised Frequency')
plt.plot(x,entropies_const,'gx-',label='Constant 20Hz')
plt.plot(x,entropies_nf,'kx-',label='Baseline frequency')
plt.legend()
plt.show()
    

