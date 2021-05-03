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

truevalues = np.array([0.005,0.02])

def rmse(targets, predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())

def rmse_norm(targets, predictions):
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    norm = rmse / targets
    return(norm)


### WHOLE HISTORY STUFF ####

estimates_optim = np.load('WholeHistoryEstimates2.npy')
estimates_random = np.load('EstimatesWholeRandom2.npy')
estimates_const = np.load('EstimatesWholeConst20_2.npy')
estimates_nf = np.load('EstimatesWholeNoFreq2.npy')

optimal_freqs = np.load('WholeHistoryOptfrqs2.npy')
mutinfs= np.load('WholeHistoryMutInfs2.npy')
#optimal_const = np.load('OptFrequenciesWholeConst20_1.npy')

#entropies_optim = np.load('EntropiesWholeOptimTrueValuesInOptimUpdatedW0est.npy')
entropies_optim = np.load('WholeHistoryEntropies2.npy')
entropies_random = np.load('EntropiesWholeRandom2.npy')
entropies_const = np.load('EntropiesWholeConst20_2.npy')
entropies_nf = np.load('EntropiesWholeNoFreq2.npy')

mse_optim = []
mse_random = []
mse_const = []
mse_nf = []

mse_optim_a = []
mse_random_a = []
mse_const_a = []
mse_nf_a = []

mse_optim_t = []
mse_random_t = []
mse_const_t = []
mse_nf_t = []

for i in range(len(estimates_optim)):
    mse_optim.append(rmse(truevalues, estimates_optim[i]))
    mse_random.append(rmse(truevalues, estimates_random[i]))
    mse_const.append(rmse(truevalues, estimates_const[i]))
    mse_nf.append(rmse(truevalues,estimates_nf[i]))
    mse_optim_a.append(rmse_norm(truevalues[0], estimates_optim[i][0]))
    mse_random_a.append(rmse_norm(truevalues[0], estimates_random[i][0]))
    mse_const_a.append(rmse_norm(truevalues[0], estimates_const[i][0]))
    mse_nf_a.append(rmse_norm(truevalues[0],estimates_nf[i][0]))
    mse_optim_t.append(rmse_norm(truevalues[1], estimates_optim[i][1]))
    mse_random_t.append(rmse_norm(truevalues[1], estimates_random[i][1]))
    mse_const_t.append(rmse_norm(truevalues[1], estimates_const[i][1]))
    mse_nf_t.append(rmse_norm(truevalues[1],estimates_nf[i][1]))
'''
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
'''
plt.figure()
plt.title('A inference on whole spike history')
plt.xlabel('Trial')
plt.ylabel('Normalised RMSE')
x = np.linspace(1,13,13)
plt.plot(x,mse_optim_a,'rx-',label='Optimised Frequency')
plt.plot(x,mse_random_a,'bx-',label='Randomised Frequency')
plt.plot(x,mse_const_a,'gx-',label='Constant 20Hz')
plt.plot(x,mse_nf_a,'kx-',label='Baseline firing')
plt.yscale('log')
plt.legend()
plt.show()

plt.figure()
plt.title('Tau inference on whole spike history')
plt.xlabel('Trial')
plt.ylabel('Normalised RMSE')
x = np.linspace(1,13,13)
plt.plot(x,mse_optim_t,'rx-',label='Optimised Frequency')
plt.plot(x,mse_random_t,'bx-',label='Randomised Frequency')
plt.plot(x,mse_const_t,'gx-',label='Constant 20Hz')
plt.plot(x,mse_nf_t,'kx-',label='Baseline firing')
plt.yscale('log')
plt.legend()
plt.show()

plt.figure()
plt.title('Entropies on whole spike history')
plt.xlabel('Trial')
plt.ylabel('Entropy')
x = np.linspace(1,13,13)
plt.plot(x,entropies_optim,'rx-',label='Optimised Frequency')
plt.plot(x,entropies_random,'bx-',label='Randomised Frequency')
plt.plot(x,entropies_const,'gx-',label='Constant 20Hz')
plt.plot(x,entropies_nf,'kx-',label='Baseline firing')
plt.legend()
plt.show()

plt.figure()
plt.title('Mutual information')
plt.xlabel('Trial')
plt.ylabel('MC approximated MU')
x = np.linspace(1,12,12)
plt.plot(x,mutinfs[:,0],'rx-',label='20Hz')
plt.plot(x,mutinfs[:,1],'bx-',label='50Hz')
plt.plot(x,mutinfs[:,3],'gx-',label='200Hz')
plt.plot(x,mutinfs[:,2],'kx-',label='100Hz')

plt.legend()
# init history!

'''

estimates_init_optim = np.load('EstimatesInitOptimUpdatedW0est.npy')
estimates_init_random = np.load('EstimatesInitRandom1.npy')
estimates_init_const = np.load('EstimatesInitConst20_1.npy')
estimates_init_nf = np.load('EstimatesInitNoFreq1.npy')

optimal_init_freqs = np.load('OptFrequenciesInitOptimUpdatedW0est.npy')
#optimal_const = np.load('OptFrequenciesWholeConst20_1.npy')

entropies_init_optim = np.load('EntropiesInitOptimUpdatedW0est.npy')
entropies_init_random = np.load('EntropiesInitRandom1.npy')
entropies_init_const = np.load('EntropiesInitConst20_1.npy')
entropies_init_nf = np.load('EntropiesInitNoFreq1.npy')

mse_init_optim = []
mse_init_random = []
mse_init_const = []
mse_init_nf = []

mse_init_optim_a = []
mse_init_random_a = []
mse_init_const_a = []
mse_init_nf_a = []

mse_init_optim_t = []
mse_init_random_t = []
mse_init_const_t = []
mse_init_nf_t = []

for i in range(len(estimates_init_optim)):
    mse_init_optim.append(rmse(truevalues, estimates_init_optim[i]))
    mse_init_random.append(rmse(truevalues, estimates_init_random[i]))
    mse_init_const.append(rmse(truevalues, estimates_init_const[i]))
    mse_init_nf.append(rmse(truevalues,estimates_init_nf[i]))
    mse_init_optim_a.append(rmse(truevalues[0], estimates_init_optim[i][0]))
    mse_init_random_a.append(rmse(truevalues[0], estimates_init_random[i][0]))
    mse_init_const_a.append(rmse(truevalues[0], estimates_init_const[i]))
    mse_init_nf_a.append(rmse(truevalues[0],estimates_init_nf[i][0]))
    mse_init_optim_t.append(rmse(truevalues[1], estimates_init_optim[i][1]))
    mse_init_random_t.append(rmse(truevalues[1], estimates_init_random[i][1]))
    mse_init_const_t.append(rmse(truevalues[1], estimates_init_const[i][1]))
    mse_init_nf_t.append(rmse(truevalues[1],estimates_init_nf[i][1]))

plt.figure()
plt.title('Inference with initial data')
plt.xlabel('Trial')
plt.ylabel('RMSE')
x = np.linspace(1,13,13)
plt.plot(x,mse_init_optim,'rx-',label='Optimised Frequency')
plt.plot(x,mse_init_random,'bx-',label='Randomised Frequency')
plt.plot(x,mse_init_const,'gx-',label='Constant 20Hz')
plt.plot(x,mse_init_nf,'kx-',label='Baseline frequency')
plt.legend()
plt.show()

plt.figure()
plt.title('A inference with initial data')
plt.xlabel('Trial')
plt.ylabel('RMSE')
x = np.linspace(1,13,13)
plt.plot(x,mse_init_optim_a,'rx-',label='Optimised Frequency')
plt.plot(x,mse_init_random_a,'bx-',label='Randomised Frequency')
plt.plot(x,mse_init_const_a,'gx-',label='Constant 20Hz')
plt.plot(x,mse_init_nf_a,'kx-',label='Baseline frequency')
plt.legend()
plt.show()

plt.figure()
plt.title('Tau inference with initial data')
plt.xlabel('Trial')
plt.ylabel('RMSE')
x = np.linspace(1,13,13)
plt.plot(x,mse_init_optim_t,'rx-',label='Optimised Frequency')
plt.plot(x,mse_init_random_t,'bx-',label='Randomised Frequency')
plt.plot(x,mse_init_const_t,'gx-',label='Constant 20Hz')
plt.plot(x,mse_init_nf_t,'kx-',label='Baseline frequency')
plt.legend()
plt.show()

plt.figure()
plt.title('Entropies with initial data')
plt.xlabel('Trial')
plt.ylabel('Entropy')
x = np.linspace(1,13,13)
plt.plot(x,entropies_init_optim,'rx-',label='Optimised Frequency')
plt.plot(x,entropies_init_random,'bx-',label='Randomised Frequency')
plt.plot(x,entropies_init_const,'gx-',label='Constant 20Hz')
plt.plot(x,entropies_init_nf,'kx-',label='Baseline frequency')
plt.legend()
plt.show()
'''  

