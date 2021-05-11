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
'''

### WHOLE HISTORY STUFF ####
estimates_optim = np.load('EstimatesWholeOptimUpdatedW0est.npy')
#estimates_optim = np.load('WholeHistoryEstimates2.npy')
estimates_random = np.load('EstimatesWholeRandom1.npy')
estimates_const = np.load('EstimatesWholeConst20_1.npy')
estimates_nf = np.load('EstimatesWholeNoFreq1.npy')

optimal_freqs = np.load('WholeHistoryOptfrqs2.npy')
mutinfs= np.load('WholeHistoryMutInfs2.npy')
#optimal_const = np.load('OptFrequenciesWholeConst20_1.npy')

entropies_optim = np.load('EntropiesWholeOptimTrueValuesInOptimUpdatedW0est.npy')
#entropies_optim = np.load('WholeHistoryEntropies2.npy')
entropies_random = np.load('EntropiesWholeRandom1.npy')
entropies_const = np.load('EntropiesWholeConst20_1.npy')
entropies_nf = np.load('EntropiesWholeNoFreq1.npy')

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

plt.figure()
plt.title('Inference on whole spike history')
plt.xlabel('Trial')
plt.ylabel('RMSE')
x = np.linspace(1,13,13)
plt.plot(x,mse_optim,'rx-',label='Optimised Frequency')
plt.plot(x,mse_random,'bx-',label='Randomised Frequency')
plt.plot(x,mse_const,'gx-',label='Constant 20Hz')
plt.plot(x,mse_nf,'kx-',label='Baseline frequency')
plt.yscale('log')
plt.legend()
plt.show()

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
'''
'''
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
'''
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


# LARGE SCALE STUFF!

def lr1(s2,s1,Ap,delta,taup):
    return s2*s1*Ap*np.exp(-delta/taup)

def lr2(s1,s2,Am,delta,taum):
    return -s1*s2*Am*np.exp(delta/taum)


deltas = np.linspace(0,0.1,10000)
deltas2 = np.linspace(-0.1,0,10000)   
lrs1 = lr1(1,1,0.005,deltas,0.02)
lrs2 = lr2(1,1,0.005,deltas2,0.02) 

lrref = np.concatenate((lrs2,lrs1))



# optimal regime


Westimates1 = np.load('WHestmimates_1.npy')
Westimates2 = np.load('WHestmimates_2.npy')
Westimates3 = np.load('WHestmimates_3.npy')
Westimates4= np.load('WHestmimates_4.npy')
#estimates5= np.load('WHestmimates_5.npy')
Westimates6= np.load('WHestmimates_6.npy')
#estimates7= np.load('WHestmimates_7.npy')
#estimates8= np.load('WHestmimates_8.npy')
#estimates9= np.load('WHestmimates_9.npy')
#estimates10= np.load('WHestmimates_10.npy')
Westimates11= np.load('WHestmimates_11.npy')
Westimates12= np.load('WHestmimates_12.npy')
Westimates13= np.load('WHestmimates_13.npy')
Westimates14= np.load('WHestmimates_14.npy')
Westimates15= np.load('WHestmimates_15.npy')
Westimates16= np.load('WHestmimates_16.npy')
Westimates17= np.load('WHestmimates_17.npy')
Westimates18= np.load('WHestmimates_18.npy')
Westimates19= np.load('WHestmimates_19.npy')
Westimates20= np.load('WHestmimates_20.npy')

Wests= [Westimates1,Westimates2,Westimates3,Westimates4,Westimates6,Westimates11,Westimates12,Westimates13,Westimates14,\
        Westimates15,Westimates16,Westimates17,Westimates18,Westimates19,Westimates20]
Wmse_lr = []
Wmse = []
Wmse_a = []
Wmse_t = []

for i in range(len(Wests)):
    Wmse_lr_temp = []
    Wmse_temp = []
    Wmse_a_temp = []
    Wmse_t_temp = []
    for j in range(len(Wests[i])):
        lrs1_temp = lr1(1,1,Wests[i][j][0],deltas,Wests[i][j][1])
        lrs2_temp = lr2(1,1,Wests[i][j][0],deltas2,Wests[i][j][1]) 
        lr_est = np.concatenate((lrs2_temp,lrs1_temp))
        Wmse_lr_temp.append(rmse(lrref, lr_est))
        Wmse_temp.append(rmse(truevalues,Wests[i][j]))
        Wmse_a_temp.append(rmse_norm(truevalues[0],Wests[i][j][0]))
        Wmse_t_temp.append(rmse_norm(truevalues[1],Wests[i][j][1]))
    Wmse_lr.append(Wmse_lr_temp)
    Wmse.append(Wmse_temp)
    Wmse_a.append(Wmse_a_temp)
    Wmse_t.append(Wmse_t_temp)
        




Wentropies1 = np.load('WHentropies_1.npy')
Wentropies2 = np.load('WHentropies_2.npy')
Wentropies3 = np.load('WHentropies_3.npy')
Wentropies4= np.load('WHentropies_4.npy')
#entropies5= np.load('WHentropies_5.npy')
Wentropies6= np.load('WHentropies_6.npy')
#entropies7= np.load('WHentropies_7.npy')
#entropies8= np.load('WHentropies_8.npy')
#entropies9= np.load('WHentropies_9.npy')
#entropies10= np.load('WHentropies_10.npy')
Wentropies11= np.load('WHentropies_12.npy')
Wentropies12= np.load('WHentropies_12.npy')
Wentropies13= np.load('WHentropies_13.npy')
Wentropies14= np.load('WHentropies_14.npy')
#Wentropies15= np.load('WHentropies_15.npy')
Wentropies16= np.load('WHentropies_16.npy')
Wentropies17= np.load('WHentropies_17.npy')
Wentropies18= np.load('WHentropies_18.npy')
Wentropies19= np.load('WHentropies_19.npy')
#Wentropies20= np.load('WHentropies_20.npy')

WHoptf1 = np.load('WHoptfrqs_1.npy')
WHoptf2 = np.load('WHoptfrqs_2.npy')
WHoptf3 = np.load('WHoptfrqs_3.npy')
WHoptf4 = np.load('WHoptfrqs_4.npy')
#WHoptf5 = np.load('WHoptfrqs_5.npy')
WHoptf6 = np.load('WHoptfrqs_6.npy')
#WHoptf7 = np.load('WHoptfrqs_7.npy')
#WHoptf8 = np.load('WHoptfrqs_8.npy')
#WHoptf9 = np.load('WHoptfrqs_9.npy')
#WHoptf10 = np.load('WHoptfrqs_10.npy')
WHoptf11 = np.load('WHoptfrqs_11.npy')
WHoptf12 = np.load('WHoptfrqs_12.npy')
WHoptf13 = np.load('WHoptfrqs_13.npy')
WHoptf14 = np.load('WHoptfrqs_14.npy')
WHoptf15 = np.load('WHoptfrqs_15.npy')
WHoptf16 = np.load('WHoptfrqs_16.npy')
WHoptf17 = np.load('WHoptfrqs_17.npy')
WHoptf18 = np.load('WHoptfrqs_18.npy')
WHoptf19 = np.load('WHoptfrqs_19.npy')
WHoptf20 = np.load('WHoptfrqs_20.npy')

# constant 20hz 
Cestimates1 = np.load('ConstEstimates_1.npy')
Cestimates2 = np.load('ConstEstimates_2.npy')
Cestimates3 = np.load('ConstEstimates_3.npy')
Cestimates4= np.load('ConstEstimates_4.npy')
#Cestimates5= np.load('ConstEstimates_5.npy')
Cestimates6= np.load('ConstEstimates_6.npy')
Cestimates7= np.load('ConstEstimates_7.npy')
Cestimates8= np.load('ConstEstimates_8.npy')
Cestimates9= np.load('ConstEstimates_9.npy')
#Cestimates10= np.load('ConstEstimates_10.npy')
Cestimates11= np.load('ConstEstimates_11.npy')
Cestimates12= np.load('ConstEstimates_12.npy')
Cestimates13= np.load('ConstEstimates_13.npy')
Cestimates14= np.load('ConstEstimates_14.npy')
#Cestimates15= np.load('ConstEstimates_15.npy')
Cestimates16= np.load('ConstEstimates_16.npy')
Cestimates17= np.load('ConstEstimates_17.npy')
Cestimates18= np.load('ConstEstimates_18.npy')
Cestimates19= np.load('ConstEstimates_19.npy')
#Cestimates20= np.load('ConstEstimates_20.npy')

Cests= [Cestimates1,Cestimates2,Cestimates3,Cestimates4,Cestimates6,Cestimates7,Cestimates8,Cestimates9,Cestimates11,\
        Cestimates12,Cestimates13,Cestimates14,Cestimates16,Cestimates17,Cestimates18,Cestimates19]
Cmse_lr = []
Cmse = []
Cmse_a = []
Cmse_t = []
for i in range(len(Cests)):
    Cmse_lr_temp = []
    Cmse_temp = []
    Cmse_a_temp = []
    Cmse_t_temp = []
    for j in range(len(Cests[i])):
        lrs1_temp = lr1(1,1,Cests[i][j][0],deltas,Cests[i][j][1])
        lrs2_temp = lr2(1,1,Cests[i][j][0],deltas2,Cests[i][j][1]) 
        lr_est = np.concatenate((lrs2_temp,lrs1_temp))
        Cmse_lr_temp.append(rmse(lrref, lr_est))
        Cmse_temp.append(rmse(truevalues,Cests[i][j]))
        Cmse_a_temp.append(rmse_norm(truevalues[0],Cests[i][j][0]))
        Cmse_t_temp.append(rmse_norm(truevalues[1],Cests[i][j][1]))
    Cmse_lr.append(Cmse_lr_temp)
    Cmse.append(Cmse_temp)
    Cmse_a.append(Cmse_a_temp)
    Cmse_t.append(Cmse_t_temp)
    
for i in range(len(Cmse)):
    for j in range(len(Cmse[i])):
        if math.isnan(Cmse[i][j])==True:
            Cmse_temp = np.delete(np.asarray(Cmse)[:,j],i)
            Cmse[i][j] = np.mean(Cmse_temp)
        if math.isnan(Cmse_lr[i][j])==True:
            Cmse_temp = np.delete(np.asarray(Cmse_lr)[:,j],i)
            Cmse_lr[i][j] = np.mean(Cmse_temp)
        if math.isnan(Cmse_a[i][j])==True:
            Cmse_temp = np.delete(np.asarray(Cmse_a)[:,j],i)
            Cmse_a[i][j] = np.mean(Cmse_temp)
        if math.isnan(Cmse_t[i][j])==True:
            Cmse_temp = np.delete(np.asarray(Cmse_t)[:,j],i)
            Cmse_t[i][j] = np.mean(Cmse_temp)
        
           

    
Centropies1 = np.load('ConstEntropies_1.npy')
Centropies2 = np.load('ConstEntropies_2.npy')
Centropies3 = np.load('ConstEntropies_3.npy')
Centropies4= np.load('ConstEntropies_4.npy')
#Centropies5= np.load('ConstEntropies_5.npy')
Centropies6= np.load('ConstEntropies_6.npy')
Centropies7= np.load('ConstEntropies_7.npy')
Centropies8= np.load('ConstEntropies_8.npy')
Centropies9= np.load('ConstEntropies_9.npy')
#Centropies10= np.load('ConstEntropies_10.npy')
Centropies11= np.load('ConstEntropies_12.npy')
Centropies12= np.load('ConstEntropies_12.npy')
Centropies13= np.load('ConstEntropies_13.npy')
Centropies14= np.load('ConstEntropies_14.npy')
#Centropies15= np.load('ConstEntropies_15.npy')
Centropies16= np.load('ConstEntropies_16.npy')
Centropies17= np.load('ConstEntropies_17.npy')
Centropies18= np.load('ConstEntropies_18.npy')
Centropies19= np.load('ConstEntropies_19.npy')
#Centropies20= np.load('ConstEntropies_20.npy')

# random frequency

Restimates1 = np.load('RandEstimates_1.npy')
Restimates2 = np.load('RandEstimates_2.npy')
Restimates3 = np.load('RandEstimates_3.npy')
Restimates4= np.load('RandEstimates_4.npy')
#Restimates5= np.load('RandEstimates_5.npy')
Restimates6= np.load('RandEstimates_6.npy')
Restimates7= np.load('RandEstimates_7.npy')
Restimates8= np.load('RandEstimates_8.npy')
Restimates9= np.load('RandEstimates_9.npy')
#Restimates10= np.load('RandEstimates_10.npy')
Restimates11= np.load('RandEstimates_11.npy')
Restimates12= np.load('RandEstimates_12.npy')
Restimates13= np.load('RandEstimates_13.npy')
Restimates14= np.load('RandEstimates_14.npy')
#Restimates15= np.load('RandEstimates_15.npy')
Restimates16= np.load('RandEstimates_16.npy')
Restimates17= np.load('RandEstimates_17.npy')
Restimates18= np.load('RandEstimates_18.npy')
Restimates19= np.load('RandEstimates_19.npy')
#Restimates20= np.load('RandEstimates_20.npy')

Rests= [Restimates1,Restimates2,Restimates3,Restimates4,Restimates6,Restimates7,Restimates8,Restimates9,Restimates11,\
        Restimates12,Restimates13,Restimates14,Restimates16,Restimates17,Restimates18,Restimates19]
Rmse_lr = []
Rmse = []
Rmse_a = []
Rmse_t = []
for i in range(len(Rests)):
    Rmse_lr_temp = []
    Rmse_temp = []
    Rmse_a_temp = []
    Rmse_t_temp = []
    for j in range(len(Rests[i])):
        lrs1_temp = lr1(1,1,Rests[i][j][0],deltas,Rests[i][j][1])
        lrs2_temp = lr2(1,1,Rests[i][j][0],deltas2,Rests[i][j][1]) 
        lr_est = np.concatenate((lrs2_temp,lrs1_temp))
        Rmse_lr_temp.append(rmse(lrref, lr_est))
        Rmse_temp.append(rmse(truevalues,Rests[i][j]))
        Rmse_a_temp.append(rmse_norm(truevalues[0],Rests[i][j][0]))
        Rmse_t_temp.append(rmse_norm(truevalues[1],Rests[i][j][1]))
    Rmse_lr.append(Rmse_lr_temp)
    Rmse.append(Rmse_temp)
    Rmse_a.append(Rmse_a_temp)
    Rmse_t.append(Rmse_t_temp)
    
    
Rentropies1 = np.load('RandEntropies_1.npy')
Rentropies2 = np.load('RandEntropies_2.npy')
Rentropies3 = np.load('RandEntropies_3.npy')
Rentropies4= np.load('RandEntropies_4.npy')
#Rentropies5= np.load('RandEntropies_5.npy')
Rentropies6= np.load('RandEntropies_6.npy')
Rentropies7= np.load('RandEntropies_7.npy')
Rentropies8= np.load('RandEntropies_8.npy')
Rentropies9= np.load('RandEntropies_9.npy')
#Rentropies10= np.load('RandEntropies_10.npy')
Rentropies11= np.load('RandEntropies_12.npy')
Rentropies12= np.load('RandEntropies_12.npy')
Rentropies13= np.load('RandEntropies_13.npy')
Rentropies14= np.load('RandEntropies_14.npy')
#Rentropies15= np.load('RandEntropies_15.npy')
Rentropies16= np.load('RandEntropies_16.npy')
Rentropies17= np.load('RandEntropies_17.npy')
Rentropies18= np.load('RandEntropies_18.npy')
Rentropies19= np.load('RandEntropies_19.npy')
#Rentropies20= np.load('RandEntropies_20.npy')

# baseline

Bestimates1 = np.load('BaseEstimates_1.npy')
Bestimates2 = np.load('BaseEstimates_2.npy')
Bestimates3 = np.load('BaseEstimates_3.npy')
Bestimates4= np.load('BaseEstimates_4.npy')
#Bestimates5= np.load('BaseEstimates_5.npy')
Bestimates6= np.load('BaseEstimates_6.npy')
Bestimates7= np.load('BaseEstimates_7.npy')
Bestimates8= np.load('BaseEstimates_8.npy')
Bestimates9= np.load('BaseEstimates_9.npy')
#Bestimates10= np.load('BaseEstimates_10.npy')
Bestimates11= np.load('BaseEstimates_11.npy')
Bestimates12= np.load('BaseEstimates_12.npy')
Bestimates13= np.load('BaseEstimates_13.npy')
Bestimates14= np.load('BaseEstimates_14.npy')
#Bestimates15= np.load('BaseEstimates_15.npy')
Bestimates16= np.load('BaseEstimates_16.npy')
Bestimates17= np.load('BaseEstimates_17.npy')
Bestimates18= np.load('BaseEstimates_18.npy')
Bestimates19= np.load('BaseEstimates_19.npy')
#Bestimates20= np.load('BaseEstimates_20.npy')

Bests= [Bestimates1,Bestimates2,Bestimates3,Bestimates4,Bestimates6,Bestimates7,Bestimates8,Bestimates9,Bestimates11,\
        Bestimates12,Bestimates13,Bestimates14,Bestimates16,Bestimates17,Bestimates18,Bestimates19]
Bmse_lr = []
Bmse = []
Bmse_a = []
Bmse_t = []
for i in range(len(Bests)):
    Bmse_lr_temp = []
    Bmse_temp = []
    Bmse_a_temp = []
    Bmse_t_temp = []
    for j in range(len(Bests[i])):
        lrs1_temp = lr1(1,1,Bests[i][j][0],deltas,Bests[i][j][1])
        lrs2_temp = lr2(1,1,Bests[i][j][0],deltas2,Bests[i][j][1]) 
        lr_est = np.concatenate((lrs2_temp,lrs1_temp))
        Bmse_lr_temp.append(rmse(lrref, lr_est))
        Bmse_temp.append(rmse(truevalues,Bests[i][j]))
        Bmse_a_temp.append(rmse_norm(truevalues[0],Bests[i][j][0]))
        Bmse_t_temp.append(rmse_norm(truevalues[1],Bests[i][j][1]))
    Bmse_lr.append(Bmse_lr_temp)
    Bmse.append(Bmse_temp)
    Bmse_a.append(Bmse_a_temp)
    Bmse_t.append(Bmse_t_temp)
    
Bentropies1 = np.load('BaseEntropies_1.npy')
Bentropies2 = np.load('BaseEntropies_2.npy')
Bentropies3 = np.load('BaseEntropies_3.npy')
Bentropies4= np.load('BaseEntropies_4.npy')
#Bentropies5= np.load('BaseEntropies_5.npy')
Bentropies6= np.load('BaseEntropies_6.npy')
Bentropies7= np.load('BaseEntropies_7.npy')
Bentropies8= np.load('BaseEntropies_8.npy')
Bentropies9= np.load('BaseEntropies_9.npy')
#Bentropies10= np.load('BaseEntropies_10.npy')
Bentropies11= np.load('BaseEntropies_12.npy')
Bentropies12= np.load('BaseEntropies_12.npy')
Bentropies13= np.load('BaseEntropies_13.npy')
Bentropies14= np.load('BaseEntropies_14.npy')
#Bentropies15= np.load('BaseEntropies_15.npy')
Bentropies16= np.load('BaseEntropies_16.npy')
Bentropies17= np.load('BaseEntropies_17.npy')
Bentropies18= np.load('BaseEntropies_18.npy')
Bentropies19= np.load('BaseEntropies_19.npy')
#Bentropies20= np.load('BaseEntropies_20.npy')

## Standard deviations

stdW_lrmse = np.sqrt(np.var(Wmse_lr,axis=0))
stdW_mse= np.sqrt(np.var(Wmse,axis=0))

stdC_lrmse = np.sqrt(np.var(Cmse_lr,axis=0))
stdC_mse = np.sqrt(np.var(Cmse,axis=0))

stdR_lrmse = np.sqrt(np.var(Rmse_lr,axis=0))
stdR_mse = np.sqrt(np.var(Rmse,axis=0))

stdB_lrmse = np.sqrt(np.var(Bmse_lr,axis=0))
stdB_mse = np.sqrt(np.var(Bmse,axis=0))

Wmse_lr = np.mean(Wmse_lr,axis=0)
Wmse= np.mean(Wmse,axis=0)

Cmse_lr = np.mean(Cmse_lr,axis=0)
Cmse = np.mean(Cmse,axis=0)

Rmse_lr = np.mean(Rmse_lr,axis=0)
Rmse = np.mean(Rmse,axis=0)

Bmse_lr = np.mean(Bmse_lr,axis=0)
Bmse = np.mean(Bmse,axis=0)

Wmse_a = np.mean(Wmse_a,axis=0)
Wmse_t= np.mean(Wmse_t,axis=0)

Cmse_a = np.mean(Cmse_a,axis=0)
Cmse_t = np.mean(Cmse_t,axis=0)

Rmse_a = np.mean(Rmse_a,axis=0)
Rmse_t = np.mean(Rmse_t,axis=0)

Bmse_a = np.mean(Bmse_a,axis=0)
Bmse_t = np.mean(Bmse_t,axis=0)
'''
plt.figure()
plt.title('RMSE on estimations in 2D space')
plt.xlabel('Trial')
plt.ylabel('RMSE')
x = np.linspace(1,13,13)
plt.plot(x,Wmse,'rx-',label='Optimised Frequency')
plt.plot(x,Rmse,'bx-',label='Randomised Frequency')
plt.plot(x,Cmse,'gx-',label='Constant 20Hz')
plt.plot(x,Bmse,'kx-',label='Baseline firing')
#plt.errorbar(x,Wmse,stdW_mse,color='r')
#plt.errorbar(x,Rmse,stdR_mse,color='b')
#plt.errorbar(x,Cmse,stdC_mse,color='g')
#plt.errorbar(x,Bmse,stdB_mse,color='k')
plt.yscale('log')
plt.legend()
plt.show()

plt.figure()
plt.title('RMSE on learning rule')
plt.xlabel('Trial')
plt.ylabel('RMSE')
x = np.linspace(1,13,13)
plt.plot(x,Wmse_lr,'rx-',label='Optimised Frequency')
plt.plot(x,Rmse_lr,'bx-',label='Randomised Frequency')
plt.plot(x,Cmse_lr,'gx-',label='Constant 20Hz')
plt.plot(x,Bmse_lr,'kx-',label='Baseline firing')
#plt.errorbar(x,Wmse_lr,stdW_lrmse,color='r')
#plt.errorbar(x,Cmse_lr,stdC_lrmse,color='g')
plt.yscale('log')
plt.legend()
plt.show()

plt.figure()
plt.title('RMSE on A estimation')
plt.xlabel('Trial')
plt.ylabel('RMSE')
x = np.linspace(1,13,13)
plt.plot(x,Wmse_a,'rx-',label='Optimised Frequency')
plt.plot(x,Rmse_a,'bx-',label='Randomised Frequency')
plt.plot(x,Cmse_a,'gx-',label='Constant 20Hz')
plt.plot(x,Bmse_a,'kx-',label='Baseline firing')
#plt.errorbar(x,Wmse,stdW_mse,color='r')
#plt.errorbar(x,Rmse,stdR_mse,color='b')
#plt.errorbar(x,Cmse,stdC_mse,color='g')
#plt.errorbar(x,Bmse,stdB_mse,color='k')
plt.yscale('log')
plt.legend()
plt.show()

plt.figure()
plt.title('RMSE on Tau estimation')
plt.xlabel('Trial')
plt.ylabel('RMSE')
x = np.linspace(1,13,13)
plt.plot(x,Wmse_t,'rx-',label='Optimised Frequency')
plt.plot(x,Rmse_t,'bx-',label='Randomised Frequency')
plt.plot(x,Cmse_t,'gx-',label='Constant 20Hz')
plt.plot(x,Bmse_t,'kx-',label='Baseline firing')
#plt.errorbar(x,Wmse_lr,stdW_lrmse,color='r')
#plt.errorbar(x,Cmse_lr,stdC_lrmse,color='g')
plt.yscale('log')
plt.legend()
plt.show()
'''
## histograms of frequencies
'''
opts = [WHoptf1,WHoptf2,WHoptf3,WHoptf4,WHoptf6,WHoptf11,WHoptf12,WHoptf13,WHoptf14,WHoptf15,WHoptf16,WHoptf17,WHoptf18\
        ,WHoptf19,WHoptf20]
for j in range(12):
    chosen_temp = [0,0,0,0]
    freqs = ['20','50','100','200']
    for i in range(len(opts)):
        if opts[i][j] == 20:
            chosen_temp[0] += 1
        if opts[i][j] == 50:
            chosen_temp[1] += 1
        if opts[i][j] == 100:
            chosen_temp[2] += 1
        if opts[i][j] == 200:
            chosen_temp[3] += 1
    plt.figure()
    plt.title('Chosen frequencies trial '+str(j+2))
    #.add_axes([0,0,1,1])
    plt.bar(freqs,chosen_temp)
    plt.show()
        
'''

### Weight trajectories

        
W_weight1 = np.load('WHw_1.npy')
W_weight2 = np.load('WHw_2.npy')
W_weight3 =np.load('WHw_3.npy')
W_weight4 =np.load('WHw_4.npy')
#W_weight5 = np.load('WHw_5.npy')
W_weight6 = np.load('WHw_6.npy')
#W_weight7 = np.load('WHw_7.npy')
#W_weight8 = np.load('WHw_8.npy')
#W_weight9 = np.load('WHw_9.npy')
#W_weight10 =np.load('WHw_10.npy')
W_weight11 = np.load('WHw_11.npy')
W_weight12 = np.load('WHw_12.npy')
W_weight13 = np.load('WHw_13.npy')
W_weight14 = np.load('WHw_14.npy')
W_weight15 = np.load('WHw_15.npy')
W_weight16 = np.load('WHw_16.npy')
W_weight17 = np.load('WHw_17.npy')
W_weight18 = np.load('WHw_18.npy')
W_weight19 = np.load('WHw_19.npy')
W_weight20 = np.load('WHw_20.npy')

Wws = [W_weight1,W_weight2,W_weight3,W_weight4,W_weight6,W_weight11,W_weight12,W_weight13,W_weight14,W_weight15\
       ,W_weight16,W_weight17,W_weight18,W_weight19,W_weight20]

C_weight1 = np.load('ConstW_1.npy')
C_weight2 = np.load('ConstW_2.npy')
C_weight3 =np.load('ConstW_3.npy')
C_weight4 =np.load('ConstW_4.npy')
#C_weight5 = np.load('ConstW_5.npy')
C_weight6 = np.load('ConstW_6.npy')
C_weight7 = np.load('ConstW_7.npy')
C_weight8 = np.load('ConstW_8.npy')
C_weight9 = np.load('ConstW_9.npy')
#C_weight10 =np.load('ConstW_10.npy')
C_weight11 = np.load('ConstW_11.npy')
C_weight12 = np.load('ConstW_12.npy')
C_weight13 = np.load('ConstW_13.npy')
C_weight14 = np.load('ConstW_14.npy')
#C_weight15 = np.load('ConstW_15.npy')
C_weight16 = np.load('ConstW_16.npy')
C_weight17 = np.load('ConstW_17.npy')
C_weight18 = np.load('ConstW_18.npy')
C_weight19 = np.load('ConstW_19.npy')
#C_weight20 = np.load('ConstW_20.npy')

Cws = [C_weight1,C_weight2,C_weight3,C_weight4,C_weight6,C_weight7,C_weight8,C_weight9,C_weight11,C_weight12,C_weight13\
       ,C_weight14,C_weight16,C_weight17,C_weight18,C_weight19]

R_weight1 = np.load('RandW_1.npy')
R_weight2 = np.load('RandW_2.npy')
R_weight3 =np.load('RandW_3.npy')
R_weight4 =np.load('RandW_4.npy')
#R_weight5 = np.load('RandW_5.npy')
R_weight6 = np.load('RandW_6.npy')
R_weight7 = np.load('RandW_7.npy')
R_weight8 = np.load('RandW_8.npy')
R_weight9 = np.load('RandW_9.npy')
#R_weight10 =np.load('RandW_10.npy')
R_weight11 = np.load('RandW_11.npy')
R_weight12 = np.load('RandW_12.npy')
R_weight13 = np.load('RandW_13.npy')
R_weight14 = np.load('RandW_14.npy')
#R_weight15 = np.load('RandW_15.npy')
R_weight16 = np.load('RandW_16.npy')
R_weight17 = np.load('RandW_17.npy')
R_weight18 = np.load('RandW_18.npy')
R_weight19 = np.load('RandW_19.npy')
#R_weight20 = np.load('RandW_20.npy')

Rws = [R_weight1,R_weight2,R_weight3,R_weight4,R_weight6,R_weight7,R_weight8,R_weight9,R_weight11,R_weight12,R_weight13\
       ,R_weight14,R_weight16,R_weight17,R_weight18,R_weight19]
    
B_weight1 = np.load('BaseW_1.npy')
B_weight2 = np.load('BaseW_2.npy')
B_weight3 =np.load('BaseW_3.npy')
B_weight4 =np.load('BaseW_4.npy')
#B_weight5 = np.load('BaseW_5.npy')
B_weight6 = np.load('BaseW_6.npy')
B_weight7 = np.load('BaseW_7.npy')
B_weight8 = np.load('BaseW_8.npy')
B_weight9 = np.load('BaseW_9.npy')
#B_weight10 =np.load('BaseW_10.npy')
B_weight11 = np.load('BaseW_11.npy')
B_weight12 = np.load('BaseW_12.npy')
B_weight13 = np.load('BaseW_13.npy')
B_weight14 = np.load('BaseW_14.npy')
#B_weight15 = np.load('BaseW_15.npy')
B_weight16 = np.load('BaseW_16.npy')
B_weight17 = np.load('BaseW_17.npy')
B_weight18 = np.load('BaseW_18.npy')
B_weight19 = np.load('BaseW_19.npy')
#B_weight20 = np.load('BaseW_20.npy')

Bws = [B_weight1,B_weight2,B_weight3,B_weight4,B_weight6,B_weight7,B_weight8,B_weight9,B_weight11,B_weight12,B_weight13\
       ,B_weight14,B_weight16,B_weight17,B_weight18,B_weight19]

Wws = np.mean(Wws,axis=0)
Cws = np.mean(Cws,axis=0)
Rws = np.mean(Rws,axis=0)
Bws = np.mean(Bws,axis=0)

plt.figure()
plt.title('Average weight trajectories')
plt.plot(np.linspace(0,65,32500),Wws,label='Optimal Design')
plt.plot(np.linspace(0,65,32500),Cws,label='Constant 20hz')
plt.plot(np.linspace(0,65,32500),Rws,label='Random frequency')
plt.plot(np.linspace(0,65,32500),Bws,label='Baseline firing')
plt.legend()
plt.show()
