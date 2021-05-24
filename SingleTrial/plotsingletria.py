#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:35:37 2021

@author: emilam
"""

import numpy as np              
import matplotlib.pyplot as plt 
from scipy.interpolate import make_interp_spline, BSpline
import math
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
from scipy.stats import norm

plt.style.use('default')

emsb = np.load('1msbase3.npy') #1s trial baseline inference
emsf = np.load('1msfreq3.npy') #1s trial 50hz inference 
fmsb = np.load('5msbase3.npy') #5s trial baseline infernce 
fmsf = np.load('5msfreq3.npy') #5s trial 50hz inference
tensb = np.load('TenSecBase.npy')#10s trial baseline inference
tensf = np.load('TenSec50hz.npy') #10s trial baseline inference
preb = np.load('prebase3.npy') #presyn spiketrain baseline
postb = np.load('postbase3.npy') #postsyn spiketrain baseline
pref= np.load('prefreq3.npy')#presyn spiketrain 50hz
postf=np.load('postfreq3.npy')#postsyn spiketrain 50hz
wb=np.load('weightbase3.npy')#weight traj baseline
wf=np.load('weightfreq3.npy')#weight traj 50hz

w20f = np.load('weight20hz.npy')#weight traj 20hz
pre20hz = np.load('pre20hz.npy')#presyn spiketrain 20hz
post20hz = np.load('pre20hz.npy')#postsyn spiketrain 20hz

ts20hz = np.load('tensec20hzSample.npy') #10s trial 20hz inference
fs20hz= np.load('fivesec20hzSample.npy') #5s trial 20hz inference
es20hz= np.load('onesec20hzSample.npy') #1s trial 20hz inference

w100f = np.load('weight100hz.npy')#weight traj 100hz
ts100hz = np.load('tensec100hzSample.npy') #10s trial 100hz inference
fs100hz= np.load('fivesec100hzSample.npy') #5s trial 100hz inference
es100hz= np.load('onesec100hzSample.npy') #1s trial 100hz inference
pre100hz = np.load('pre100hz.npy')#presyn spiketrain 100hz
post100hz = np.load('pre100hz.npy')#postsyn spiketrain 100hz

    
frbase = []
frf = []
fr20 = []
fr100 = []
for i in range(120):
    counter = 0
    counter2 = 0
    counter3 =0
    counter4 =0
    for j in range(i*500,(i+1)*500):
        if postb[j] == 1:
            counter += 1
        if postf[j] == 1:
            counter2 += 1
        if post20hz[j] == 1:
            counter3 += 1
        if post100hz[j] == 1:
            counter4 += 1
    frbase.append(counter)
    frf.append(counter2)
    fr20.append(counter3)
    fr100.append(counter4)


trueA = 0.005
trueTau = 0.02
    
def rmse(targets, predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())

def lr1(s2,s1,Ap,delta,taup):
    return s2*s1*Ap*np.exp(-delta/taup)

def lr2(s1,s2,Ap,delta,taum):
    return -s1*s2*Ap*1.05*np.exp(delta/taum)


deltas = np.linspace(0,0.1,10000)
deltas2 = np.linspace(-0.1,0,10000)   
lrs1 = lr1(1,1,trueA,deltas,trueTau)
lrs2 = lr2(1,1,trueA,deltas2,trueTau) 

lrref = np.concatenate((lrs2,lrs1))

def NormEntropy(sigma):
    return 0.5 * np.log(np.linalg.det(2*np.pi*np.exp(1)*sigma))

    
base1msmeans = []
base1msstds = []
base5msmeans = []
base5msstds = []
f1msmeans = []
f1msstds = []
f5msmeans = []
f5msstds = []
base10msmeans = []
base10msstds = []
f10msmeans= []
f10msstds = []

absoluteErrorA1s = []
absoluteErrorA5s = []
absoluteErrorA10s = []

absoluteErrorTau1s = []
absoluteErrorTau5s = []
absoluteErrorTau10s = []

entropiesA1s = []
entropiesA5s = []
entropiesA10s = []

entropiesTau1s = []
entropiesTau5s = []
entropiesTau10s = []

entropy2d1s = []
entropy2d5s = []
entropy2d10s = []
rmselr1s = []
rmselr5s = []
rmselr10s = []

for i in range(120):
    base1msmeans.append(np.mean(emsb[i][300:],axis=0))
    base1msstds.append(np.sqrt(np.var(emsb[i][300:],axis=0)))
    f1msmeans.append(np.mean(emsf[i][300:],axis=0))
    f1msstds.append(np.sqrt(np.var(emsf[i][300:],axis=0)))
    absoluteErrorA1s.append(abs(np.mean(emsb[i][300:],axis=0) - trueA))
    absoluteErrorA1s.append(abs(np.mean(emsf[i][300:],axis=0) - trueA))
    lrs1_temp = lr1(1,1,base1msmeans[-1][0],deltas,base1msmeans[-1][1])
    lrs2_temp = lr2(1,1,base1msmeans[-1][0],deltas2,base1msmeans[-1][1])
    lr_est = np.concatenate((lrs2_temp,lrs1_temp))
    rmselr1s.append(rmse(lrref, lr_est))
    lrs1_temp = lr1(1,1,f1msmeans[-1][0],deltas,f1msmeans[-1][1])
    lrs2_temp = lr2(1,1,f1msmeans[-1][0],deltas2,f1msmeans[-1][1])
    lr_est = np.concatenate((lrs2_temp,lrs1_temp))
    rmselr1s.append(rmse(lrref, lr_est))
    entropiesA1s.append(norm.entropy(loc = np.mean(emsb[i][300:],axis=0),scale = np.sqrt(np.var(emsb[i][300:],axis=0))))
    entropiesA1s.append(norm.entropy(loc = np.mean(emsf[i][300:],axis=0),scale = np.sqrt(np.var(emsf[i][300:],axis=0))))
    entropy2d1s.append(NormEntropy(np.cov(np.transpose(emsb[i][300:,:]))))
    entropy2d1s.append(NormEntropy(np.cov(np.transpose(emsf[i][300:,:]))))
for i in range(24):
    base5msmeans.append(np.mean(fmsb[i][300:],axis=0))
    base5msstds.append(np.sqrt(np.var(fmsb[i][300:],axis=0)))
    f5msmeans.append(np.mean(fmsf[i][300:],axis=0))
    f5msstds.append(np.sqrt(np.var(fmsf[i][300:],axis=0)))
    absoluteErrorA5s.append(abs(np.mean(fmsb[i][300:],axis=0) - trueA))
    absoluteErrorA5s.append(abs(np.mean(fmsf[i][300:],axis=0) - trueA))
    lrs1_temp = lr1(1,1,base5msmeans[-1][0],deltas,base5msmeans[-1][1])
    lrs2_temp = lr2(1,1,base5msmeans[-1][0],deltas2,base5msmeans[-1][1])
    lr_est = np.concatenate((lrs2_temp,lrs1_temp))
    rmselr5s.append(rmse(lrref, lr_est))
    lrs1_temp = lr1(1,1,f5msmeans[-1][0],deltas,f5msmeans[-1][1])
    lrs2_temp = lr2(1,1,f5msmeans[-1][0],deltas2,f5msmeans[-1][1])
    lr_est = np.concatenate((lrs2_temp,lrs1_temp))
    rmselr5s.append(rmse(lrref, lr_est))
    entropiesA5s.append(norm.entropy(loc = np.mean(fmsb[i][300:],axis=0),scale = np.sqrt(np.var(fmsb[i][300:],axis=0))))
    entropiesA5s.append(norm.entropy(loc = np.mean(fmsf[i][300:],axis=0),scale = np.sqrt(np.var(fmsf[i][300:],axis=0))))
    entropy2d5s.append(NormEntropy(np.cov(np.transpose(fmsb[i][300:,:]))))
    entropy2d5s.append(NormEntropy(np.cov(np.transpose(fmsf[i][300:,:]))))
for i in range(12):
    base10msmeans.append(np.mean(tensb[i][300:],axis=0))
    base10msstds.append(np.sqrt(np.var(tensb[i][300:],axis=0)))
    f10msmeans.append(np.mean(tensf[i][300:],axis=0))
    f10msstds.append(np.sqrt(np.var(tensf[i][300:],axis=0)))
    absoluteErrorA10s.append(abs(np.mean(tensb[i][300:],axis=0) - trueA))
    absoluteErrorA10s.append(abs(np.mean(tensf[i][300:],axis=0) - trueA))
    lrs1_temp = lr1(1,1,base10msmeans[-1][0],deltas,base10msmeans[-1][1])
    lrs2_temp = lr2(1,1,base10msmeans[-1][0],deltas2,base10msmeans[-1][1])
    lr_est = np.concatenate((lrs2_temp,lrs1_temp))
    rmselr10s.append(rmse(lrref, lr_est))
    lrs1_temp = lr1(1,1,f10msmeans[-1][0],deltas,f10msmeans[-1][1])
    lrs2_temp = lr2(1,1,f10msmeans[-1][0],deltas2,f10msmeans[-1][1])
    lr_est = np.concatenate((lrs2_temp,lrs1_temp))
    rmselr10s.append(rmse(lrref, lr_est))
    entropiesA10s.append(norm.entropy(loc = np.mean(tensb[i][300:],axis=0),scale = np.sqrt(np.var(tensb[i][300:],axis=0))))
    entropiesA10s.append(norm.entropy(loc = np.mean(tensf[i][300:],axis=0),scale = np.sqrt(np.var(tensf[i][300:],axis=0))))
    entropy2d10s.append(NormEntropy(np.cov(np.transpose(tensb[i][300:,:]))))
    entropy2d10s.append(NormEntropy(np.cov(np.transpose(tensf[i][300:,:]))))

tw1msmeans = []
tw1msstds = []
tw5msmeans = []
tw5msstds = []
hun1msmeans = []
hun1msstds = []
hun5msmeans = []
hun5msstds = []
tw10msmeans = []
tw10msstds = []
hun10msmeans= []
hun10msstds = []

for i in range(120):
    tw1msmeans.append(np.mean(es20hz[i][300:],axis=0))
    tw1msstds.append(np.sqrt(np.var(es20hz[i][300:],axis=0)))
    hun1msmeans.append(np.mean(es100hz[i][300:],axis=0))
    hun1msstds.append(np.sqrt(np.var(es100hz[i][300:],axis=0)))
    absoluteErrorA1s.append(abs(np.mean(es20hz[i][300:],axis=0) - trueA))
    absoluteErrorA1s.append(abs(np.mean(es100hz[i][300:],axis=0) - trueA))
    lrs1_temp = lr1(1,1,tw1msmeans[-1][0],deltas,tw1msmeans[-1][1])
    lrs2_temp = lr2(1,1,tw1msmeans[-1][0],deltas2,tw1msmeans[-1][1])
    lr_est = np.concatenate((lrs2_temp,lrs1_temp))
    rmselr1s.append(rmse(lrref, lr_est))
    lrs1_temp = lr1(1,1,hun1msmeans[-1][0],deltas,hun1msmeans[-1][1])
    lrs2_temp = lr2(1,1,hun1msmeans[-1][0],deltas2,hun1msmeans[-1][1])
    lr_est = np.concatenate((lrs2_temp,lrs1_temp))
    rmselr1s.append(rmse(lrref, lr_est))
    entropiesA1s.append(norm.entropy(loc = np.mean(es20hz[i][300:],axis=0),scale = np.sqrt(np.var(es20hz[i][300:],axis=0))))
    entropiesA1s.append(norm.entropy(loc = np.mean(es100hz[i][300:],axis=0),scale = np.sqrt(np.var(es100hz[i][300:],axis=0))))
    entropy2d1s.append(NormEntropy(np.cov(np.transpose(es20hz[i][300:,:]))))
    entropy2d1s.append(NormEntropy(np.cov(np.transpose(es100hz[i][300:,:]))))
for i in range(24):
    tw5msmeans.append(np.mean(fs20hz[i][300:],axis=0))
    tw5msstds.append(np.sqrt(np.var(fs20hz[i][300:],axis=0)))
    hun5msmeans.append(np.mean(fs100hz[i][300:],axis=0))
    hun5msstds.append(np.sqrt(np.var(fs100hz[i][300:],axis=0)))
    absoluteErrorA5s.append(abs(np.mean(fs20hz[i][300:],axis=0) - trueA))
    absoluteErrorA5s.append(abs(np.mean(fs100hz[i][300:],axis=0) - trueA))
    lrs1_temp = lr1(1,1,tw5msmeans[-1][0],deltas,tw5msmeans[-1][1])
    lrs2_temp = lr2(1,1,tw5msmeans[-1][0],deltas2,tw5msmeans[-1][1])
    lr_est = np.concatenate((lrs2_temp,lrs1_temp))
    rmselr5s.append(rmse(lrref, lr_est))
    lrs1_temp = lr1(1,1,hun5msmeans[-1][0],deltas,hun5msmeans[-1][1])
    lrs2_temp = lr2(1,1,hun5msmeans[-1][0],deltas2,hun5msmeans[-1][1])
    lr_est = np.concatenate((lrs2_temp,lrs1_temp))
    rmselr5s.append(rmse(lrref, lr_est))
    entropiesA5s.append(norm.entropy(loc = np.mean(fs20hz[i][300:],axis=0),scale = np.sqrt(np.var(fs20hz[i][300:],axis=0))))
    entropiesA5s.append(norm.entropy(loc = np.mean(fs100hz[i][300:],axis=0),scale = np.sqrt(np.var(fs100hz[i][300:],axis=0))))
    entropy2d5s.append(NormEntropy(np.cov(np.transpose(fs20hz[i][300:,:]))))
    entropy2d5s.append(NormEntropy(np.cov(np.transpose(fs100hz[i][300:,:]))))
for i in range(12):
    tw10msmeans.append(np.mean(ts20hz[i][300:],axis=0))
    tw10msstds.append(np.sqrt(np.var(ts20hz[i][300:],axis=0)))
    hun10msmeans.append(np.mean(ts100hz[i][300:],axis=0))
    hun10msstds.append(np.sqrt(np.var(ts100hz[i][300:],axis=0)))
    absoluteErrorA10s.append(abs(np.mean(ts20hz[i][300:],axis=0) - trueA))
    absoluteErrorA10s.append(abs(np.mean(ts100hz[i][300:],axis=0) - trueA))
    lrs1_temp = lr1(1,1,tw10msmeans[-1][0],deltas,tw10msmeans[-1][1])
    lrs2_temp = lr2(1,1,tw10msmeans[-1][0],deltas2,tw10msmeans[-1][1])
    lr_est = np.concatenate((lrs2_temp,lrs1_temp))
    rmselr10s.append(rmse(lrref, lr_est))
    lrs1_temp = lr1(1,1,hun10msmeans[-1][0],deltas,hun10msmeans[-1][1])
    lrs2_temp = lr2(1,1,hun10msmeans[-1][0],deltas2,hun10msmeans[-1][1])
    lr_est = np.concatenate((lrs2_temp,lrs1_temp))
    rmselr10s.append(rmse(lrref, lr_est))
    entropiesA10s.append(norm.entropy(loc = np.mean(ts20hz[i][300:],axis=0),scale = np.sqrt(np.var(ts20hz[i][300:],axis=0))))
    entropiesA10s.append(norm.entropy(loc = np.mean(ts100hz[i][300:],axis=0),scale = np.sqrt(np.var(ts100hz[i][300:],axis=0))))
    entropy2d10s.append(NormEntropy(np.cov(np.transpose(ts20hz[i][300:,:]))))
    entropy2d10s.append(NormEntropy(np.cov(np.transpose(ts100hz[i][300:,:]))))


trueA = 0.005
trueTau = 0.02

'''
x = np.linspace(1,120,120)

fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 1s trials A estimation - Baseline vs 50Hz', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            #it_temp = []
            #meanstemp = []
            for i in range(120):
                ax.errorbar(x[i], base1msmeans[i][0], yerr = base1msstds[i][0],marker = 'o')
                #if math.isnan(base1msmeans[i][0]) == False:
                #    meanstemp.append(base1msmeans[i][0])
                #    it_temp.append(i+1)
            #Final_array_smooth = gaussian_filter1d(meanstemp, sigma=2)
            #x_int = np.linspace(1,120,len(Final_array_smooth))
            #smooth = make_interp_spline(it_temp, meanstemp,k=1)
            #y_int = smooth(x_int)
            #ax.plot(it_temp,Final_array_smooth,'k-')
            #sns.lineplot(it_temp,meanstemp)
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            #ax.plot(it_temp,Final_array_smooth,'k-')
            ax.set_ylabel('A estimation')
            ax.set_title('Baseline stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(120):
                ax.errorbar(x[i], f1msmeans[i][0], yerr = f1msstds[i][0],marker = 'o')
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('50 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wb)),wb, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frbase, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wf)),wf, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frf, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()


x = np.linspace(5,120,24)
fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 5s trials A estimation - Baseline vs 50Hz', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(24):
                ax.errorbar(x[i], base5msmeans[i][0], yerr = base5msstds[i][0],marker = 'o') 
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('A estimation')
            ax.set_title('Baseline stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(24):
                ax.errorbar(x[i], f5msmeans[i][0], yerr = f5msstds[i][0],marker = 'o')   
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('50 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wb)),wb, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frbase, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wf)),wf, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frf, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()

x = np.linspace(1,120,120)
fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 1s trials Tau estimation - Baseline vs 50Hz', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(120):
                ax.errorbar(x[i], base1msmeans[i][1], yerr = base1msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('Tau estimation')
            ax.set_title('Baseline stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(120):
                ax.errorbar(x[i], f1msmeans[i][1], yerr = f1msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            #.set_xlabel('A estimation')
            ax.set_title('50 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wb)),wb, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frbase, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wf)),wf, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frf, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()

x = np.linspace(5,120,24)
fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 5s trials Tau estimation - Baseline vs 50Hz', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(24):
                ax.errorbar(x[i], base5msmeans[i][1], yerr = base5msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('Tau estimation')
            ax.set_title('Baseline stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(24):
                ax.errorbar(x[i], f5msmeans[i][1], yerr = f5msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('50 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wb)),wb, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frbase, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wf)),wf, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frf, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()

x = np.linspace(10,120,12)
fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 10s trials Tau estimation - Baseline vs 50Hz', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(12):
                ax.errorbar(x[i], base10msmeans[i][1], yerr = base10msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('Tau estimation')
            ax.set_title('Baseline stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(12):
                ax.errorbar(x[i], f10msmeans[i][1], yerr = f10msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('50 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wb)),wb, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frbase, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wf)),wf, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frf, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 10s trials A estimation - Baseline vs 50Hz', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(12):
                ax.errorbar(x[i], base10msmeans[i][0], yerr = base10msstds[i][0],marker = 'o')        
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('A estimation')
            ax.set_title('Baseline stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(12):
                ax.errorbar(x[i], f10msmeans[i][0], yerr = f10msstds[i][0],marker = 'o')        
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('50 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wb)),wb, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frbase, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wf)),wf, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frf, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()


x = np.linspace(1,120,120)
fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 1s trials A estimation - 20Hz vs 100Hz', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(120):
                ax.errorbar(x[i], tw1msmeans[i][0], yerr = tw1msstds[i][0],marker = 'o')        
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('A estimation')
            ax.set_title('20Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(120):
                ax.errorbar(x[i], hun1msmeans[i][0], yerr = hun1msstds[i][0],marker = 'o')        
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('100 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(w20f)),w20f, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),fr20, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(w100f)),w100f, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),fr100, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 1s trials Tau estimation - 20Hz vs 100Hz', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(120):
                ax.errorbar(x[i], tw1msmeans[i][1], yerr = tw1msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('Tau estimation')
            ax.set_title('20Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(120):
                ax.errorbar(x[i], hun1msmeans[i][1], yerr = hun1msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('100 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(w20f)),w20f, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),fr20, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(w100f)),w100f, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),fr100, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()

x = np.linspace(5,120,24)
fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 5s trials A estimation - 20Hz vs 100Hz', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(24):
                ax.errorbar(x[i], tw5msmeans[i][0], yerr = tw5msstds[i][0],marker = 'o')        
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('A estimation')
            ax.set_title('20Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(24):
                ax.errorbar(x[i], hun5msmeans[i][0], yerr = hun5msstds[i][0],marker = 'o')        
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('100 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(w20f)),w20f, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),fr20, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(w100f)),w100f, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),fr100, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 5s trials Tau estimation - 20Hz vs 100Hz', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(24):
                ax.errorbar(x[i], tw5msmeans[i][1], yerr = tw5msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('Tau estimation')
            ax.set_title('20Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(24):
                ax.errorbar(x[i], hun5msmeans[i][1], yerr = hun5msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('100 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(w20f)),w20f, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),fr20, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(w100f)),w100f, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),fr100, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()

x = np.linspace(10,120,12)
fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 10s trials A estimation - 20Hz vs 100Hz', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(12):
                ax.errorbar(x[i], tw10msmeans[i][0], yerr = tw10msstds[i][0],marker = 'o')        
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('A estimation')
            ax.set_title('20Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(12):
                ax.errorbar(x[i], hun10msmeans[i][0], yerr = hun10msstds[i][0],marker = 'o')        
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('100 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(w20f)),w20f, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),fr20, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(w100f)),w100f, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),fr100, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 10s trials Tau estimation - 20Hz vs 100Hz', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(12):
                ax.errorbar(x[i], tw10msmeans[i][1], yerr = tw10msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('Tau estimation')
            ax.set_title('20Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(12):
                ax.errorbar(x[i], hun10msmeans[i][1], yerr = hun10msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('100 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(w20f)),w20f, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),fr20, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(w100f)),w100f, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),fr100, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()
'''
plt.figure()
plt.title('Entropy versus RMSE (A estimation)')
plt.xlabel('RMSE')
plt.ylabel('Entropy')
plt.yticks(ticks=[-2,-3,-4,-5],labels=['-2','-3','-4','-5'])
for i in range(len(absoluteErrorA1s)):
    if math.isnan(absoluteErrorA1s[i][0]) == False and entropiesA1s[i][0] > -6:
        if i == 0:
            plt.plot(absoluteErrorA1s[i][0],entropiesA1s[i][0],'ro',label = '1s trials',alpha=0.4)
        else:
            plt.plot(absoluteErrorA1s[i][0],entropiesA1s[i][0],'ro',alpha=0.4)
for i in range(len(absoluteErrorA5s)):
    if math.isnan(absoluteErrorA5s[i][0]) == False and entropiesA5s[i][0] > -6:
        if i == 0:
            plt.plot(absoluteErrorA5s[i][0],entropiesA5s[i][0],'go',label = '5s trials',alpha=0.4)
        else:
            plt.plot(absoluteErrorA5s[i][0],entropiesA5s[i][0],'go',alpha=0.4)
for i in range(len(absoluteErrorA10s)):
    if math.isnan(absoluteErrorA10s[i][0]) == False and entropiesA10s[i][0] > -6:
        if i == 2:
            plt.plot(absoluteErrorA10s[i][0],entropiesA10s[i][0],'bo',label = '10s trials',alpha=0.4)
        else:
            plt.plot(absoluteErrorA10s[i][0],entropiesA10s[i][0],'bo',alpha=0.4)
plt.legend()
plt.show()

plt.figure()
plt.title('Entropy versus RMSE (Tau estimation)')
plt.xlabel('RMSE')
plt.ylabel('Entropy')
plt.yticks(ticks=[-2,-3,-4],labels=['-2','-3','-4'])
for i in range(len(absoluteErrorA1s)):
    if math.isnan(absoluteErrorA1s[i][1]) == False and entropiesA1s[i][1] > -5:
        if i == 0:
            plt.plot(absoluteErrorA1s[i][1],entropiesA1s[i][1],'ro',label = '1s trials',alpha=0.4)
        else:
            plt.plot(absoluteErrorA1s[i][1],entropiesA1s[i][1],'ro',alpha=0.4)
for i in range(len(absoluteErrorA5s)):
    if math.isnan(absoluteErrorA5s[i][1]) == False and entropiesA5s[i][1] > -5:
        if i == 0:
            plt.plot(absoluteErrorA5s[i][1],entropiesA5s[i][1],'go',label = '5s trials',alpha=0.4)
        else:
            plt.plot(absoluteErrorA5s[i][1],entropiesA5s[i][1],'go',alpha=0.4)
for i in range(len(absoluteErrorA10s)):
    if math.isnan(absoluteErrorA10s[i][1]) == False and entropiesA10s[i][1] > -5:
        if i == 2:
            plt.plot(absoluteErrorA10s[i][1],entropiesA10s[i][1],'bo',label = '10s trials',alpha=0.4)
        else:
            plt.plot(absoluteErrorA10s[i][1],entropiesA10s[i][1],'bo',alpha=0.4)
plt.legend()
plt.show()

plt.figure()
plt.title('Entropy versus RMSE learning rule')
plt.xlabel('RMSE')
plt.ylabel('Entropy')
plt.yticks(ticks=[-4,-6,-8],labels=['-4','-6','-8'])
for i in range(len(rmselr1s)):
    if math.isnan(rmselr1s[i]) == False and entropy2d1s[i] > -10:
        if i == 0:
            plt.plot(rmselr1s[i],entropy2d1s[i],'ro',label = '1s trials',alpha=0.4)
        else:
            plt.plot(rmselr1s[i],entropy2d1s[i],'ro',alpha=0.4)
for i in range(len(rmselr5s)):
    if math.isnan(rmselr5s[i]) == False and entropy2d5s[i] > -10:
        if i == 0:
            plt.plot(rmselr5s[i],entropy2d5s[i],'go',label = '5s trials',alpha=0.4)
        else:
            plt.plot(rmselr5s[i],entropy2d5s[i],'go',alpha=0.4)
for i in range(len(rmselr10s)):
    if math.isnan(rmselr10s[i]) == False and entropy2d10s[i] > -10:
        if i == 2:
            plt.plot(rmselr10s[i],entropy2d10s[i],'bo',label = '10s trials',alpha=0.4)
        else:
            plt.plot(rmselr10s[i],entropy2d10s[i],'bo',alpha=0.4)
plt.legend()
plt.show()
            
    