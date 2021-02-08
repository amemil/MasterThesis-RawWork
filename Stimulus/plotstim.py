#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:13:26 2021

@author: emilam
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import gamma


sns.set_style("darkgrid")

tau0 = np.load('TauSampleStimulatedm0.npy')
tau1 = np.load('TauSampleStimulatedm1.5.npy')

means0 = []
stds0 = []
means1 = []
stds1 = []
for i in range(5):
    means0.append(np.mean(tau0[i,300:,0]))
    means1.append(np.mean(tau1[i,300:,0]))
    stds0.append(np.sqrt(np.var((tau0[i,300:,0]))))
    stds1.append(np.sqrt(np.var((tau1[i,300:,0]))))

x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('$Tau$ sample means - High input')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$Tau$ estimation')
plt.ylim([0,0.04])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], means0[i], yerr = stds0[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('$Tau$ sample means - Medium input')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$Tau$ estimation')
plt.ylim([0,0.04])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], means1[i], yerr = stds1[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()



a0 = np.load('ASampleStimulatedm0.npy')
a1 = np.load('ASampleStimulatedm1.5.npy')

means00 = []
stds00 = []
means11 = []
stds11 = []
for i in range(5):
    means00.append(np.mean(a0[i,300:,0]))
    means11.append(np.mean(a1[i,300:,0]))
    stds00.append(np.sqrt(np.var((a0[i,300:,0]))))
    stds11.append(np.sqrt(np.var((a1[i,300:,0]))))

x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('$A_+$ sample means - High input')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$Tau$ estimation')
plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], means00[i], yerr = stds00[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('$A_+$ sample means - Medium input')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$Tau$ estimation')
plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], means11[i], yerr = stds11[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()