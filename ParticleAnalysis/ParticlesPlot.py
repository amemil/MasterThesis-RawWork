#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:14:17 2021

@author: emilam
"""
## A
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

A1 = np.load('Aps3to4.npy')
A2= np.load('Aps9to10.npy')
A3=np.load('Aps11to12.npy')
A4=np.load('Aps13to14.npy')
A5=np.load('Aps19to20.npy')

As = [A1,A2,A3,A4,A5]
means =[]
stds = []
for i in range(5):
    for j in range(2):
        means_temp = []
        stds_temp = []
        for k in range(7):
            means_temp.append(np.mean(As[i][j][k][300:]))
            stds_temp.append(np.sqrt(np.var(As[i][j][k][300:])))
        stds.append(stds_temp)
        means.append(means_temp)
        
meansvar = np.sqrt(np.asarray(means).var(0))     
means = np.asarray(means).mean(0)

stds = np.asarray(stds).mean(0)
    
## Tau

Tau1 = np.load('Taus3to4.npy')
Tau2= np.load('Taus9to10.npy')
Tau3=np.load('Taus11to12.npy')
Tau4=np.load('Taus13to14.npy')
Tau5=np.load('Taus19to20.npy')

Taus = [Tau1,Tau2,Tau3,Tau4,Tau5]
means2 =[]
stds2 = []
for i in range(5):
    for j in range(2):
        means_temp = []
        stds_temp = []
        for k in range(7):
            means_temp.append(np.mean(Taus[i][j][k][300:]))
            stds_temp.append(np.sqrt(np.var(Taus[i][j][k][300:])))
        stds2.append(stds_temp)
        means2.append(means_temp)
        
meansvar2 = np.sqrt(np.asarray(means2).var(0))     
means2 = np.asarray(means2).mean(0)

stds2 = np.asarray(stds2).mean(0)


x = [1,2,3,4,5,6,7]
ticksss = ['50','100','200','500','1000','2000','5000']
plt.figure()
plt.title('Inference of $A_+$ - Means 20 datasets')
plt.xlabel('Number of particles')
plt.ylabel('$A_+$ estimation')
plt.ylim([0,0.01])
plt.xlim([0,8])
plt.xticks(x,labels = ticksss)
for i in range(7):
    plt.errorbar(x[i], means[i], yerr = stds[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Inference of $Tau$ - Means 20 datasets')
plt.xlabel('Number of particles')
plt.ylabel('$Tau$ estimation')
plt.ylim([0,0.04])
plt.xlim([0,8])
plt.xticks(x,labels = ticksss)
for i in range(7):
    plt.errorbar(x[i], means2[i], yerr = stds2[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()