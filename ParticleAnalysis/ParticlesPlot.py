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
import matplotlib as mpl

A1 = np.load('Aps1to2.npy')
A2= np.load('Aps3to4.npy')
A3=np.load('Aps5to6.npy')
A4=np.load('Aps7to8.npy')
A5=np.load('Aps9to10.npy')
A6 = np.load('Aps11to12.npy')
A7= np.load('Aps13to14.npy')
A8=np.load('Aps15to16.npy')
A9=np.load('Aps17to18.npy')
A10=np.load('Aps19to20.npy')

As = [A1,A2,A3,A4,A5,A6,A7,A8,A9,A10]
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
Tau1 = np.load('Taus1to2.npy')
Tau2 = np.load('Taus3to4.npy')
Tau3 = np.load('Taus5to6.npy')
Tau4 = np.load('Taus7to8.npy')
Tau5= np.load('Taus9to10.npy')
Tau6=np.load('Taus11to12.npy')
Tau7=np.load('Taus13to14.npy')
Tau8=np.load('Taus15to16.npy')
Tau9=np.load('Taus17to18.npy')
Tau10=np.load('Taus19to20.npy')

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

mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 2
x = [50,100,200,500,1000,2000,5000]
ticksss = ['50','100','200','500','1000','2000','5000']
plt.figure()
#plt.title('Inference of $A_+$ - Means 20 datasets')
plt.xlabel('Number of particles')
plt.ylabel('$A_+$ estimation')
plt.ylim([0.0045,0.0055])
#plt.xlim([0,8])
plt.xticks(x,labels = ticksss)
for i in range(7):
    plt.errorbar(x[i], means[i], yerr = meansvar[i],c=[0.4,0.3,0.9],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
#plt.legend()
plt.xscale('log')
#plt.minorticks_off()
plt.show()
'''
plt.figure()
plt.title('Inference of $Tau$ - Means 20 datasets')
plt.xlabel('Number of particles')
plt.ylabel('$Tau$ estimation')
plt.ylim([0,0.04])
plt.xlim([0,8])
plt.xticks(x,labels = ticksss)
for i in range(7):
    plt.errorbar(x[i], means2[i], yerr = meansvar2[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''