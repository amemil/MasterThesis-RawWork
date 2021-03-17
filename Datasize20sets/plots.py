#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:14:41 2021

@author: emilam
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

Tau1 = np.load('TauSamples1to4.npy')
Tau2 = np.load('TauSamples5to8.npy')

taus = [Tau1,Tau2]
means = []
stds = []
for i in range(2):
    for j in range(4):
        means_temp = []
        stds_temp = []
        for k in range(5):
            means_temp.append(np.mean(taus[i][j][k][300:][0]))
            stds_temp.append(np.sqrt(np.var(taus[i][j][k][300:][0])))
        stds.append(stds_temp)
        means.append(means_temp)
        
meansvar = np.sqrt(np.asarray(means).var(0))     
means = np.asarray(means).mean(0)

stds = np.asarray(stds).mean(0)
    


x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $Tau$ - Not stimulated')
plt.xlabel('Datasize')
plt.ylabel('$Tau$ estimation')
plt.ylim([0,0.05])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], means[i], yerr = meansvar[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()