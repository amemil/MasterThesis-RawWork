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

sns.set_style("darkgrid")

'''
Tau1 = np.load('TauSamples1to4.npy')
Tau2 = np.load('TauSamples5to8.npy')
Tau3 = np.load('TauSamples9to12.npy')
Tau4 = np.load('TauSamples13to16.npy')
Tau5 = np.load('TauSamples17to20.npy')

taus = [Tau1,Tau2,Tau3,Tau4,Tau5]
means = []
stds = []
for i in range(5):
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
plt.ylim([0,0.1])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], means[i], yerr = meansvar[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

#A

A1 = np.load('ASamples1to4.npy')
A2 = np.load('ASamples5to8.npy')
A3 = np.load('ASamples9to12.npy')
A4 = np.load('ASamples13to16.npy')
A5 = np.load('ASamples17to20.npy')

As = [A1,A2,A3,A4,A5]
meansA = []
stdsa = []
for i in range(5):
    for j in range(4):
        meansA_temp = []
        stdsa_temp = []
        for k in range(5):
            meansA_temp.append(np.mean(As[i][j][k][300:][0]))
            stdsa_temp.append(np.sqrt(np.var(As[i][j][k][300:][0])))
        stdsa.append(stdsa_temp)
        meansA.append(meansA_temp)
        
meansvarA = np.sqrt(np.asarray(meansA).var(0))     
meansA = np.asarray(meansA).mean(0)

stdsa = np.asarray(stdsa).mean(0)
    


x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $A$ - Not stimulated')
plt.xlabel('Datasize')
plt.ylabel('$A$ estimation')
plt.ylim([0,0.02])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meansA[i], yerr = meansvarA[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''

#Sim
 

Sim1 = np.load('SimSamples1to4.npy')
Sim2 = np.load('SimSamples5to8.npy')
Sim3 =np.load('SimSamples9to12.npy')
Sim4 =np.load('SimSamples13to16.npy')
Sim5 =np.load('SimSamples17to20.npy')

sims = [Sim1,Sim3,Sim4,Sim5]

meanssA = []
meanssT = []

stdssA = []
stdssT = []

for i in range(4):
    for j in range(4):
        mat = []
        mtt = []
        sat = []
        stt = []
        for k in range(5):
            mat.append(np.mean(sims[i][j][k][300:],axis=0)[0])
            mtt.append(np.mean(sims[i][j][k][300:],axis=0)[1])
            sat.append(np.sqrt(np.var(sims[i][j][k][300:],axis=0)[0]))
            stt.append(np.sqrt(np.var(sims[i][j][k][300:],axis=0)[1]))
        meanssA.append(mat)
        meanssT.append(mtt)
        stdssA.append(sat)
        stdssT.append(stt)
        
#meanssA.pop(5)
#meanssT.pop(5)
#stdssA.pop(5)
#stdssT.pop(5)

def rmse(targets, predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())


def lr1(s2,s1,Ap,delta,taup):
    return s2*s1*Ap*np.exp(-delta/taup)

def lr2(s1,s2,Am,delta,taum):
    return -s1*s2*Am*1.05*np.exp(delta/taum)

deltas = np.linspace(0,0.1,10000)
deltas2 = np.linspace(-0.1,0,10000)   
lrs1 = lr1(1,1,0.005,deltas,0.02)
lrs2 = lr2(1,1,0.005,deltas2,0.02) 

lrref = np.concatenate((lrs2,lrs1))


rmselrbase = []
datasizes3 = []
for i in range(len(meanssA)):
    for j in range(5):
        lrs1_temp = lr1(1,1,meanssA[i][j],deltas,meanssT[i][j])
        lrs2_temp = lr2(1,1,meanssA[i][j],deltas2,meanssT[i][j])
        lr_est = np.concatenate((lrs2_temp,lrs1_temp))
        rmselrbase.append(rmse(lrref, lr_est))
        datasizes3.append((j+1)*60)
      
'''
meanssA = np.mean(meanssA,axis=0)
meanssT = np.mean(meanssT,axis=0)
stdssA = np.mean(stdssA,axis=0)
stdssT = np.mean(stdssT,axis=0)

x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $A$ - Baseline stimulus')
plt.xlabel('Datasize')
plt.ylabel('$A$ estimation')
plt.ylim([0,0.02])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meanssA[i], yerr = stdssA[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Inference of $Tau$ - Baseline stimulus')
plt.xlabel('Datasize')
plt.ylabel('$Tau$ estimation')
plt.ylim([0,0.06])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meanssT[i], yerr = stdssT[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()


'''