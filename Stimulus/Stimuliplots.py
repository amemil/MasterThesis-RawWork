#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 12:17:00 2021

@author: emilam
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

sns.set_style("darkgrid")


Tau1s = np.load('TauSamples1to4stim.npy')
Tau2s = np.load('TauSamples5to8stim.npy')
Tau3s= np.load('TauSamples9to12stim.npy')
Tau4s = np.load('TauSamples13to16stim.npy')
Tau5s = np.load('TauSamples17to20stim.npy')

tauss = [Tau1s,Tau2s,Tau3s,Tau4s,Tau5s]
meanss = []
stdss = []
for i in range(5):
    for j in range(4):
        meanss_temp = []
        stdss_temp = []
        for k in range(5):
            meanss_temp.append(np.mean(tauss[i][j][k][300:]))
            stdss_temp.append(np.sqrt(np.var(tauss[i][j][k][300:])))
        stdss.append(stdss_temp)
        meanss.append(meanss_temp)
        
meansvars = np.sqrt(np.asarray(meanss).var(0))     
meanss = np.asarray(meanss).mean(0)

stdss = np.asarray(stdss).mean(0)
    


x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $Tau$ - Medium stimuli')
plt.xlabel('Datasize')
plt.ylabel('$Tau$ estimation')
plt.ylim([0,0.1])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meanss[i], yerr = meansvars[i],marker = 'o',color='b')
    #plt.errorbar(x[i],meanss[i],yerr = stdss[i],marker='o',color='k')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

Tau1hs = np.load('TauSamples1to4highstim.npy')
Tau2hs = np.load('TauSamples5to8highstim.npy')
Tau3hs= np.load('TauSamples9to12highstim.npy')


taushs = [Tau1hs,Tau2hs,Tau3hs]
meanshs = []
stdshs = []
for i in range(3):
    for j in range(4):
        meanshs_temp = []
        stdshs_temp = []
        for k in range(5):
            meanshs_temp.append(np.mean(taushs[i][j][k][300:]))
            stdshs_temp.append(np.sqrt(np.var(taushs[i][j][k][300:])))
        stdshs.append(stdshs_temp)
        meanshs.append(meanshs_temp)
        
meansvarhs = np.sqrt(np.asarray(meanshs).var(0))     
meanshs = np.asarray(meanshs).mean(0)

stdshs = np.asarray(stdshs).mean(0)
    


x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $Tau$ - High stimuli')
plt.xlabel('Datasize')
plt.ylabel('$Tau$ estimation')
plt.ylim([0,0.1])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meanshs[i], yerr = meansvarhs[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()


## Frequency stimuli

s1mf = np.load('Samples1to4medfreq.npy')
s2mf = np.load('Samples5to8medfreq.npy')
s3mf= np.load('Samples9to12medfreq.npy')

s1hf = np.load('Samples1to4highfreq.npy')
s2hf = np.load('Samples5to8highfreq.npy')
s3hf= np.load('Samples9to12highfreq.npy')

smf = [s1mf,s2mf,s3mf]
shf = [s1hf,s2hf,s3hf]

meansAmf = []
stdsAmf = []
for i in range(3):
    for j in range(4):
        meansAmf_temp = []
        stdsAmf_temp = []
        for k in range(5):
            meansAmf_temp.append(np.mean(smf[i][j][k][300:].mean(0)[0]))
            stdsAmf_temp.append(np.sqrt(np.var(smf[i][j][k][300:].mean(0)[0])))
        stdsAmf.append(stdsAmf_temp)
        meansAmf.append(meansAmf_temp)
        
meansvarAmf = np.sqrt(np.asarray(meansAmf).var(0))     
meansAmf = np.asarray(meansAmf).mean(0)

stdsAmf = np.asarray(stdsAmf).mean(0)

meansAhf = []
stdsAhf = []
for i in range(3):
    for j in range(4):
        meansAhf_temp = []
        stdsAhf_temp = []
        for k in range(5):
            if i == 1 and j == 0:
                pass
            else:
                meansAhf_temp.append(np.mean(shf[i][j][k][300:].mean(0)[0]))
                stdsAhf_temp.append(np.sqrt(np.var(shf[i][j][k][300:].mean(0)[0])))
            if i != 1 and j != 0:
                stdsAhf.append(stdsAhf_temp)
                meansAhf.append(meansAhf_temp)
        
meansvarAhf = np.sqrt(np.asarray(meansAhf).var(0))     
meansAhf = np.asarray(meansAhf).mean(0)

stdsAhf = np.asarray(stdsAhf).mean(0)

meansTmf = []
stdsTmf = []
for i in range(3):
    for j in range(4):
        meansTmf_temp = []
        stdsTmf_temp = []
        for k in range(5):
            meansTmf_temp.append(np.mean(smf[i][j][k][300:].mean(0)[1]))
            stdsTmf_temp.append(np.sqrt(np.var(smf[i][j][k][300:].mean(0)[1])))
        stdsTmf.append(stdsTmf_temp)
        meansTmf.append(meansTmf_temp)
        
meansvarTmf = np.sqrt(np.asarray(meansTmf).var(0))     
meansTmf = np.asarray(meansTmf).mean(0)

stdsTmf = np.asarray(stdsTmf).mean(0)


meansThf = []
stdsThf = []
for i in range(3):
    for j in range(4):
        meansThf_temp = []
        stdsThf_temp = []
        for k in range(5):
            if i == 1 and j == 0:
                pass
            else:
                meansThf_temp.append(np.mean(shf[i][j][k][300:].mean(0)[1]))
                stdsThf_temp.append(np.sqrt(np.var(shf[i][j][k][300:].mean(0)[1])))
            if i != 1 and j != 0:
                stdsThf.append(stdsThf_temp)
                meansThf.append(meansThf_temp)
        
meansvarThf = np.sqrt(np.asarray(meansThf).var(0))     
meansThf = np.asarray(meansThf).mean(0)

stdsThf = np.asarray(stdsThf).mean(0)

x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $A$ - Med frequency stimuli')
plt.xlabel('Datasize')
plt.ylabel('$A$ estimation')
plt.ylim([0,0.02])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meansAmf[i], yerr = meansvarAmf[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $A$ - High frequency stimuli')
plt.xlabel('Datasize')
plt.ylabel('$A$ estimation')
plt.ylim([0,0.02])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meansAhf[i], yerr = meansvarAhf[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $Tau$ - Medium frequency stimuli')
plt.xlabel('Datasize')
plt.ylabel('$Tau$ estimation')
plt.ylim([0,0.1])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meansTmf[i], yerr = meansvarTmf[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $Tau$ - High frequency stimuli')
plt.xlabel('Datasize')
plt.ylabel('$Tau$ estimation')
plt.ylim([0,0.1])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meansThf[i], yerr = meansvarThf[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

'''
datamf = np.load('data1to4medfreq.npy')
datamf2 = np.load('data5to8medfreq.npy')
datamf3 = np.load('data9to12medfreq.npy')

datahf = np.load('data1to4highfreq.npy')


Traj = np.zeros(150000)
Traj[0] = 1
t = np.zeros(150000)
for i in range(1,150000):
    Traj[i] = Traj[i-1] + ut.learning_rule(datahf[0][0][:],datahf[0][1][:],0.005,0.005*1.05,0.02,0.02,t,i,0.002)
    t[i] = 0.002*i

plt.figure()
plt.plot(t,Traj)
plt.show()
'''