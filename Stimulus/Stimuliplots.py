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

plt.style.use('default')

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


'''
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

### SIM 

## DETERMINISTIC STIMULI 
'''

Sim1ms = np.load('SimSamples1to4medstim.npy')
Sim2ms = np.load('SimSamples5to8medstim.npy')
#Sim3ms =np.load('SimSamples9to12medstim.npy')
Sim4ms =np.load('SimSamples13to16medstim.npy')
Sim5ms =np.load('SimSamples17to20medstim.npy')

sims = [Sim1ms,Sim2ms,Sim4ms,Sim5ms]

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

meanssA.pop(4)
meanssA.pop(7)
meanssA.pop(8)
meanssA.pop(11)
meanssA.pop(11)

stdssA.pop(4)
stdssA.pop(7)
stdssA.pop(8)
stdssA.pop(11)
stdssA.pop(11)

meanssT.pop(8)
meanssT.pop(9)
meanssT.pop(12)
meanssT.pop(12)

stdssT.pop(8)
stdssT.pop(9)
stdssT.pop(12)
stdssT.pop(12)

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
### TETANIC STIMULI

## 20 HZ
'''
Sim120hz = np.load('Samples1to4medfreq.npy')
Sim220hz = np.load('Samples5to8medfreq.npy')
Sim320hz =np.load('Samples9to12medfreq.npy')
#Sim420hz =np.load('SimSamples13to16medfreq.npy')
#Sim520hz =np.load('SimSamples17to20medfreq.npy')

sims = [Sim120hz,Sim220hz,Sim320hz]#,Sim5ms]

meanssA = []
meanssT = []

stdssA = []
stdssT = []

for i in range(3):
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
        
meanssA.pop(6)
meanssT.pop(6)
stdssA.pop(6)
stdssT.pop(6)

meanssA = np.mean(meanssA,axis=0)
meanssT = np.mean(meanssT,axis=0)
stdssA = np.mean(stdssA,axis=0)
stdssT = np.mean(stdssT,axis=0)

x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $A$ - 20Hz additional stimulus')
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
plt.title('Inference of $Tau$ - 20Hz additional stimulus')
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
## 100hz
'''
Sim1_100hz = np.load('Samples1to4highfreq.npy')
Sim2_100hz = np.load('Samples5to8highfreq.npy')
Sim3_100hz =np.load('Samples9to12highfreq.npy')
Sim4_100hz =np.load('Samples13to16highfreq.npy')
#Sim5_100hz =np.load('Samples17to20highfreq.npy')

sims = [Sim1_100hz,Sim2_100hz,Sim3_100hz,Sim4_100hz]#,Sim5ms]

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

meanssA.pop(1)
meanssA.pop(2)
meanssA.pop(2)
meanssA.pop(8)
meanssA.pop(8)
meanssA.pop(8)


stdssA.pop(1)
stdssA.pop(2)
stdssA.pop(2)
stdssA.pop(8)
stdssA.pop(8)
stdssA.pop(8)

meanssT.pop(1)
meanssT.pop(2)
meanssT.pop(2)
meanssT.pop(8)
meanssT.pop(8)
meanssT.pop(8)

stdssT.pop(1)
stdssT.pop(2)
stdssT.pop(2)
stdssT.pop(8)
stdssT.pop(8)
stdssT.pop(8)

rmselr100 = []
datasizes = []
for i in range(10):
    for j in range(5):
        lrs1_temp = lr1(1,1,meanssA[i][j],deltas,meanssT[i][j])
        lrs2_temp = lr2(1,1,meanssA[i][j],deltas2,meanssT[i][j])
        lr_est = np.concatenate((lrs2_temp,lrs1_temp))
        rmselr100.append(rmse(lrref, lr_est))
        datasizes.append((j+1)*60)
'''     

#meanssA = np.mean(meanssA,axis=0)
#meanssT = np.mean(meanssT,axis=0)
#stdssA = np.mean(stdssA,axis=0)
#stdssT = np.mean(stdssT,axis=0)
'''
x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $A$ - 100Hz additional stimulus')
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
plt.title('Inference of $Tau$ - 100Hz additional stimulus')
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
### 200hz
'''
Sim1_200hz = np.load('Samples1to4megahighfreq.npy')
Sim2_200hz = np.load('Samples5to8megahighfreq.npy')
Sim3_200hz =np.load('Samples9to12megahighfreq.npy')
Sim4_200hz =np.load('Samples13to16megahighfreq.npy')
Sim5_200hz =np.load('Samples17to20megahighfreq.npy')

sims = [Sim1_200hz,Sim2_200hz,Sim3_200hz,Sim4_200hz,Sim5_200hz]

meanssA = []
meanssT = []

stdssA = []
stdssT = []

for i in range(5):
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

rmselr200 = []
datasizes2 = []
for i in range(20):
    for j in range(5):
        lrs1_temp = lr1(1,1,meanssA[i][j],deltas,meanssT[i][j])
        lrs2_temp = lr2(1,1,meanssA[i][j],deltas2,meanssT[i][j])
        lr_est = np.concatenate((lrs2_temp,lrs1_temp))
        rmselr200.append(rmse(lrref, lr_est))
        datasizes2.append((j+1)*60)
'''



'''
meanssA = np.mean(meanssA,axis=0)
meanssT = np.mean(meanssT,axis=0)
stdssA = np.mean(stdssA,axis=0)
stdssT = np.mean(stdssT,axis=0)

x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $A$ - 200Hz additional stimulus')
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
plt.title('Inference of $Tau$ - 200Hz additional stimulus')
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

rmse100 = np.load('rmselr100hz.npy')
rmse200 = np.load('rmselr250hz.npy')
rmsebase = np.load('rmselrbase.npy')

datasizes100 = np.load('DatasizeLabels100hz.npy')
datasizes200 = np.load('DatasizeLabels250hz.npy')
datasizesbase = np.load('DatasizeLabelsBase.npy')

rmses = np.hstack((rmse100,rmse200,rmsebase))
datasizes = np.hstack((datasizes100,datasizes200,datasizesbase))

stimulus100 = []
stimulus200 = []
stimulusbase = []
for i in range(len(rmse100)):
    stimulus100.append(100)
for i in range(len(rmse200)):
    stimulus200.append(200)
for i in range(len(rmsebase)):
    stimulusbase.append(0)
    
stimulus = np.hstack((stimulus100,stimulus200,stimulusbase))

data = np.transpose(np.asarray([rmses,datasizes,stimulus]))
df = pd.DataFrame(data, columns =['RMSE', 'Datasize (sec)','Stimulus'])
#df = df.pivot("Datasize (sec)", "Stimulus", "RMSE")

ax = sns.lineplot(data=df, x="Datasize (sec)", y="RMSE",hue="Stimulus",palette=['orangered','chartreuse','royalblue'])
ax.legend(['Baseline firing', '100Hz', '250Hz'])

