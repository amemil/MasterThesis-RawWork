#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:25:15 2021

@author: emilam
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


## w0 sensitivity ##

'''
Aps = np.load('Aps.npy',allow_pickle = True)
loglikes = np.load('ApLoglikes10sec.npy')
w0ests = np.load('w0estslong.npy')
ApPeaksInd = []
for i in range(200):
    ApPeaksInd.append(np.where(loglikes[i] == np.amax(loglikes[i])))


sns.set_style('darkgrid')
PeakAps = []
for i in range(200):
    PeakAps.append(Aps[ApPeaksInd[i][0].astype(int)][0])
PeakAps = np.asarray(PeakAps)

df = pd.DataFrame(data={'$w^0$ estimate': w0ests, '$A_+$: max $P(s_2^{(0:T)}$ given $A_+$)': np.asarray(PeakAps)})


sns.jointplot(
    data=df,
    x="$w^0$ estimate", y="$A_+$: max $P(s_2^{(0:T)}$ given $A_+$)",
    kind="kde",xlim=(0.4,1.6),color='b',marginal_kws={'lw':3, 'color':'red'})
#plt.plot(1,0.005,'ro',label='True Value')
plt.legend()
plt.show()

sns.jointplot(
    data=df,
    x="$w^0$ estimate", y="$A_+$: max $P(s_2^{(0:T)}$ given $A_+$)",
    kind="reg",xlim=(0.4,1.6),color='b',marginal_kws={'lw':3, 'color':'red'})
plt.plot(1,0.005,'ro',label='True Value')
plt.legend()
plt.show()
'''
## A inference different time domains! ##
'''
A = np.load('ASamples.npy')

secs = np.load('seconds.npy')
means = []
medians = []
stds = []
maps = np.load('MapsA.npy')

for i in range(5):
    means.append(np.mean(A[i,300:,:]))
    medians.append(np.median(A[i,300,:]))
    stds.append(np.sqrt(np.var(A[i,300:,:])))
    
x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $A_+$ - Means')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$A_+$ estimation')
plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], means[i], yerr = stds[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Inference of $A_+$ - Medians')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$A_+$ estimation')
plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], medians[i], yerr = stds[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Inference of $A_+$ - MAPs')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$A_+$ estimation')
plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], maps[i], yerr = stds[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''  

## Tau inference different time domains! ##
'''
Tau = np.load('TauSamples.npy')

secs = np.load('seconds.npy')
meanst = []
medianst = []
stdst = []
mapst = np.load('MapsTau')

for i in range(5):
    meanst.append(np.mean(Tau[i,300:,:]))
    medianst.append(np.median(Tau[i,300,:]))
    stdst.append(np.sqrt(np.var(Tau[i,300:,:])))
    
x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Inference of $Tau$ - Means')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$Tau$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meanst[i], yerr = stdst[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Inference of $Tau$ - Medians')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$Tau$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], medianst[i], yerr = stdst[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Inference of $Tau$ - MAPs')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$Tau$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], mapst[i], yerr = stdst[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''
## Simultaneous inference different time domains ## 
'''
Sim = np.load('SimSamples.npy')

secs = np.load('seconds.npy')
meansAsim = []
mediansAsim = []
stdsAsim = []
mapsAsim = np.load('MapsASim.npy')

meansTausim = []
mediansTausim = []
stdsTausim = []
mapsTausim = np.load('MapsTauSim.npy')

for i in range(5):
    meansAsim.append(np.mean(Sim[i,300:,0]))
    meansTausim.append(np.mean(Sim[i,300:,1]))
    mediansAsim.append(np.median(Sim[i,300:,0]))
    mediansTausim.append(np.median(Sim[i,300:,1]))
    stdsAsim.append(np.sqrt(np.var(Sim[i,300:,0])))
    stdsTausim.append(np.sqrt(np.var(Sim[i,300:,1])))
    
    
    
x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Standard MH : Inference of $Tau$ - Means')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$Tau$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meansTausim[i], yerr = stdsTausim[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Standard MH : Inference of $Tau$ - Medians')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$Tau$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], mediansTausim[i], yerr = stdsTausim[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Standard MH : Inference of $Tau$ - MAPs')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$Tau$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], mapsTausim[i], yerr = stdsTausim[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Standard MH : Inference of $A_+$ - Means')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$A_+$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meansAsim[i], yerr = stdsAsim[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Standard MH : Inference of $A_+$ - Medians')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$A_+$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], mediansAsim[i], yerr = stdsAsim[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Standard MH : Inference of $A_+$ - MAPs')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$A_+$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], mapsAsim[i], yerr = stdsAsim[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
'''

## Simultaneous inference different time domains ## 

Alt = np.load('AltSamples.npy')

secs = np.load('seconds.npy')
meansAalt = []
mediansAalt = []
stdsAalt = []
mapsAalt = np.load('MapsASim.npy')

meansTaualt = []
mediansTaualt = []
stdsTaualt = []
mapsTaualt = np.load('MapsTauSim.npy')

for i in range(5):
    meansAalt.append(np.mean(Alt[i,300:,0]))
    meansTaualt.append(np.mean(Alt[i,300:,1]))
    mediansAalt.append(np.median(Alt[i,300:,0]))
    mediansTaualt.append(np.median(Alt[i,300:,1]))
    stdsAalt.append(np.sqrt(np.var(Alt[i,300:,0])))
    stdsTaualt.append(np.sqrt(np.var(Alt[i,300:,1])))
    
    
    
x = [1,2,3,4,5]
ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Alternating MH : Inference of $Tau$ - Means')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$Tau$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meansTaualt[i], yerr = stdsTaualt[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Alternating MH : Inference of $Tau$ - Medians')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$Tau$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], mediansTaualt[i], yerr = stdsTaualt[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Alternating MH : Inference of $Tau$ - MAPs')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$Tau$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], mapsTaualt[i], yerr = stdsTaualt[i],marker = 'o')
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Alternating MH : Inference of $A_+$ - Means')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$A_+$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], meansAalt[i], yerr = stdsAalt[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Alternating MH : Inference of $A_+$ - Medians')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$A_+$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], mediansAalt[i], yerr = stdsAalt[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

plt.figure()
plt.title('Alternating MH : Inference of $A_+$ - MAPs')
plt.xlabel('Domain size (seconds)')
plt.ylabel('$A_+$ estimation')
#plt.ylim([0,0.025])
plt.xlim([0,6])
plt.xticks(x,labels = ticksss)
for i in range(5):
    plt.errorbar(x[i], mapsAalt[i], yerr = stdsAalt[i],marker = 'o')
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

