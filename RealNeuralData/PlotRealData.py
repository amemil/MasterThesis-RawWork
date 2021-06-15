#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:06:50 2021

@author: emilam
"""

import sys, os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma
import matplotlib
import pandas as pd
#plt.style.use('seaborn-darkgrid')
plt.style.use('default')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


#### Sample 2_1,2_2 correspond to maps 3_1,3_2. Same with 3_1 -> 4_1 and 4_1 > 5_1. Map alltid i+1 v sample :D:D (messed up)
#StimSample = np.load('StimCand1Sample.npy')
#NonStimSample = np.load('NonstimCand1Sample.npy')

#StimSample = np.load('SampleStim20sec_cand1.npy')
#NonStimSample = np.load('SampleNonstim20sec_cand1.npy')

StimSample = np.load('SampleStimCand2_1.npy')
NonStimSample = np.load('SampleNonStimCand2_1.npy')


piAstim = [np.sort(StimSample[300:,0])[30],np.sort(StimSample[300:,0])[-31]]
piTaustim = [np.sort(StimSample[300:,1])[30],np.sort(StimSample[300:,1])[-31]]
piTaunonstim = [np.sort(NonStimSample[300:,1])[30],np.sort(NonStimSample[300:,1])[-31]]
piAnonstim = [np.sort(NonStimSample[300:,0])[30],np.sort(NonStimSample[300:,0])[-31]]

medAstim = np.median(StimSample[300:,0])
medTaustim = np.median(StimSample[300:,1])
medAnonstim = np.median(NonStimSample[300:,0])
medTaunonstim = np.median(NonStimSample[300:,1])

meanAstim = np.mean(StimSample[300:,0])
meanTaustim = np.mean(StimSample[300:,1])
meanAnonstim = np.mean(NonStimSample[300:,0])
meanTaunonstim = np.mean(NonStimSample[300:,1])

mapAstim = np.load('MapAStimCand3_1.npy')
mapTaustim =  np.load('MapTauStimCand3_1.npy')
mapAnonstim =  np.load('MapANonStimCand3_1.npy')
mapTaunonstim =  np.load('MapTauNonStimCand3_1.npy')

entAstim = norm.entropy(loc = meanAstim,scale = np.sqrt(np.var(StimSample[300:,0])))
entAnonstim = norm.entropy(loc = meanAnonstim,scale = np.sqrt(np.var(NonStimSample[300:,0])))
entTaustim = norm.entropy(loc = meanTaustim,scale = np.sqrt(np.var(StimSample[300:,1])))
entTaunonstim = norm.entropy(loc = meanTaunonstim,scale = np.sqrt(np.var(NonStimSample[300:,1])))

x = np.linspace(0,0.2,10000)
priorA = gamma.pdf(x,a=4,scale=1/50)
priorT = gamma.pdf(x,a=5,scale = 1/100)
matplotlib.rcParams.update({'font.size': 15})

plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams.update({'font.size': 15}) 
plt.rc('axes', labelsize=20) 
plt.rc('figure',titlesize=20)
def lr1(s2,s1,Ap,delta,taup):
    return s2*s1*Ap*np.exp(-delta/taup)

def lr2(s1,s2,Am,delta,taum):
    return -s1*s2*Am*1.05*np.exp(delta/taum)





'''
plt.figure()
sns.distplot(StimSample[300:,0],kde = True,color='g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
plt.xlim([0.00,0.2])
plt.xticks([0,0.05,0.1,0.15],labels=['0','0.05','0.1','0.15'])
#plt.ylabel('')
plt.xlabel(r'$A_+$')
plt.axvline(meanAstim,color='k',linestyle='-',label='Sample mean')
plt.plot(x,priorA,color='#AAA662',alpha=0.9,label='Prior')
#plt.axvline(medAstim,color='g',linestyle='--',label='Median: '+str(medAstim.round(3)))
#plt.axvline(meanAstim,color='m',linestyle='--',label='Mean: '+str(meanAstim.round(3)))
plt.axvline(piAstim[0],color='k',linestyle='--',label='95% CI')
plt.ylabel('')
plt.axvline(piAstim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
#plt.title(r'Stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
#plt.legend()

plt.figure()
sns.distplot(NonStimSample[300:,0],kde = True,color = 'g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
plt.xlim([0.00,0.2])
plt.ylabel('')
plt.xlabel(r'$A_+$')
#plt.xticks([0,0.05,0.1,0.15,0.2],labels=['0','0.05','0.1','0.15','0.2'])
plt.xticks([0,0.05,0.1,0.15],labels=['0','0.05','0.1','0.15'])
plt.yticks([0,40,80,120],labels=['0','40','80','120'])
plt.axvline(meanAnonstim,color='k',linestyle='-',label='Sample mean')#+str(mapAnonstim[0].round(3)))
plt.plot(x,priorA,color='#AAA662',alpha=0.9,label='Prior')
#plt.yticks([0,20,40,80,120],labels=['0','20','40','80','120'])
#plt.axvline(medAnonstim,color='g',linestyle='--',label='Median: '+str(medAnonstim.round(3)))
#plt.axvline(meanAnonstim,color='m',linestyle='--',label='Mean: '+str(meanAnonstim.round(3)))
plt.axvline(piAnonstim[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piAnonstim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
#plt.title(r'No stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
#plt.legend()
#plt.show
'''

plt.figure()
sns.distplot(StimSample[300:,1],kde = True,color='g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
plt.xlim([0.00,0.16])
plt.xticks([0,0.05,0.1,0.15],labels=['0','0.05','0.1','0.15'])
plt.yticks([0,10,20],labels=['0','10','20'])
#plt.ylabel('')
plt.xlabel(r'Tau')
plt.axvline(meanTaustim,color='k',linestyle='-',label='Sample mean')
plt.plot(x,priorT,color='#AAA662',alpha=0.9,label='Prior')
#plt.axvline(medAstim,color='g',linestyle='--',label='Median: '+str(medAstim.round(3)))
#plt.axvline(meanAstim,color='m',linestyle='--',label='Mean: '+str(meanAstim.round(3)))
plt.axvline(piTaustim[0],color='k',linestyle='--',label='95% CI')
plt.ylabel('')
plt.axvline(piTaustim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
#plt.title(r'Stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
#plt.legend()

plt.figure()
sns.distplot(NonStimSample[300:,1],kde = True,color = 'g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
plt.xlim([0.00,0.16])
plt.xticks([0,0.05,0.1,0.15],labels=['0','0.05','0.1','0.15'])
plt.yticks([0,10,20,30],labels=['0','10','20','30'])
plt.ylabel('')
plt.xlabel(r'Tau')
#plt.xticks([0,0.05,0.1,0.15,0.2],labels=['0','0.05','0.1','0.15','0.2'])
#plt.xticks([0,0.05,0.1,0.15],labels=['0','0.05','0.1','0.15'])
#plt.yticks([0,40,80,120],labels=['0','40','80','120'])
plt.axvline(meanTaunonstim,color='k',linestyle='-',label='Sample mean')#+str(mapAnonstim[0].round(3)))
plt.plot(x,priorT,color='#AAA662',alpha=0.9,label='Prior')
#plt.yticks([0,20,40,80,120],labels=['0','20','40','80','120'])
#plt.axvline(medAnonstim,color='g',linestyle='--',label='Median: '+str(medAnonstim.round(3)))
#plt.axvline(meanAnonstim,color='m',linestyle='--',label='Mean: '+str(meanAnonstim.round(3)))
plt.axvline(piTaunonstim[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piTaunonstim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
#plt.title(r'No stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
#plt.legend()
#plt.show

#StimSample2 = np.load('SampleStim20sec_cand2.npy')
#NonStimSample2 = np.load('SampleNonstim20sec_cand2.npy')
StimSample2 = np.load('SampleStimCand2_2.npy')
NonStimSample2 = np.load('SampleNonStimCand2_2.npy')

piAstim2 = [np.sort(StimSample2[300:,0])[30],np.sort(StimSample2[300:,0])[-31]]
piTaustim2 = [np.sort(StimSample2[300:,1])[30],np.sort(StimSample2[300:,1])[-31]]
piTaunonstim2 = [np.sort(NonStimSample2[300:,1])[30],np.sort(NonStimSample2[300:,1])[-31]]
piAnonstim2 = [np.sort(NonStimSample2[300:,0])[30],np.sort(NonStimSample2[300:,0])[-31]]

medAstim = np.median(StimSample2[300:,0])
medTaustim = np.median(StimSample2[300:,1])
medAnonstim = np.median(NonStimSample2[300:,0])
medTaunonstim = np.median(NonStimSample2[300:,1])

meanAstim2 = np.mean(StimSample2[300:,0])
meanTaustim2 = np.mean(StimSample2[300:,1])
meanAnonstim2 = np.mean(NonStimSample2[300:,0])
meanTaunonstim2 = np.mean(NonStimSample2[300:,1])

mapAstim2= np.load('MapAStimCand3_2.npy')
mapTaustim2 =  np.load('MapTauStimCand3_2.npy')
mapAnonstim2 =  np.load('MapANonStimCand3_2.npy')
mapTaunonstim2 =  np.load('MapTauNonStimCand3_2.npy')

entAstim2 = norm.entropy(loc = meanAstim2,scale = np.sqrt(np.var(StimSample2[300:,0])))
entAnonstim = norm.entropy(loc = meanAnonstim2,scale = np.sqrt(np.var(NonStimSample2[300:,0])))
entTaustim = norm.entropy(loc = meanTaustim2,scale = np.sqrt(np.var(StimSample2[300:,1])))
entTaunonstim = norm.entropy(loc = meanTaunonstim2,scale = np.sqrt(np.var(NonStimSample2[300:,1])))

deltas = np.linspace(0,0.1,10000)
deltas2 = np.linspace(-0.1,0,10000)   
lrs1 = lr1(1,1,meanAstim,deltas,meanTaustim)
lrs2 = lr2(1,1,meanAstim,deltas2,meanTaustim) 
'''
plt.rcParams["font.family"] = "Times New Roman"
deltass = np.concatenate((deltas2,deltas))
lrref = np.concatenate((lrs2,lrs1))

lrs3 = lr1(1,1,meanAstim2,deltas,meanTaustim2)
lrs4 = lr2(1,1,meanAstim2,deltas2,meanTaustim2) 
lrref2 = np.concatenate((lrs4,lrs3))
matplotlib.rcParams.update({'font.size': 18})

plt.rc('axes', labelsize=24) 
#plt.rc('figure',titlesize=30)
plt.figure()
#plt.title('Learning rules')
plt.xlabel(r'$\Delta$ t [sec]')
plt.ylabel(r'$\Delta$ w')
plt.axhline(y=0,color = 'k',linestyle='--')
plt.plot(deltass,lrref,c='rebeccapurple',linewidth=3)#,label=r'$\Theta_1$')#,label=r'$\Theta$ = [0.0075,0.0075,0.02,0.02]')
plt.plot(deltass,lrref2,c='chocolate',linewidth=3,linestyle= '--')#,label=r'$\Theta_2$')#,label=r'$\Theta$ = [0.005,0.005,0.04,0.04]')
#plt.yticks([-0.008,-0.004,0,0.004,0.008],labels=['-0.008','-0.004','0','0.004','0.008'])
#plt.xticks([-0.1,-0.05,0,0.05,0.1],labels=['-0.1','-0.05','0','0.05','0.1'])
plt.legend()
plt.show()

deltas = np.linspace(0,0.1,10000)
deltas2 = np.linspace(-0.1,0,10000)   
lrs1 = lr1(1,1,meanAnonstim,deltas,meanTaunonstim)
lrs2 = lr2(1,1,meanAnonstim,deltas2,meanTaunonstim) 

plt.rcParams["font.family"] = "Times New Roman"
deltass = np.concatenate((deltas2,deltas))
lrref = np.concatenate((lrs2,lrs1))

lrs3 = lr1(1,1,meanAnonstim2,deltas,meanTaunonstim2)
lrs4 = lr2(1,1,meanAnonstim2,deltas2,meanTaunonstim2) 
lrref2 = np.concatenate((lrs4,lrs3))
matplotlib.rcParams.update({'font.size': 18})

plt.rc('axes', labelsize=24) 
#plt.rc('figure',titlesize=30)
plt.figure()
#plt.title('Learning rules')
plt.xlabel(r'$\Delta$ t [sec]')
plt.ylabel(r'$\Delta$ w')
plt.axhline(y=0,color = 'k',linestyle='--')
plt.plot(deltass,lrref,c='rebeccapurple',linewidth=3)#,label=r'$\Theta_1$')#,label=r'$\Theta$ = [0.0075,0.0075,0.02,0.02]')
plt.plot(deltass,lrref2,c='chocolate',linewidth=3,linestyle= '--')#,label=r'$\Theta_2$')#,label=r'$\Theta$ = [0.005,0.005,0.04,0.04]')
#plt.yticks([-0.008,-0.004,0,0.004,0.008],labels=['-0.008','-0.004','0','0.004','0.008'])
#plt.xticks([-0.1,-0.05,0,0.05,0.1],labels=['-0.1','-0.05','0','0.05','0.1'])
plt.legend()
plt.show()
'''
'''
plt.figure()
sns.distplot(StimSample[300:,0],kde = True,color='g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
plt.xlim([0.00,0.2])
plt.xlabel(r'$A_+$')
plt.axvline(meanAstim2,color='k',linestyle='-',label='Sample mean')
#plt.xticks([0,0.05,0.1,0.15,0.2],labels=['0','0.05','0.1','0.15','0.2'])
plt.ylabel('')
plt.yticks([0,100,200,300],labels=['0','100','200','300'])
plt.plot(x,priorA,color='#AAA662',alpha=0.9,label='Prior')
plt.xticks([0,0.05,0.1,0.15],labels=['0','0.05','0.1','0.15'])
#plt.yticks([0,100,200,300],labels=['0','100','200','300'])
#plt.axvline(medAstim,color='g',linestyle='--',label='Median: '+str(medAstim.round(3)))
#plt.axvline(meanAstim,color='m',linestyle='--',label='Mean: '+str(meanAstim.round(3)))
plt.axvline(piAstim[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piAstim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
#plt.title(r'Stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()

plt.figure()
sns.distplot(NonStimSample[300:,0],kde = True,color = 'g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
plt.xlim([0.00,0.2])
#plt.xticks([0,0.05,0.1,0.15,0.2],labels=['0','0.05','0.1','0.15','0.2'])
plt.xticks([0,0.05,0.1,0.15],labels=['0','0.05','0.1','0.15'])
plt.xlabel(r'$A_+$')
plt.ylabel('')
plt.axvline(meanAnonstim2,color='k',linestyle='-',label='Sample mean')#+str(mapAnonstim[0].round(3)))
plt.plot(x,priorA,color='#AAA662',alpha=0.9,label='Prior')
#plt.yticks([0,10,20,30,40],labels=['0','10','20','30','40'])
#plt.axvline(medAnonstim,color='g',linestyle='--',label='Median: '+str(medAnonstim.round(3)))
#plt.axvline(meanAnonstim,color='m',linestyle='--',label='Mean: '+str(meanAnonstim.round(3)))
plt.axvline(piAnonstim[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piAnonstim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
#plt.title(r'No stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
#plt.legend()
'''

plt.figure()
sns.distplot(StimSample2[300:,1],kde = True,color='g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
plt.xlim([0.00,0.16])
plt.xticks([0,0.05,0.1,0.15],labels=['0','0.05','0.1','0.15'])
plt.yticks([0,10,20,30],labels=['0','10','20','30'])
plt.xlabel(r'Tau')
plt.axvline(meanTaustim2,color='k',linestyle='-',label='Sample mean')
#plt.xticks([0,0.05,0.1,0.15,0.2],labels=['0','0.05','0.1','0.15','0.2'])
plt.ylabel('')
#plt.yticks([0,100,200,300],labels=['0','100','200','300'])
plt.plot(x,priorT,color='#AAA662',alpha=0.9,label='Prior')
#plt.xticks([0,0.05,0.1,0.15],labels=['0','0.05','0.1','0.15'])
#plt.yticks([0,100,200,300],labels=['0','100','200','300'])
#plt.axvline(medAstim,color='g',linestyle='--',label='Median: '+str(medAstim.round(3)))
#plt.axvline(meanAstim,color='m',linestyle='--',label='Mean: '+str(meanAstim.round(3)))
plt.axvline(piTaustim2[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piTaustim2[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
#plt.title(r'Stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()

plt.figure()
sns.distplot(NonStimSample2[300:,1],kde = True,color = 'g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
plt.xlim([0.00,0.16])
plt.xticks([0,0.05,0.1,0.15],labels=['0','0.05','0.1','0.15'])
plt.yticks([0,20,40,60],labels=['0','20','40','60'])
#plt.xticks([0,0.05,0.1,0.15,0.2],labels=['0','0.05','0.1','0.15','0.2'])
#plt.xticks([0,0.05,0.1,0.15],labels=['0','0.05','0.1','0.15'])
plt.xlabel(r'Tau')
plt.ylabel('')
plt.axvline(meanTaunonstim2,color='k',linestyle='-',label='Sample mean')#+str(mapAnonstim[0].round(3)))
plt.plot(x,priorT,color='#AAA662',alpha=0.9,label='Prior')
#plt.yticks([0,10,20,30,40],labels=['0','10','20','30','40'])
#plt.axvline(medAnonstim,color='g',linestyle='--',label='Median: '+str(medAnonstim.round(3)))
#plt.axvline(meanAnonstim,color='m',linestyle='--',label='Mean: '+str(meanAnonstim.round(3)))
plt.axvline(piTaunonstim2[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piTaunonstim2[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
#plt.title(r'No stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
#plt.legend()



'''
SubStim = np.load('SubsamplesStim1.npy')
SubNonStim = np.load('SubsamplesNonstim1.npy')

datasizes = [20,40,60,80,100]
entropiesAStim = []
entropiesANonStim = []
entropiesTauStim = []
entropiesTauNonStim = []


for i in range(4):
    entropiesAStim.append(norm.entropy(loc = np.mean(SubStim[i,300:,0]),scale = np.sqrt(np.var(SubStim[i,300:,0]))))
    entropiesTauStim.append(norm.entropy(loc = np.mean(SubStim[i,300:,1]),scale = np.sqrt(np.var(SubStim[i,300:,1]))))
    entropiesANonStim.append(norm.entropy(loc = np.mean(SubNonStim[i,300:,0]),scale = np.sqrt(np.var(SubNonStim[i,300:,0]))))
    entropiesTauNonStim.append(norm.entropy(loc = np.mean(SubNonStim[i,300:,1]),scale = np.sqrt(np.var(SubNonStim[i,300:,1]))))

entropiesAStim.append(norm.entropy(loc = np.mean(StimSample[300:,0]),scale = np.sqrt(np.var(StimSample[300:,0]))))
entropiesTauStim.append(norm.entropy(loc = np.mean(StimSample[300:,1]),scale = np.sqrt(np.var(StimSample[300:,1]))))
entropiesANonStim.append(norm.entropy(loc = np.mean(NonStimSample[300:,0]),scale = np.sqrt(np.var(NonStimSample[300:,0]))))
entropiesTauNonStim.append(norm.entropy(loc = np.mean(NonStimSample[300:,1]),scale = np.sqrt(np.var(NonStimSample[300:,1]))))

plt.figure()
plt.title('A posterior entropies')
plt.xlabel('Datasize (sec)')
plt.ylabel('Entropy')
plt.plot(datasizes,entropiesAStim,'rx-',alpha = 0.7,label='Stimulated data')
plt.plot(datasizes,entropiesANonStim,'gx-',alpha = 0.7,label='Non-stimulated data')
plt.legend()
plt.show()

plt.figure()
plt.title('Tau posterior entropies')
plt.xlabel('Datasize (sec)')
plt.ylabel('Entropy')
plt.plot(datasizes,entropiesTauStim,'rx-',alpha = 0.7,label='Stimulated data')
plt.plot(datasizes,entropiesTauNonStim,'gx-',alpha = 0.7,label='Non-stimulated data')
plt.legend()
plt.show()
'''

#LARGESCALE!
'''
Stim1 = np.load('LargeScaleInvStim_1.npy')
Stim2 = np.load('LargeScaleInvStim_2.npy')
Stim3 = np.load('LargeScaleInvStim_3.npy')
Stim4 = np.load('LargeScaleInvStim_4.npy')
Stim5 = np.load('LargeScaleInvStim_5.npy')
Stim6 = np.load('LargeScaleInvStim_6.npy')
Stim7 = np.load('LargeScaleInvStim_7.npy')
Stim8 = np.load('LargeScaleInvStim_8.npy')
Stim9 = np.load('LargeScaleInvStim_9.npy')
Stim10 = np.load('LargeScaleInvStim_10.npy')

Stims=[Stim1,Stim2,Stim3,Stim4,Stim5,Stim6,Stim7,Stim8,Stim9,Stim10]


NonStim1 = np.load('LargeScaleInvNonStim_1.npy')
NonStim2 = np.load('LargeScaleInvInvNonStim_2.npy')
NonStim3 = np.load('LargeScaleInvNonStim_3.npy')
NonStim4 = np.load('LargeScaleInvNonStim_4.npy')
NonStim5 = np.load('LargeScaleInvNonStim_5.npy')
NonStim6 = np.load('LargeScaleInvNonStim_6.npy')
NonStim7 = np.load('LargeScaleInvNonStim_7.npy')
NonStim8 = np.load('LargeScaleInvNonStim_8.npy')
NonStim9 = np.load('LargeScaleInvNonStim_9.npy')
NonStim10 = np.load('LargeScaleInvNonStim_10.npy')

NonStims=[NonStim1,NonStim2,NonStim3,NonStim4,NonStim5,NonStim6,NonStim7,NonStim8,NonStim9,NonStim10]


def NormEntropy(sigma):
        return 0.5 * np.log(np.linalg.det(2*np.pi*np.exp(1)*sigma))
    
#[10,20,30,40,50,60]
  
Entropies = []
Datasize = []
Label = []
for i in range(len(Stims)):
    for j in range(6):
        entropy = NormEntropy(np.cov(np.transpose(Stims[i][j][300:,:])))
        if entropy> -15:
            Entropies.append(entropy)
            Datasize.append(10+(10*j))
            #print('STIM - Datasize: '+str(20+(20*j))+', entropy: '+str(entropy))
            Label.append(1)
    
for i in range(len(NonStims)):
    for j in range(6):
        entropy = NormEntropy(np.cov(np.transpose(NonStims[i][j][300:,:])))
        if entropy >-15:
            Entropies.append(entropy)
            Datasize.append(10+(10*j))
            #print('NONSTIM - Datasize: '+str(20+(20*j))+', entropy: '+str(entropy))
            Label.append(2)

data = np.transpose(np.asarray([Entropies,Datasize,Label]))
df = pd.DataFrame(data, columns =['Entropy', 'Datasize','Label'])
ax = sns.lineplot(data=df, x="Datasize", y="Entropy",hue="Label",palette=['orangered','royalblue'])#,'royalblue'])
ax.legend(['Stimulated','Non Stimulated'])#,'Randomised Frequency','Optimal [10-100hz] grid','Dales Law'])


plt.figure()
plt.xlabel('Datasize')
plt.ylabel('Entropy')
plt.title('Sample entropies real data')
for i in range(len(Entropies)):
    if Label[i] == 1:
        if i == 0:
            plt.plot([Datasize[i]],Entropies[i],'rx',label='Stimulated')
        else:
            plt.plot([Datasize[i]],Entropies[i],'rx')
    else:
        if i == len(Entropies)-3:
            plt.plot([Datasize[i]],Entropies[i],'bx',label = 'Non stimulated')
        else:
            plt.plot([Datasize[i]],Entropies[i],'bx')
plt.legend()
plt.show()
'''
'''

plt.figure()
plt.title('Single experiments stimulation')
plt.xlabel('Datasize')
plt.ylabel('Entropy')
for i in range(0,39,4):
    if min(Entropies[i:i+4])>-14:
        plt.plot([20,40,60,80],Entropies[i:i+4],'r-')
plt.show()

plt.figure()
plt.title('Single experiments non-stimulation')
plt.xlabel('Datasize')
plt.ylabel('Entropy')
for i in range(40,79,4):
    if min(Entropies[i:i+4])>-14:
        plt.plot([20,40,60,80],Entropies[i:i+4],'b-')
plt.show()


## STIM 20secs ###

s20_1 = np.load('StimSamples20sec1to10.npy')
s20_2 = np.load('StimSamples20sec11to20.npy')
s20_3 = np.load('StimSamples20sec21to31.npy')
s20_4 = np.load('StimSamples20sec31to41.npy')

## NONSTIM 20sec ###

ns20_1 = np.load('NonStimSamples20sec1to10.npy')
ns20_2 = np.load('NonStimSamples20sec11to21.npy')
ns20_3 = np.load('NonStimSamples20sec21to31.npy')
ns20_4 = np.load('NonStimSamples20sec31to41.npy')

entropies_stim = []
entropies_nonstim = []

for i in range(10):
    entropy = NormEntropy(np.cov(np.transpose(s20_1[i][300:,:])))
    entropies_stim.append(entropy)
for i in range(10):
    entropy = NormEntropy(np.cov(np.transpose(s20_2[i][300:,:])))
    entropies_stim.append(entropy)
for i in range(10):
    entropy = NormEntropy(np.cov(np.transpose(s20_3[i][300:,:])))
    entropies_stim.append(entropy)
for i in range(10):
    entropy = NormEntropy(np.cov(np.transpose(s20_4[i][300:,:])))
    entropies_stim.append(entropy)
for i in range(10):
    entropy = NormEntropy(np.cov(np.transpose(ns20_1[i][300:,:])))
    if entropy > -8:
        entropies_nonstim.append(entropy)
for i in range(10):
    entropy = NormEntropy(np.cov(np.transpose(ns20_2[i][300:,:])))
    if entropy > -8:
        entropies_nonstim.append(entropy)
for i in range(10):
    entropy = NormEntropy(np.cov(np.transpose(ns20_3[i][300:,:])))
    if entropy > -8:
        entropies_nonstim.append(entropy)
for i in range(10):
    entropy = NormEntropy(np.cov(np.transpose(ns20_4[i][300:,:])))
    if entropy > -8:
        entropies_nonstim.append(entropy)
    

plt.figure()
sns.histplot(data=entropies_stim)
plt.show()   

plt.figure()
sns.histplot(data=entropies_nonstim)
plt.show()

labels = []
for i in range(len(entropies_stim)):
    labels.append(0)
for i in range(len(entropies_nonstim)):
    labels.append(1)
matplotlib.rcParams.update({'font.size': 13})
plt.rc('axes', labelsize=15)  
ents = np.hstack((entropies_stim,entropies_nonstim))
data = np.transpose(np.asarray([ents,labels]))
df = pd.DataFrame(data, columns =['Entropy', 'Label'])
ax = sns.histplot(data=df, x="Entropy", hue="Label",kde=True,edgecolor = 'w')
ax.legend(['Non-stimulated','Stimulated'])

plt.figure()
plt.title('Chronological entropy evolution')
plt.ylabel('Entropy')
plt.xlabel('Sample')
plt.plot(np.linspace(1,len(entropies_stim),len(entropies_stim)),entropies_stim,label='Stimulated')
plt.plot(np.linspace(1,len(entropies_nonstim),len(entropies_nonstim)),entropies_nonstim,label='Non stimulated')
plt.legend()
plt.show()


### A Tau space ###
A_20sec_stim = []
A_40sec_stim = []
A_60sec_stim =[]
A_80sec_stim = []
Tau_20sec_stim = []
Tau_40sec_stim = []
Tau_60sec_stim = []
Tau_80sec_stim = []


for i in range(len(Stims)):
    for j in range(4):
        #if j == 0:
        #    A_20sec_stim.append(np.mean(Stims[i][j][300:,0]))
        #    Tau_20sec_stim.append(np.mean(Stims[i][j][300:,1]))
        if j == 1:
            A_40sec_stim.append(np.mean(Stims[i][j][300:,0]))
            Tau_40sec_stim.append(np.mean(Stims[i][j][300:,1]))
        if j == 2:
            A_60sec_stim.append(np.mean(Stims[i][j][300:,0]))
            Tau_60sec_stim.append(np.mean(Stims[i][j][300:,1]))
        if j == 3:
            A_80sec_stim.append(np.mean(Stims[i][j][300:,0]))
            Tau_80sec_stim.append(np.mean(Stims[i][j][300:,1]))
           
for i in range(10):
    A_20sec_stim.append(np.mean(s20_1[i][300:,0]))
for i in range(10):
    A_20sec_stim.append(np.mean(s20_2[i][300:,0]))
for i in range(10):
    A_20sec_stim.append(np.mean(s20_3[i][300:,0]))
for i in range(10):
    A_20sec_stim.append(np.mean(s20_4[i][300:,0]))


A_20sec_Nonstim = []
A_40sec_Nonstim = []
A_60sec_Nonstim =[]
A_80sec_Nonstim = []
Tau_20sec_Nonstim = []
Tau_40sec_Nonstim = []
Tau_60sec_Nonstim = []
Tau_80sec_Nonstim = []
            

for i in range(len(NonStims)):
    for j in range(4):
        #if j == 0:
        #    A_20sec_Nonstim.append(np.mean(NonStims[i][j][300:,0]))
        #   Tau_20sec_Nonstim.append(np.mean(NonStims[i][j][300:,1]))
        if j == 1:
            A_40sec_Nonstim.append(np.mean(NonStims[i][j][300:,0]))
            Tau_40sec_Nonstim.append(np.mean(NonStims[i][j][300:,1]))
        if j == 2:
            A_60sec_Nonstim.append(np.mean(NonStims[i][j][300:,0]))
            Tau_60sec_Nonstim.append(np.mean(NonStims[i][j][300:,1]))
        if j == 3:
            A_80sec_Nonstim.append(np.mean(NonStims[i][j][300:,0]))
            Tau_80sec_Nonstim.append(np.mean(NonStims[i][j][300:,1]))

            
for i in range(10):
    A_20sec_Nonstim.append(np.mean(ns20_1[i][300:,0]))
for i in range(10):
    A_20sec_Nonstim.append(np.mean(ns20_2[i][300:,0]))
for i in range(10):
    A_20sec_Nonstim.append(np.mean(ns20_3[i][300:,0]))
for i in range(10):
    A_20sec_Nonstim.append(np.mean(ns20_4[i][300:,0]))


plt.figure()
plt.title('Fixed 20 sec')
plt.xlabel('A')
plt.ylabel('Tau')
for i in range(len(A_20sec_stim)):
    if i == 0:
        plt.plot([A_20sec_stim[i]],Tau_20sec_stim[i],'ro',label='Stimulated')
        plt.plot([A_20sec_Nonstim[i]],Tau_20sec_Nonstim[i],'bo',label='Non-stimulated')
    else:
        plt.plot([A_20sec_stim[i]],Tau_20sec_stim[i],'ro')
        plt.plot([A_20sec_Nonstim[i]],Tau_20sec_Nonstim[i],'bo')
plt.legend()
plt.show()

plt.figure()
plt.title('Fixed 40 sec')
plt.xlabel('A')
plt.ylabel('Tau')
for i in range(len(Stims)):
    if i == 0:
        plt.plot([A_40sec_stim[i]],Tau_40sec_stim[i],'ro',label='Stimulated')
        plt.plot([A_40sec_Nonstim[i]],Tau_40sec_Nonstim[i],'bo',label='Non-stimulated')
    else:
        plt.plot([A_40sec_stim[i]],Tau_40sec_stim[i],'ro')
        plt.plot([A_40sec_Nonstim[i]],Tau_40sec_Nonstim[i],'bo')
plt.legend()
plt.show()

plt.figure()
plt.title('Fixed 60 sec')
plt.xlabel('A')
plt.ylabel('Tau')
for i in range(len(Stims)):
    if i == 0:
        plt.plot([A_60sec_stim[i]],Tau_60sec_stim[i],'ro',label='Stimulated')
        plt.plot([A_60sec_Nonstim[i]],Tau_60sec_Nonstim[i],'bo',label='Non-stimulated')
    else:
        plt.plot([A_60sec_stim[i]],Tau_60sec_stim[i],'ro')
        plt.plot([A_60sec_Nonstim[i]],Tau_60sec_Nonstim[i],'bo')
plt.legend()
plt.show()

plt.figure()
plt.title('Fixed 80 sec')
plt.xlabel('A')
plt.ylabel('Tau')
for i in range(len(Stims)):
    if i == 0:
        plt.plot([A_80sec_stim[i]],Tau_80sec_stim[i],'ro',label='Stimulated')
        plt.plot([A_80sec_Nonstim[i]],Tau_80sec_Nonstim[i],'bo',label='Non-stimulated')
    else:
        plt.plot([A_80sec_stim[i]],Tau_80sec_stim[i],'ro')
        plt.plot([A_80sec_Nonstim[i]],Tau_80sec_Nonstim[i],'bo')
plt.legend()
plt.show()


labels2 = ['Sitmulated','Non-stimulated']
secs20 = np.concatenate((A_20sec_stim,A_20sec_Nonstim))
labels = []
for i in range(len(A_20sec_stim)):
    labels.append(1)
for i in range(len(A_20sec_Nonstim)):
    labels.append(2)
matplotlib.rcParams.update({'font.size': 13})
plt.rc('axes', labelsize=15)  
data = np.transpose(np.asarray([secs20,labels]))
df = pd.DataFrame(data, columns =['A Estimate', 'Label'])
ax = sns.histplot(data=df, x="A Estimate", hue="Label",kde=True,edgecolor = 'w',palette=(['cornflowerblue','lightcoral']))
#h, l = ax.get_legend_handles_labels()
#ax.legend(h, labels2, title="Trial")#,loc=[0.33,0.70])
ax.set_title('20 sec')
ax.legend(['Non-stimulated','Stimulated'])



secs40 = np.concatenate((A_40sec_stim,A_40sec_Nonstim))
labels = []
#for i in range(len(A_40sec_stim)):
#    labels.append(0)
for i in range(len(A_40sec_Nonstim)):
    labels.append(1)
matplotlib.rcParams.update({'font.size': 13})
plt.rc('axes', labelsize=15)  
data = np.transpose(np.asarray([A_40sec_Nonstim,labels]))
df = pd.DataFrame(data, columns =['A estimate', 'Label'])
ax = sns.histplot(data=df, x="A estimate", hue="Label",kde=True,edgecolor = 'w',palette=(['lightcoral']))
ax.set_title('40 sec')
ax.legend(['Non-stimulated'])


secs60 = np.concatenate((A_60sec_stim,A_60sec_Nonstim))
labels = []
#for i in range(len(A_60sec_stim)):
#    labels.append(0)
for i in range(len(A_60sec_Nonstim)):
    labels.append(1)
matplotlib.rcParams.update({'font.size': 13})
plt.rc('axes', labelsize=15)  
data = np.transpose(np.asarray([A_60sec_Nonstim,labels]))
df = pd.DataFrame(data, columns =['A estimate', 'Label'])
ax = sns.histplot(data=df, x="A estimate", hue="Label",kde=True,edgecolor = 'w',palette=(['lightcoral']))
ax.set_title('60 sec')
ax.set_xticks([0,0.01,0.02])
#ax.set_xlim([0,0.08])
ax.legend(['Non-stimulated'])


secs80 = np.concatenate((A_80sec_stim,A_80sec_Nonstim))
labels = []
#for i in range(len(A_80sec_stim)):
#    labels.append(0)
for i in range(len(A_80sec_Nonstim)):
    labels.append(1)
matplotlib.rcParams.update({'font.size': 13})
plt.rc('axes', labelsize=15)  
data = np.transpose(np.asarray([A_80sec_Nonstim,labels]))
df = pd.DataFrame(data, columns =['A estimate', 'Label'])
ax = sns.histplot(data=df, x="A estimate", hue="Label",kde=True,edgecolor = 'w',palette=(['lightcoral']))
ax.set_title('80 sec')
ax.legend(['Non-stimulated'])



datasizes = [20,40,60,80]
mediansAStim = [np.median(A_20sec_stim),np.median(A_40sec_stim),np.median(A_60sec_stim),np.median(A_80sec_stim)]
mediansANonStim = [np.median(A_20sec_Nonstim),np.median(A_40sec_Nonstim),np.median(A_60sec_Nonstim),np.median(A_80sec_Nonstim)]
stdsStim = [np.sqrt(np.var(A_20sec_stim)),np.sqrt(np.var(A_40sec_stim)),np.sqrt(np.var(A_60sec_stim)),np.sqrt(np.var(A_80sec_stim))]
stdsNonStim = [np.sqrt(np.var(A_20sec_Nonstim)),np.sqrt(np.var(A_40sec_Nonstim)),np.sqrt(np.var(A_60sec_Nonstim)),np.sqrt(np.var(A_80sec_Nonstim))]
plt.figure()
plt.xlabel('Datasize')
plt.ylabel('median A estimate')
plt.errorbar(datasizes,mediansAStim,yerr=stdsStim,label='Stimulated')
plt.errorbar(datasizes,mediansANonStim,yerr=stdsNonStim,label='Non-stimulated')
#plt.plot(datasizes,mediansAStim,label='Stimulated')
#plt.plot(datasizes,mediansANonStim,label='Non-stimulated')
plt.legend()
plt.plot()
'''

