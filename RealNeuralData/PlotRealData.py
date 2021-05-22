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
#plt.style.use('seaborn-darkgrid')
plt.style.use('default')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


#### Sample 2_1,2_2 correspond to maps 3_1,3_2. Same with 3_1 -> 4_1 and 4_1 > 5_1. Map alltid i+1 v sample :D:D (messed up)
#StimSample = np.load('StimCand1Sample.npy')
#NonStimSample = np.load('NonstimCand1Sample.npy')



StimSample = np.load('SampleStimCand4_1.npy')
NonStimSample = np.load('SampleNonStimCand4_1.npy')


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

mapAstim = np.load('MapAStimCand5_1.npy')
mapTaustim =  np.load('MapTauStimCand5_1.npy')
mapAnonstim =  np.load('MapANonStimCand5_1.npy')
mapTaunonstim =  np.load('MapTauNonStimCand5_1.npy')

entAstim = norm.entropy(loc = meanAstim,scale = np.sqrt(np.var(StimSample[300:,0])))
entAnonstim = norm.entropy(loc = meanAnonstim,scale = np.sqrt(np.var(NonStimSample[300:,0])))
entTaustim = norm.entropy(loc = meanTaustim,scale = np.sqrt(np.var(StimSample[300:,1])))
entTaunonstim = norm.entropy(loc = meanTaunonstim,scale = np.sqrt(np.var(NonStimSample[300:,1])))

x = np.linspace(0,0.2,10000)
priorA = gamma.pdf(x,a=4,scale=1/50)
priorT = gamma.pdf(x,a=5,scale = 1/100)

plt.figure()
sns.distplot(StimSample[300:,0],kde = True,color='g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
#plt.xlim([0.00,0.05])
plt.xlabel(r'A')
plt.axvline(mapAstim,color='k',linestyle='-',label='Sample MAP')
plt.plot(x,priorA,color='#AAA662',alpha=0.9,label='Prior')
#plt.axvline(medAstim,color='g',linestyle='--',label='Median: '+str(medAstim.round(3)))
#plt.axvline(meanAstim,color='m',linestyle='--',label='Mean: '+str(meanAstim.round(3)))
plt.axvline(piAstim[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piAstim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'With stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()

plt.figure()
sns.distplot(NonStimSample[300:,0],kde = True,color = 'g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
#plt.xlim([0.00,0.15])
plt.xlabel(r'A')
plt.plot(x,priorA,color='#AAA662',alpha=0.9,label='Prior')
plt.axvline(mapAnonstim,color='k',linestyle='-',label='Sample MAP')#+str(mapAnonstim[0].round(3)))
#plt.axvline(medAnonstim,color='g',linestyle='--',label='Median: '+str(medAnonstim.round(3)))
#plt.axvline(meanAnonstim,color='m',linestyle='--',label='Mean: '+str(meanAnonstim.round(3)))
plt.axvline(piAnonstim[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piAnonstim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'Without stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()

plt.figure()
sns.distplot(StimSample[300:,1],kde = True,color='g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
#plt.xlim([0.00,0.2])
plt.xlabel(r'Tau')
plt.axvline(mapTaustim,color='k',linestyle='-',label='Sample MAP')
plt.plot(x,priorT,color='#AAA662',alpha=0.9,label='Prior')
#plt.axvline(medAstim,color='g',linestyle='--',label='Median: '+str(medAstim.round(3)))
#plt.axvline(meanAstim,color='m',linestyle='--',label='Mean: '+str(meanAstim.round(3)))
plt.axvline(piTaustim[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piTaustim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'With stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()

plt.figure()
sns.distplot(NonStimSample[300:,1],kde = True,color='g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
#plt.xlim([0.00,0.15])
plt.xlabel(r'Tau')
plt.axvline(mapTaunonstim,color='k',linestyle='-',label='Sample MAP')
plt.plot(x,priorT,color='#AAA662',alpha=0.9,label='Prior')
#plt.axvline(medAstim,color='g',linestyle='--',label='Median: '+str(medAstim.round(3)))
#plt.axvline(meanAstim,color='m',linestyle='--',label='Mean: '+str(meanAstim.round(3)))
plt.axvline(piTaunonstim[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piTaunonstim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'Without stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend(loc='upper center')

StimSample = np.load('SampleStimCand4_2.npy')
NonStimSample = np.load('SampleNonStimCand4_2.npy')


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

mapAstim = np.load('MapAStimCand5_2.npy')
mapTaustim =  np.load('MapTauStimCand5_2.npy')
mapAnonstim =  np.load('MapANonStimCand5_2.npy')
mapTaunonstim =  np.load('MapTauNonStimCand5_2.npy')

entAstim = norm.entropy(loc = meanAstim,scale = np.sqrt(np.var(StimSample[300:,0])))
entAnonstim = norm.entropy(loc = meanAnonstim,scale = np.sqrt(np.var(NonStimSample[300:,0])))
entTaustim = norm.entropy(loc = meanTaustim,scale = np.sqrt(np.var(StimSample[300:,1])))
entTaunonstim = norm.entropy(loc = meanTaunonstim,scale = np.sqrt(np.var(NonStimSample[300:,1])))

plt.figure()
sns.distplot(StimSample[300:,0],kde = True,color='g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
#plt.xlim([0.00,0.05])
plt.xlabel(r'A')
plt.axvline(mapAstim,color='k',linestyle='-',label='Sample MAP')
plt.plot(x,priorA,color='#AAA662',alpha=0.9,label='Prior')
#plt.axvline(medAstim,color='g',linestyle='--',label='Median: '+str(medAstim.round(3)))
#plt.axvline(meanAstim,color='m',linestyle='--',label='Mean: '+str(meanAstim.round(3)))
plt.axvline(piAstim[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piAstim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'With stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()

plt.figure()
sns.distplot(NonStimSample[300:,0],kde = True,color = 'g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
#plt.xlim([0.00,0.15])
plt.xlabel(r'A')
plt.plot(x,priorA,color='#AAA662',alpha=0.9,label='Prior')
plt.axvline(mapAnonstim,color='k',linestyle='-',label='Sample MAP')#+str(mapAnonstim[0].round(3)))
#plt.axvline(medAnonstim,color='g',linestyle='--',label='Median: '+str(medAnonstim.round(3)))
#plt.axvline(meanAnonstim,color='m',linestyle='--',label='Mean: '+str(meanAnonstim.round(3)))
plt.axvline(piAnonstim[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piAnonstim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'Without stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()

plt.figure()
sns.distplot(StimSample[300:,1],kde = True,color='g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
#plt.xlim([0.00,0.2])
plt.xlabel(r'Tau')
plt.axvline(mapTaustim,color='k',linestyle='-',label='Sample MAP')
plt.plot(x,priorT,color='#AAA662',alpha=0.9,label='Prior')
#plt.axvline(medAstim,color='g',linestyle='--',label='Median: '+str(medAstim.round(3)))
#plt.axvline(meanAstim,color='m',linestyle='--',label='Mean: '+str(meanAstim.round(3)))
plt.axvline(piTaustim[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piTaustim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'With stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()

plt.figure()
sns.distplot(NonStimSample[300:,1],kde = True,color='g')
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
#plt.xlim([0.00,0.15])
plt.xlabel(r'Tau')
plt.axvline(mapTaunonstim,color='k',linestyle='-',label='Sample MAP')
plt.plot(x,priorT,color='#AAA662',alpha=0.9,label='Prior')
#plt.axvline(medAstim,color='g',linestyle='--',label='Median: '+str(medAstim.round(3)))
#plt.axvline(meanAstim,color='m',linestyle='--',label='Mean: '+str(meanAstim.round(3)))
plt.axvline(piTaunonstim[0],color='k',linestyle='--',label='95% CI')
plt.axvline(piTaunonstim[1],color='k',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'Without stimulation')
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()

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
