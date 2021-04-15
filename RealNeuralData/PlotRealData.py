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
plt.style.use('seaborn-darkgrid')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

StimSample = np.load('StimCand1Sample.npy')
NonStimSample = np.load('NonstimCand1Sample.npy')

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

mapAstim = np.load('MapAStimCand1.npy')
mapTaustim =  np.load('MapTauStimCand1.npy')
mapAnonstim =  np.load('MapANonStimCand1.npy')
mapTaunonstim =  np.load('MapTauNonStimCand1.npy')

entAstim = norm.entropy(loc = meanAstim,scale = np.sqrt(np.var(StimSample[300:,0])))
entAnonstim = norm.entropy(loc = meanAnonstim,scale = np.sqrt(np.var(NonStimSample[300:,0])))
entTaustim = norm.entropy(loc = meanTaustim,scale = np.sqrt(np.var(StimSample[300:,1])))
entTaunonstim = norm.entropy(loc = meanTaunonstim,scale = np.sqrt(np.var(NonStimSample[300:,1])))

plt.figure()
sns.displot(StimSample[300:,0],kde = True,color=[0.2,0.4,0.5],bins=100)
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
plt.xlim([0.00,0.06])
plt.xlabel(r'A')
plt.axvline(mapAstim,color='r',linestyle='--',label='MAP: '+str(mapAstim[0].round(3)))
plt.axvline(medAstim,color='g',linestyle='--',label='Median: '+str(medAstim.round(3)))
plt.axvline(meanAstim,color='m',linestyle='--',label='Mean: '+str(meanAstim.round(3)))
plt.axvline(piAstim[0],color='b',linestyle='--',label='95% CI')
plt.axvline(piAstim[1],color='b',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'Posterior distribution A - Stimulated sample - Entropy = '+str(entAstim.round(3)))
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()

plt.figure()
sns.displot(NonStimSample[300:,0],kde = True,color=[0.2,0.4,0.5],bins=100)
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
plt.xlim([0.00,0.06])
plt.xlabel(r'A')
plt.axvline(mapAnonstim,color='r',linestyle='--',label='MAP: '+str(mapAnonstim[0].round(3)))
plt.axvline(medAnonstim,color='g',linestyle='--',label='Median: '+str(medAnonstim.round(3)))
plt.axvline(meanAnonstim,color='m',linestyle='--',label='Mean: '+str(meanAnonstim.round(3)))
plt.axvline(piAnonstim[0],color='b',linestyle='--',label='95% CI')
plt.axvline(piAnonstim[1],color='b',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'Posterior distribution A - Non stimulated sample - Entropy = '+str(entAnonstim.round(3)))
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()

plt.figure()
sns.displot(StimSample[300:,1],kde = True,color=[0.2,0.4,0.5],bins=100)
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
plt.xlim([0.00,0.15])
plt.xlabel(r'$\tau$')
plt.axvline(mapTaustim,color='r',linestyle='--',label='MAP: '+str(mapTaustim[0].round(3)))
plt.axvline(medTaustim,color='g',linestyle='--',label='Median: '+str(medTaustim.round(3)))
plt.axvline(meanTaustim,color='m',linestyle='--',label='Mean: '+str(meanTaustim.round(3)))
plt.axvline(piTaustim[0],color='b',linestyle='--',label='95% CI')
plt.axvline(piTaustim[1],color='b',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'Posterior distribution $\tau$ - Stimulated sample - Entropy = '+str(entTaustim.round(3)))
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()

plt.figure()
sns.displot(NonStimSample[300:,1],kde = True,color=[0.2,0.4,0.5],bins=100)
#plt.plot(np.linspace(1,1500,1500),AltReal[:,0],'ko')
plt.xlim([0.00,0.15])
plt.xlabel(r'$\tau$')
plt.axvline(mapTaunonstim,color='r',linestyle='--',label='MAP: '+str(mapTaunonstim[0].round(3)))
plt.axvline(medTaunonstim,color='g',linestyle='--',label='Median: '+str(medTaunonstim.round(3)))
plt.axvline(meanTaunonstim,color='m',linestyle='--',label='Mean: '+str(meanTaunonstim.round(3)))
plt.axvline(piTaunonstim[0],color='b',linestyle='--',label='95% CI')
plt.axvline(piTaunonstim[1],color='b',linestyle='--')
#plt.plot(X,DensAp1.pdf(X),label='Scipy')
plt.title(r'Posterior distribution $\tau$ - Non stimulated sample - Entropy = '+str(entTaunonstim.round(3)))
#plt.axvline(np.mean(Simest1[300:,0]),label = 'mean')
#plt.axvline(Map_x,color='g',linestyle='--',label='MAP')
plt.legend()