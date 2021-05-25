#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 13:08:17 2021

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
frequencies = [10,20,30,40,50,60,70,80,100,125,167,250]
## TRIAL7 ##

mutInfs7 = np.load('MutinfsTrial7.npy')
 
plt.figure()
plt.title('Trial 7')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs7)
plt.show()

## TRIAL8 ##

mutInfs8 = np.load('MutinfsTrial8.npy')
plt.figure()
plt.title('Trial 8')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs8)
plt.show()
## TRIAL 9 ##

mutInfs9 = np.load('MutinfsTrial9.npy')
plt.figure()
plt.title('Trial 9')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs9)
plt.show()
## TRIAL 10 ##
mutInfs10 = np.load('MutinfsTrial10.npy')
plt.figure()
plt.title('Trial 10')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs10)
plt.show()
## TRIAL 11 ##

mutInfs11 = np.load('MutinfsTrial11.npy')
plt.figure()
plt.title('Trial 11')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs11)
plt.show()

## TRIAL 12 ##
mutInfs12 = np.load('MutinfsTrial12.npy')
plt.figure()
plt.title('Trial 12')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs12)
plt.show()
## TRIAL 13 ##

mutInfs13 = np.load('MutinfsTrial13.npy')
plt.figure()
plt.title('Trial 13')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs13)
plt.show()
