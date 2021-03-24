#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 21:10:05 2021

@author: emilam
"""

import sys, os
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt 

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut
'''
Ws = []
for i in tqdm(range(100)):
    data = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.0001,b1=-3.1, b2=-3.1, w0=1.0,sec = 300, binsize = 1/500.0,freq = 250)
    data.create_freq_data()
    s1,s2,t,W = data.get_data()
    inference = ut.ParameterInference(s1,s2,P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                      , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=300\
                                          ,binsize = 1/500.0,Afix = 0.005)
    b1est1 = inference.b1_estimation()
    b2est1,w0est1 = inference.b2_w0_estimation()
    Ws.append(w0est1)
'''

plt.figure()
plt.title('$w^0$ estimation - 250 frequency stim')
sns.set_style('darkgrid')
plt.xlabel('$w^0$')
sns.distplot(Ws, norm_hist = True)
plt.axvline(1,color='r',linestyle='--',label='True Value')
#plt.axvline(ci3[0],color='g',linestyle='--')
#plt.axvline(ci3[1],color='g',linestyle='--',label='95% CI')
plt.legend()
plt.show()