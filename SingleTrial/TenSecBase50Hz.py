#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:43:19 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtOld as ut
'''
data=ut.SimulatedData(Ap=0.005, tau=0.02, std=0.0001,b1=-3.1, b2=-3.1, w0=1.0,sec = 120, binsize = 1/500.0,freq = 50)
data.create_data()
s1,s2,_,W=data.get_data()
#data.plot_weight_trajectory()
data2 = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.0001,b1=-3.1, b2=-3.1, w0=1.0,sec = 120, binsize = 1/500.0,freq = 50)
data2.create_freq_data()
s12,s22,_,W2 = data2.get_data()
'''

s1 = np.load('prebase3.npy',allow_pickle=True)
s2 = np.load('postbase3.npy',allow_pickle=True)
s12 = np.load('prefreq3.npy',allow_pickle=True)
s22 = np.load('postfreq3.npy',allow_pickle=True)
W = np.load('weightbase3.npy',allow_pickle=True)
W2 = np.load('weightfreq3.npy',allow_pickle=True)

theta_5ms_baseline = []
theta_5ms_freq = []

for i in range(12):    
    inference1 = ut.ParameterInference(s1[i*5000:(i+1)*5000],s2[i*5000:(i+1)*5000],P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                    , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=10\
                                            ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
    theta_5ms_baseline.append(inference1.standardMH(W[i*5000],-3.1,-3.1))
    inference2 = ut.ParameterInference(s12[i*5000:(i+1)*5000],s22[i*5000:(i+1)*5000],P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                        , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=10\
                                            ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
    theta_5ms_freq.append(inference2.standardMH(W2[i*5000],-3.1,-3.1))

    
np.save('TenSecBase',theta_5ms_baseline)
np.save('TenSec50hz',theta_5ms_freq)