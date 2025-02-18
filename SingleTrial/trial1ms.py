#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 00:35:58 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

data=ut.SimulatedData(Ap=0.005, tau=0.02, std=0.0001,b1=-3.1, b2=-3.1, w0=1.0,sec = 120, binsize = 1/500.0,freq = 50)
data.create_data()
s1,s2,_,W=data.get_data()

data2 = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.0001,b1=-3.1, b2=-3.1, w0=1.0,sec = 120, binsize = 1/500.0,freq = 50)
data2.create_freq_data()
s12,s22,_,W2 = data2.get_data()
    
    
theta_1ms_baseline = [] 
theta_1ms_freq = []
for i in range(120):    
    inference1 = ut.ParameterInference(s1[i*500:(i+1)*500],s2[i*500:(i+1)*500],P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                    , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=1\
                                        ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
    theta_1ms_baseline.append(inference1.standardMH(W[i*500],-3.1,-3.1))
    inference2 = ut.ParameterInference(s12[i*500:(i+1)*500],s22[i*500:(i+1)*500],P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                    , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=1\
                                        ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
    theta_1ms_freq.append(inference2.standardMH(W2[i*500],-3.1,-3.1))
    

np.save('1msbase2',theta_1ms_baseline)
np.save('1msfreq2',theta_1ms_freq)