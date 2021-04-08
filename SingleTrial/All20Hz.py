#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:08:12 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtOld as ut

data2 = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.0001,b1=-3.1, b2=-3.1, w0=1.0,sec = 120, binsize = 1/500.0,freq = 20)
data2.create_freq_data()
s12,s22,_,W2 = data2.get_data()

theta_10ms_freq = []


for j in range(12):    
    inference4 = ut.ParameterInference(s12[j*5000:(j+1)*5000],s22[j*5000:(j+1)*5000],P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                    , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=10\
                                            ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
    theta_10ms_freq.append(inference4.standardMH(W2[j*5000],-3.1,-3.1))


theta_5ms_freq=[]

theta_1ms_freq=[]

for i in range(120):    
    inference2 = ut.ParameterInference(s12[i*500:(i+1)*500],s22[i*500:(i+1)*500],P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                        , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=1\
                                            ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
    theta_1ms_freq.append(inference2.standardMH(W2[i*500],-3.1,-3.1))


for k in range(24):    
    inference4 = ut.ParameterInference(s12[k*2500:(k+1)*2500],s22[k*2500:(k+1)*2500],P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                    , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=5\
                                            ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
    theta_5ms_freq.append(inference4.standardMH(W2[k*2500],-3.1,-3.1))
    
np.save('weight20hz',W2)
np.save('tensec20hzSample',theta_10ms_freq)
np.save('fivesec20hzSample',theta_5ms_freq)
np.save('onesec20hzSample',theta_1ms_freq)
np.save('pre20hz',s12)
np.save('post20hz',s22)