#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:31:46 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

w0s = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5]

samples = []

for i in range(20):
    samples_temp = []
    for j in range(len(w0s)):
        data = ut.SimulatedData(Ap=0.02, tau=0.02, std=0.0001,b1=-3.1, b2=-3.1, w0=w0s[j],sec = 5, binsize = 1/500.0,freq = 10)
        data.create_freq_data()
        s1,s2,_,W = data.get_data()
        infer= ut.ParameterInference(s1,s2,P = 50, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                     , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=5\
                                         ,binsize = 1/500.0,taufix = 0.02,Afix = 0.02,b1est = -3.1,b2est = -3.1,w0est = w0s[j])
        sample = infer.standardMH()
        samples_temp.append(sample)
    samples.append(samples_temp)
    
np.save('Samples5sec10Hz1to20_Lr4',samples)