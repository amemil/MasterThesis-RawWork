#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:51:41 2021

@author: emilam
"""

import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

datasizes = [60,120,180,240,300] #seconds

#sim, 2ms binsize, different datasizes, freq stim
'''
samples = []
datasets = []
for i in range(4):
    w0est1 = -np.inf
    samples_temp = [] 
    while (w0est1 < 0.97 or w0est1 > 1.03):
        data = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.0001,b1=-3.1, b2=-3.1, w0=1.0,sec = datasizes[-1], binsize = 1/500.0,freq = 100)
        data.create_freq_data()
        s1,s2,t,W = data.get_data()
        inference = ut.ParameterInference(s1,s2,P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
             , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=datasizes[-1]\
                 ,binsize = 1/500.0,Afix = 0.005)
        b1est1 = inference.b1_estimation()
        b2est1,w0est1 = inference.b2_w0_estimation()
    datasets_temp = [s1,s2]
    datasets.append(datasets_temp)
    for j in range(len(datasizes)):
        s1_sub = s1[:int(len(s1)*(datasizes[j]/datasizes[-1]))]
        s2_sub = s2[:int(len(s2)*(datasizes[j]/datasizes[-1]))]
        inference_sub = ut.ParameterInference(s1_sub,s2_sub,P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
             , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=datasizes[j]\
                 ,binsize = 1/500.0,Afix = 0.005)
        b1est1_sub = inference_sub.b1_estimation()
        b2est1_sub,w0est1_sub = inference_sub.b2_w0_estimation()
        est = inference_sub.standardMH() 
        samples_temp.append(est)
    samples.append(samples_temp)
    

np.save('Samples17to20highfreq',samples)
#np.save('secondsss',datasizes)
np.save('data17to20highfreq',datasets)
'''