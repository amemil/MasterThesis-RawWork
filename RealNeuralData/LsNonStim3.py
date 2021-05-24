#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:21:58 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtOld as ut

pre = np.load('PreLargeScaleNonStim_3.npy')
post = np.load('PostLargeScaleNonStim_3.npy')

secs = [20,40,60]
samples = []
for i in range(3):
    timesteps = int(secs[i]/(1/500))
    inference = ut.ParameterInference(pre[:timesteps],post[:timesteps],timesteps,P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                      , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=120\
                                          ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
    inference.b1_estimation()
    inference.b2_w0_estimation()
    sample = inference.standardMH()
    samples.append(sample)


inference = ut.ParameterInference(pre,post,len(pre),P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                      , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=120\
                                          ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
inference.b1_estimation()
inference.b2_w0_estimation()
sample = inference.standardMH()
samples.append(sample)


np.save('LargeScaleNonStim_3',samples)