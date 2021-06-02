#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:22:03 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtOld as ut

pre = np.load('PreLargeScaleNonStim_7.npy')
post = np.load('PostLargeScaleNonStim_7.npy')

secs = [10,20,30,40,50,60]
samples = []
for i in range(len(secs)):
    timesteps = int(secs[i]/(1/500))
    inference = ut.ParameterInference(pre[:timesteps],post[:timesteps],timesteps,P = 50, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                      , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=120\
                                          ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
    inference.b1_estimation()
    inference.b2_w0_estimation()
    sample = inference.standardMH()
    samples.append(sample)


np.save('LargeScaleInvNonStim_7',samples)