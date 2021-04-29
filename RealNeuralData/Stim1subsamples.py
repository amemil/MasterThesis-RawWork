#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:47:17 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtOld as ut

pre = np.load('prestim.npy')
post = np.load('poststim.npy')

secs = [20,40,60,80]
samples = []
for i in range(4):
    timesteps = int(secs[i]/(1/1000))
    inference = ut.ParameterInference(pre[:timesteps],post[:timesteps],timesteps,P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                      , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=120\
                                          ,binsize = 1/1000.0,taufix = 0.02,Afix = 0.005)
    inference.b1_estimation()
    inference.b2_w0_estimation()
    sample = inference.standardMH()
    samples.append(sample)

np.save('SubsamplesStim1',samples)