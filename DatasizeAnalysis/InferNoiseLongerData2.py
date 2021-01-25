#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:57:10 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

datasizes = [60,120,180,240,300] #seconds

#ONLY A, 2ms binsize, different datasizes

samples = []
for i in range(len(datasizes)):
    w0est1 = -np.inf
    while (w0est1 < 0.97 or w0est1 > 1.03):
        data = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.001,b1=-2, b2=-2, w0=1.0,sec = datasizes[i], binsize = 1/200.0)
        data.create_data()
        s1,s2,t,W = data.get_data()
        inference = ut.ParameterInference(s1,s2,P = 1000, Usim = 100, Ualt = 200,it = 1500, N = 1\
             , shapes_prior = np.array([5]), rates_prior = np.array([800]),sec=datasizes[i]\
                 ,binsize = 1/200.0,taufix = 0.02,Afix = 0.005)
        b1est1 = inference.b1_estimation()
        b2est1,w0est1 = inference.b2_w0_estimation()
    Noise = inference.MH_noise() 
    samples.append(Noise)

np.save('SigmaSamples0.001True',samples)