#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 12:25:25 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

datasizes = [60,120,180,240,300] #seconds

#ONLY A, 2ms binsize, different datasizes

samples = []
for i in range(4):
    w0est1 = -np.inf
    samples_temp = [] 
    while (w0est1 < 0.97 or w0est1 > 1.03):
        data = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.0001,b1=-3.1, b2=-3.1, w0=1.0,sec = datasizes[-1], binsize = 1/500.0)
        data.create_data()
        s1,s2,t,W = data.get_data()
        inference = ut.ParameterInference(s1,s2,P = 200, Usim = 100, Ualt = 200,it = 1500, std=0.0001, N = 2\
             , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=datasizes[-1]\
                 ,binsize = 1/500.0)
        b1est1 = inference.b1_estimation()
        b2est1,w0est1 = inference.b2_w0_estimation()
    for j in range(len(datasizes)):
        s1_sub = s1[:int(len(s1)*(datasizes[j]/datasizes[-1]))]
        s2_sub = s2[:int(len(s2)*(datasizes[j]/datasizes[-1]))]
        inference_sub = ut.ParameterInference(s1_sub,s2_sub,P = 200, Usim = 100, Ualt = 200,it = 1500, std=0.0001, N = 2\
             , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=datasizes[j]\
                 ,binsize = 1/500.0)
        b1est1_sub = inference_sub.b1_estimation()
        b2est1_sub,w0est1_sub = inference_sub.b2_w0_estimation()
        Tauest = inference_sub.standardMH() 
        samples_temp.append(Tauest)
    samples.append(samples_temp)
    

np.save('SimSamples1to4',samples)