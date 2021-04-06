#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 17:48:10 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

datasizes = [60,120,180,240,300] #seconds

s1 = np.load('data1to4medfreq.npy')[0][0][:]
s2 = np.load('data1to4medfreq.npy')[0][1][:]

samples = []
inference = ut.ParameterInference(s1,s2,P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 1\
             , shapes_prior = np.array([5]), rates_prior = np.array([100]),sec=datasizes[-1]\
                 ,binsize = 1/500.0,Afix = 0.005)
b1est1 = inference.b1_estimation()
b2est1,w0est1 = inference.b2_w0_estimation()

for j in range(len(datasizes)):
    s1_sub = s1[:int(len(s1)*(datasizes[j]/datasizes[-1]))]
    s2_sub = s2[:int(len(s2)*(datasizes[j]/datasizes[-1]))]
    inference_sub = ut.ParameterInference(s1_sub,s2_sub,P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 1\
         , shapes_prior = np.array([5]), rates_prior = np.array([100]),sec=datasizes[j]\
             ,binsize = 1/500.0,Afix = 0.005)
    b1est1_sub = inference_sub.b1_estimation()
    b2est1_sub,w0est1_sub = inference_sub.b2_w0_estimation()
    est = inference_sub.standardMH_afix() 
    samples.append(est)

np.save('Tauinfmedfreq1.npy')