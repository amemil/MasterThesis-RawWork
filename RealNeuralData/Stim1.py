#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 23:28:33 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtOld as ut

pre = np.load('PreStim20sec_cand1.npy')
post = np.load('PostStim20sec_cand1.npy')

inference = ut.ParameterInference(pre,post,len(pre),P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                 , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=120\
                     ,binsize = 1/1000.0,taufix = 0.02,Afix = 0.005)
inference.b1_estimation()
inference.b2_w0_estimation()
sample = inference.standardMH()

np.save('SampleStim20sec_cand1',sample)

