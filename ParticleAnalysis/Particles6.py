#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 20:37:47 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

particles = [1,5,10,20,50,100,200]
samples = []
for i in range(2):
    w0est1 = -np.inf
    samples_temp = []
    while (w0est1 < 0.97 or w0est1 > 1.03):
        data = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.0001,b1=-3.1, b2=-3.1, w0=1.0,sec = 120, binsize = 1/500.0,freq = 50)
        data.create_data()
        s1,s2,t,W = data.get_data()
        inference = ut.ParameterInference(s1,s2,P = 50, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                     , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=120\
                                         ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
        b1est1 = inference.b1_estimation()
        b2est1,w0est1 = inference.b2_w0_estimation()
    for j in range(len(particles)):
        inference.set_P(particles[j])
        sample = inference.standardMH() 
        samples_temp.append(sample)
    samples.append(samples_temp)

np.save('Particles11to12',samples)
