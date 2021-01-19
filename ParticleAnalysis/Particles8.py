#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 20:38:07 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

std = 0.005
w0 = 1.0
b1 = -2
b2 = -2
Ap = 0.005
Am = Ap*1.05
tau = 20.0e-3
seconds = 120.0
binsize = 1/200.0
P = 1000
U = 100
it = 1500
shapes_prior = 4
rates_prior = 50

w0ests = []
particles = [50,100,200,500,1000,2000,5000]
Aps = []
Taus = []
for i in range(2):
    w0est1 = -np.inf
    Aps_temp = []
    Tau_temp = []
    while (w0est1 < 0.97 or w0est1 > 1.03):
        data = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.0001,b1=-2.0, b2=-2.0, w0=1.0,sec = 120, binsize = 1/200.0)
        data.create_data()
        s1,s2,t,W = data.get_data()
        inference = ut.ParameterInference(s1,s2,P = 1000, Usim = 100, Ualt = 200,it = 1500, std=0.0001, N = 1\
                 , shapes_prior = np.array([4]), rates_prior = np.array([50]),sec=120,binsize=1/200,taufix = 0.02)
        b1est1 = inference.b1_estimation()
        b2est1,w0est1 = inference.b2_w0_estimation()
    w0est2 = -np.inf
    while (w0est2 < 0.97 or w0est2 > 1.03):
        data2 = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.0001,b1=-3.1, b2=-3.1, w0=1.0,sec = 120, binsize = 1/500.0)
        data2.create_data()
        s12,s22,t2,W2 = data2.get_data()
        inference2 = ut.ParameterInference(s12,s22,P = 1000, Usim = 100, Ualt = 200,it = 1500, std=0.0001, N = 1\
                 , shapes_prior = np.array([5]), rates_prior = np.array([100]),sec=120,binsize=1/500,Afix = 0.02)
        b1est2 = inference2.b1_estimation()
        b2est2,w0est2 = inference2.b2_w0_estimation()
    for j in range(len(particles)):
        inference.set_P(particles[j])
        inference2.set_P(particles[j])
        Apest = inference.standardMH_taufix() 
        Tauest = inference2.standardMH_afix()
        Aps_temp.append(Apest)
        Tau_temp.append(Tauest)
    Aps.append(Aps_temp)
    Taus.append(Tau_temp)
    

#np.save('S1Datasets1to4Opposite',s1s)
#np.save('S2Datasets1to4Opposite',s2s)
#np.save('b1sDatasets1to4Opposite',b1ests)
#np.save('b2sDatasets1to4Opposite',b2ests)
np.save('Aps15to16',Aps)
np.save('Taus15to16',Taus)