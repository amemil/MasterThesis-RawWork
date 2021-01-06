#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:17:41 2021

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
s1s = []
s2s = []
b1ests = []
b2ests = []
stds = [0.0001,0.0005,0.001,0.003,0.005]
Aps = []
for i in range(4):
    w0est1 = -np.inf
    Aps_temp = []
    while (w0est1 < 0.97 or w0est1 > 1.03):
        data = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.005,b1=-2.0, b2=-2.0, w0=1.0)
        data.create_data()
        s1,s2,t,W = data.get_data()
        inference = ut.ParameterInference(s1,s2,P = 1000, Usim = 100, Ualt = 200,it = 1500, std=stds[0], N = 1\
                 , shapes_prior = np.array([4]), rates_prior = np.array([50]),taufix = 0.02)
        b1est1 = inference.b1_estimation()
        b2est1,w0est1 = inference.b2_w0_estimation()
    for j in range(len(stds)):
        inference.set_std(stds[j])
        Apest = inference.standardMH_taufix() 
        Aps_temp.append(Apest)
    w0ests.append(w0est1)
    s1s.append(s1)
    s2s.append(s2)
    b1ests.append(b1est1)
    b2ests.append(b2est1)
    Aps.append(Aps_temp)
    

#np.save('S1Datasets1to4Opposite',s1s)
#np.save('S2Datasets1to4Opposite',s2s)
#np.save('b1sDatasets1to4Opposite',b1ests)
#np.save('b2sDatasets1to4Opposite',b2ests)
np.save('ApInferenceData13to16Opposite',Aps)
#np.save('w0ests1to4Opposite',w0ests)
#np.save('NoisesOpposite',stds)

#20 datasets underestimated noise