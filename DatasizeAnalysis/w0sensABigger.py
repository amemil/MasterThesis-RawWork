#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 13:26:00 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

data = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.0005,b1=-2, b2=-2, w0=1.0,sec = 600, binsize = 1/200.0)

Aps = [0.001,0.002,0.0025,0.003,0.0035,0.004,0.0045,0.0048,0.0049,0.005,0.0051,0.0052,0.0055,0.006,0.0065,0.007,0.0075,0.008]

wests = []
loglikesAp = []
for i in range(200):
    data.create_data()
    s1,s2,t,W = data.get_data()
    inference = ut.ParameterInference(s1,s2,P = 1000, Usim = 100, Ualt = 200,it = 1500, std=0.0001, N = 1\
                 , shapes_prior = np.array([4]), rates_prior = np.array([50]),sec=600\
                     ,binsize = 1/200.0,taufix = 0.02)
    b1est2 = inference.b1_estimation()
    b2est2,w0est2 = inference.b2_w0_estimation()
    wests.append(w0est2)
    particlelogliks = []
    for j in range(len(Aps)):
        particlelogliks.append(inference.particle_filter(Aps[j],0.02)[2])
    loglikesAp.append(particlelogliks)


np.save('ApLoglikes10sec',loglikesAp)
np.save('w0estslong',wests)
np.save('Aps',Aps)

#datasizes = [60,120,180,240,300]
#data = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.001,b1=-2, b2=-2, w0=1.0,sec = 600, binsize = 1/200.0)
#data.create_data()
#s1,s2,t,W = data.get_data()

#np.save('test',s2)

#inference = ut.ParameterInference(s1,s2,P = 1000, Usim = 100, Ualt = 200,it = 1500, std=0.0001, N = 1\
                 #, shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=600\
                 #    ,binsize = 1/200.0,taufix = 0.02,Afix = 0.005)
#b1est = inference.b1_estimation()
#b2est,w0est = inference.b2_w0_estimation()

#theta_sim = inference.standardMH_taufix()