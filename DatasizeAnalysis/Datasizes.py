#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 21:16:26 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

Aps = [0.001,0.002,0.0025,0.003,0.0035,0.004,0.0045,0.0048,0.0049,0.005,0.0051,0.0052,0.0055,0.006,0.0065,0.007,0.0075,0.008]

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



#Affect of the datasize (longer time scale), 2 noise levels -> check for bigger time domains
#Decorrelation of A and Tau with bigger datasets (not learning rule shape if more data)

