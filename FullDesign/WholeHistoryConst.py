#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 19:42:03 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut
s1init = np.load('FixedInitS1.npy')
s2init = np.load('FixedInitS2.npy')
Winit = np.load('FixedInitW.npy')

design = ut.ExperimentDesign(freqs_init=np.array([20,50,100,200]),maxtime=60,trialsize=5\
                 ,Ap=0.005, tau=0.02, genstd=0.0001,b1=-3.1, b2=-3.1, w0=1.0,binsize = 1/500.0,reals = 20,longinit = 60\
                     ,s1init = s1init,s2init = s2init,Winit = Winit)
means,entrs,optms = design.onlineDesign_wh(nofreq =False,constant = True, random = False, optimised = False)

np.save('EstimatesWholeConst20_1',means)
np.save('EntropiesWholeConst20_1',entrs)
#np.save('OptFrequenciesWholeConst20_1',optms)