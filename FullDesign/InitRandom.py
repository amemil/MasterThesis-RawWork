#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 19:43:25 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

design = ut.ExperimentDesign(freqs_init=np.array([20,50,100,200]),maxtime=60,trialsize=5\
                 ,Ap=0.005, tau=0.02, genstd=0.0001,b1=-3.1, b2=-3.1, w0=1.0,binsize = 1/500.0,reals = 20,longinit = 60)
means,entrs,optms = design.onlineDesign_initdata(nofreq =False,constant = False, random = True, optimised = False)

np.save('EstimatesInitRandom1',means)
np.save('EntropiesInitRandom1',entrs)
np.save('OptFrequenciesInitRandom1',optms)