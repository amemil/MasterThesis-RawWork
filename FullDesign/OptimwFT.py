#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:18:26 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

design = ut.ExperimentDesign(freqs_init=np.array([20,50,100,200]),maxtime=60,trialsize=5\
                 ,Ap=0.005, tau=0.02, genstd=0.0001,b1=-3.1, b2=-3.1, w0=1.0,binsize = 1/500.0,reals = 20,longinit = 60)
means,entrs,optms,mutinf = design.onlineDesign_wh_true(nofreq =False,constant = False, random = False, optimised = True)

np.save('EstimatesWholeOptimFirstTrial',means)
np.save('EntropiesWholeOptimFirstTrial',entrs)
np.save('OptFrequenciesWholeOptimFirstTrial',optms)
np.save('Mutinfs_wholeFirstTrial',mutinf)