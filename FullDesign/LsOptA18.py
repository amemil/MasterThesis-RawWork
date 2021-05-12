#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 19:55:15 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

s1init = np.load('s1init_18.npy')
s2init = np.load('s2init_18.npy')
Winit = np.load('Winit_18.npy')

design = ut.ExperimentDesign(freqs_init=np.array([20,50,100,200]),maxtime=60,trialsize=5\
                 ,Ap=0.005, tau=0.02, genstd=0.0001,b1=-3.1, b2=-3.1, w0=1.0,binsize = 1/500.0,reals = 15,longinit = 60\
                     ,s1init = s1init,s2init = s2init,Winit = Winit)
means,entrs,optms,mutinfs,W,posts = design.onlineDesign_wh_A(nofreq =False,constant = False, random = False, optimised = True)

np.save('WHestmimatesA_18',means)
np.save('WHentropiesA_18',entrs)
np.save('WHoptfrqsA_18',optms)
np.save('WHmutinfsA_18',mutinfs)
np.save('WHwA_18',W)
np.save('WHpostsA_18',posts)