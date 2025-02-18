#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:12:33 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

s1init = np.load('s1init_4.npy')
s2init = np.load('s2init_4.npy')
Winit = np.load('Winit_4.npy')

design = ut.ExperimentDesign(freqs_init=np.array([10,20,50,100]),maxtime=60,trialsize=5\
                 ,Ap=0.005, tau=0.02, genstd=0.0001,b1=-3.1, b2=-3.1, w0=1.0,binsize = 1/500.0,reals = 15,longinit = 60\
                     ,s1init = s1init,s2init = s2init,Winit = Winit)
means,entrs,optms,mutinfs,W,posts = design.onlineDesign_wh(nofreq =False,constant = False, random = False, optimised = True)

np.save('WHestmimatesNewGrid_4',means)
np.save('WHentropiesNewGrid_4',entrs)
np.save('WHoptfrqsNewGrid_4',optms)
np.save('WHmutinfsNewGrid_4',mutinfs)
np.save('WHwNewGrid_4',W)
np.save('WHpostsNewGrid_4',posts)
