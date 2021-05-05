#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:15:42 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

s1init = np.load('s1init_8.npy')
s2init = np.load('s2init_8.npy')
Winit = np.load('Winit_8.npy')

design = ut.ExperimentDesign(freqs_init=np.array([20,50,100,200]),maxtime=60,trialsize=5\
                 ,Ap=0.005, tau=0.02, genstd=0.0001,b1=-3.1, b2=-3.1, w0=1.0,binsize = 1/500.0,reals = 17,longinit = 60\
                     ,s1init = s1init,s2init = s2init,Winit = Winit)
means,entrs,optms,mutinfs,W,posts = design.onlineDesign_wh(nofreq =False,constant = False, random = False, optimised = True)

np.save('WHestmimates_8',means)
np.save('WHentropies_8',entrs)
np.save('WHoptfrqs_8',optms)
np.save('WHmutinfs_8',mutinfs)
np.save('WHw_8',W)
np.save('WHposts_8',posts)