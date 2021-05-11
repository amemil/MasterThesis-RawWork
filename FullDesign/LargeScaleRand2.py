#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:37:35 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

s1init1 = np.load('s1init_6.npy')
s2init1 = np.load('s2init_6.npy')
Winit1 = np.load('Winit_6.npy')

s1init2 = np.load('s1init_7.npy')
s2init2 = np.load('s2init_7.npy')
Winit2 = np.load('Winit_7.npy')

s1init3 = np.load('s1init_8.npy')
s2init3 = np.load('s2init_8.npy')
Winit3 = np.load('Winit_8.npy')

s1init4 = np.load('s1init_9.npy')
s2init4 = np.load('s2init_9.npy')
Winit4= np.load('Winit_9.npy')

s1init5 = np.load('s1init_10.npy')
s2init5 = np.load('s2init_10.npy')
Winit5 = np.load('Winit_10.npy')

indx = [6,7,8,9,10]
s1s = [s1init1,s1init2,s1init3,s1init4,s1init5]
s2s = [s2init1,s2init2,s2init3,s2init4,s2init5]
ws = [Winit1,Winit2,Winit3,Winit4,Winit5]
for i in range(5):
    design = ut.ExperimentDesign(freqs_init=np.array([20,50,100,200]),maxtime=60,trialsize=5\
                                 ,Ap=0.005, tau=0.02, genstd=0.0001,b1=-3.1, b2=-3.1, w0=1.0,binsize = 1/500.0,reals = 20,longinit = 60\
                                     ,s1init = s1s[i],s2init = s2s[i],Winit = ws[i])
    means,entrs,optms,W,posts = design.onlineDesign_wh(nofreq =False,constant = False, random = True, optimised = False)
    np.save('RandEstimates_'+str(indx[i]),means)
    np.save('RandEntropies_'+str(indx[i]),entrs)
    np.save('RandW_'+str(indx[i]),W)
    np.save('RandPosts_'+str(indx[i]),posts)
    np.save('RandFreqs_'+str(indx[i]),optms)