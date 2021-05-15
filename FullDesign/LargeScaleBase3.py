#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:37:41 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

s1init1 = np.load('s1init_11.npy')
s2init1 = np.load('s2init_11.npy')
Winit1 = np.load('Winit_11.npy')

s1init2 = np.load('s1init_12.npy')
s2init2 = np.load('s2init_12.npy')
Winit2 = np.load('Winit_12.npy')

s1init3 = np.load('s1init_13.npy')
s2init3 = np.load('s2init_13.npy')
Winit3 = np.load('Winit_13.npy')

s1init4 = np.load('s1init_14.npy')
s2init4 = np.load('s2init_14.npy')
Winit4= np.load('Winit_14.npy')

s1init5 = np.load('s1init_15.npy')
s2init5 = np.load('s2init_15.npy')
Winit5 = np.load('Winit_15.npy')

indx = [11,12,13,14,15]
s1s = [s1init1,s1init2,s1init3,s1init4,s1init5]
s2s = [s2init1,s2init2,s2init3,s2init4,s2init5]
ws = [Winit1,Winit2,Winit3,Winit4,Winit5]
for i in range(5):
    design = ut.ExperimentDesign(freqs_init=np.array([10,20,50,100]),maxtime=60,trialsize=5\
                                 ,Ap=0.005, tau=0.02, genstd=0.0001,b1=-3.1, b2=-3.1, w0=1.0,binsize = 1/500.0,reals = 20,longinit = 60\
                                     ,s1init = s1s[i],s2init = s2s[i],Winit = ws[i])
    means,entrs,optms,W,posts = design.onlineDesign_wh(nofreq =True,constant = False, random = False, optimised = False)
    np.save('BaseEstimatesNewGrid_'+str(indx[i]),means)
    np.save('BaseEntropiesNewGrid_'+str(indx[i]),entrs)
    np.save('BaseWNewGrid_'+str(indx[i]),W)
    np.save('BasePostsNewGrid_'+str(indx[i]),posts)
