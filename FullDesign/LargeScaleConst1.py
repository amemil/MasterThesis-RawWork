#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:37:24 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

s1init1 = np.load('s1init_1.npy')
s2init1 = np.load('s2init_1.npy')
Winit1 = np.load('Winit_1.npy')

s1init2 = np.load('s1init_2.npy')
s2init2 = np.load('s2init_2.npy')
Winit2 = np.load('Winit_2.npy')

s1init3 = np.load('s1init_3.npy')
s2init3 = np.load('s2init_3.npy')
Winit3 = np.load('Winit_3.npy')

s1init4 = np.load('s1init_4.npy')
s2init4 = np.load('s2init_4.npy')
Winit4= np.load('Winit_4.npy')

s1init5 = np.load('s1init_5.npy')
s2init5 = np.load('s2init_5.npy')
Winit5 = np.load('Winit_5.npy')

indx = [1,2,3,4,5]
s1s = [s1init1,s1init2,s1init3,s1init4,s1init5]
s2s = [s2init1,s2init2,s2init3,s2init4,s2init5]
ws = [Winit1,Winit2,Winit3,Winit4,Winit5]
for i in range(5):
    design = ut.ExperimentDesign(freqs_init=np.array([20,10,50,100]),maxtime=60,trialsize=5\
                                 ,Ap=0.005, tau=0.02, genstd=0.0001,b1=-3.1, b2=-3.1, w0=1.0,binsize = 1/500.0,reals = 20,longinit = 60\
                                     ,s1init = s1s[i],s2init = s2s[i],Winit = ws[i])
    means,entrs,optms,W,posts = design.onlineDesign_wh(nofreq =False,constant = True, random = False, optimised = False)
    np.save('ConstEstimatesNewGrid_'+str(indx[i]),means)
    np.save('ConstEntropiesNewGrid_'+str(indx[i]),entrs)
    np.save('ConstWNewGrid_'+str(indx[i]),W)
    np.save('ConstPostsNewGrid_'+str(indx[i]),posts)
