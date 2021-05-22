#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 13:20:18 2021

@author: emilam
"""
import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

s1 = np.load('s1Trial12.npy')
s2=np.load('s2Trial12.npy')
W = np.load('WTrial12.npy')

inference = ut.ParameterInference(s1,s2,P = 50, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                        , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=55\
                                            ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005,b1est = -1.4,b2est = -3.1,w0est = 1,W=W)
sample = inference.standardMH()
means, cov = [np.mean(sample[300:,0]),np.mean(sample[300:,1])], np.cov(np.transpose(sample[300:,:]))


design = ut.ExperimentDesign(freqs_init=np.array([10,20,29,40,50,60,70,80,100,125,150,250]),maxtime=60,trialsize=5\
                                  ,Ap=0.005, tau=0.02, genstd=0.0001,b1=-3.1, b2=-3.1, w0=1.0,binsize = 1/500.0,reals = 15,longinit = 60)

new_shapes, new_rates = design.adjust_proposal(means,sample)
inference_optim = ut.ParameterInference(1,1,P = 50, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                                 , shapes_prior = new_shapes, rates_prior = new_rates,sec=5\
                                                     ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005,b1est = -3.1,b2est = -3.1,w0est = W[-1])

_,mutinfs = design.freq_optimiser(means,cov,init=False,optim=True,l=False,inference = inference_optim)

np.save('MutinfsTrial12',mutinfs)