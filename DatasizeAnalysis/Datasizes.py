#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 21:16:26 2021

@author: emilam
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut


data = ut.SimulatedData(Ap=0.005, tau=0.02, std=0.001,b1=-2.0, b2=-2.0, w0=1.0)
data.create_data()
s1,s2,t,W = data.get_data()

#Affect of the datasize (longer time scale), 2 noise levels -> check for bigger time domains
#Decorrelation of A and Tau with bigger datasets (not learning rule shape if more data)

