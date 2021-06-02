#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 22:54:02 2021

@author: emilam
"""
import numpy as np              
import matplotlib.pyplot as plt 
from scipy.interpolate import make_interp_spline, BSpline
import math
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
from scipy.stats import norm
import pandas as pd
import matplotlib


##### INFO ::::: #######
### LR1 (NOT INCLUDED IN NPY FILES : ORIGINAL LR)
### LR2 : A = 0.005, Tau = 0.005
### LR 3 : A = 0.005, Tau = 0.06
### LR 4: A = 0.02, Tau = 0.02
###LR 5: A= 0.001, Tau = 0.02
plt.style.use('default')

truevalues = np.array([0.005,0.005])

def rmse(targets, predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())

def lr1(s2,s1,Ap,delta,taup):
    return s2*s1*Ap*np.exp(-delta/taup)

def lr2(s1,s2,Am,delta,taum):
    return -s1*s2*Am*1.05*np.exp(delta/taum)

deltas = np.linspace(0,0.1,10000)
deltas2 = np.linspace(-0.1,0,10000)   
lrs1 = lr1(1,1,truevalues[0],deltas,truevalues[1])
lrs2 = lr2(1,1,truevalues[0],deltas2,truevalues[1]) 

plt.rcParams["font.family"] = "Times New Roman"
deltass = np.concatenate((deltas2,deltas))
lrref = np.concatenate((lrs2,lrs1))

#### 1sec trials!! #### 

TenHz1 = np.load('Samples10sec10Hz1to10_Lr2.npy')
TenHz2 = np.load('Samples10sec10Hz11to20_Lr2.npy')
TwHz1 = np.load('Samples10sec20Hz1to10_Lr2.npy')
TwHz2 = np.load('Samples10sec20Hz11to20_Lr2.npy')
FiftHz1 = np.load('Samples10sec50Hz1to10_Lr2.npy')
FiftHz2 = np.load('Samples10sec50Hz11to20_Lr2.npy')
HunHz1 = np.load('Samples10sec100Hz1to10_Lr2.npy')
HunHz2 = np.load('Samples10sec100Hz11to20_Lr2.npy')
TfHz1 = np.load('Samples10sec250Hz1to10_Lr2.npy')
TfHz2 = np.load('Samples10sec250Hz11to20_Lr2.npy')


Tens = [TenHz1,TenHz2]
Tens_rmse = []
for i in range(len(Tens)):
    for j in range(len(Tens[i])):
        rmse_temp = []
        for k in range(len(Tens[i][j])):
            lrs1_temp = lr1(1,1,np.mean(Tens[i][j][k][300:,0]),deltas,np.mean(Tens[i][j][k][300:,1]))
            lrs2_temp = lr2(1,1,np.mean(Tens[i][j][k][300:,0]),deltas2,np.mean(Tens[i][j][k][300:,1])) 
            lr_est = np.concatenate((lrs2_temp,lrs1_temp))
            rmse_temp.append(rmse(lrref, lr_est))
        Tens_rmse.append(rmse_temp)

for i in range(len(Tens_rmse)):
    for j in range(len(Tens_rmse[i])):
        if math.isnan(Tens_rmse[i][j])==True:
            Tens_rmse_temp = np.delete(np.asarray(Tens_rmse)[:,j],i)
            summ = np.sum(Tens_rmse_temp)
            while math.isnan(summ) == True:
                nans = np.where(np.isnan(Tens_rmse_temp))[0][0]
                Tens_rmse_temp = np.delete(np.asarray(Tens_rmse_temp),nans)
                summ = np.sum(Tens_rmse_temp)
            Tens_rmse[i][j] = np.mean(Tens_rmse_temp)
            
        
Tws = [TwHz1,TwHz2]
Tws_rmse = []
for i in range(len(Tws)):
    for j in range(len(Tws[i])):
        rmse_temp = []
        for k in range(len(Tws[i][j])):
            lrs1_temp = lr1(1,1,np.mean(Tws[i][j][k][300:,0]),deltas,np.mean(Tws[i][j][k][300:,1]))
            lrs2_temp = lr2(1,1,np.mean(Tws[i][j][k][300:,0]),deltas2,np.mean(Tws[i][j][k][300:,1])) 
            lr_est = np.concatenate((lrs2_temp,lrs1_temp))
            rmse_temp.append(rmse(lrref, lr_est))
        Tws_rmse.append(rmse_temp)
        
for i in range(len(Tws_rmse)):
    for j in range(len(Tws_rmse[i])):
        if math.isnan(Tws_rmse[i][j])==True:
            Tws_rmse_temp = np.delete(np.asarray(Tws_rmse)[:,j],i)
            summ = np.sum(Tws_rmse_temp)
            while math.isnan(summ) == True:
                nans = np.where(np.isnan(Tws_rmse_temp))[0][0]
                Tws_rmse_temp = np.delete(np.asarray(Tws_rmse_temp),nans)
                summ = np.sum(Tws_rmse_temp)
            Tws_rmse[i][j] = np.mean(Tws_rmse_temp)
        
Fts = [FiftHz1,FiftHz2]
Fts_rmse = []
for i in range(len(Fts)):
    for j in range(len(Fts[i])):
        rmse_temp = []
        for k in range(len(Fts[i][j])):
            lrs1_temp = lr1(1,1,np.mean(Fts[i][j][k][300:,0]),deltas,np.mean(Fts[i][j][k][300:,1]))
            lrs2_temp = lr2(1,1,np.mean(Fts[i][j][k][300:,0]),deltas2,np.mean(Fts[i][j][k][300:,1])) 
            lr_est = np.concatenate((lrs2_temp,lrs1_temp))
            rmse_temp.append(rmse(lrref, lr_est))
        Fts_rmse.append(rmse_temp)
        
for i in range(len(Fts_rmse)):
    for j in range(len(Fts_rmse[i])):
        if math.isnan(Fts_rmse[i][j])==True:
            Fts_rmse_temp = np.delete(np.asarray(Fts_rmse)[:,j],i)
            summ = np.sum(Fts_rmse_temp)
            while math.isnan(summ) == True:
                nans = np.where(np.isnan(Fts_rmse_temp))[0][0]
                Fts_rmse_temp = np.delete(np.asarray(Fts_rmse_temp),nans)
                summ = np.sum(Fts_rmse_temp)
            Fts_rmse[i][j] = np.mean(Fts_rmse_temp)
            
            
Huns = [HunHz1,HunHz2]
Huns_rmse = []
for i in range(len(Huns)):
    for j in range(len(Huns[i])):
        rmse_temp = []
        for k in range(len(Huns[i][j])):
            lrs1_temp = lr1(1,1,np.mean(Huns[i][j][k][300:,0]),deltas,np.mean(Huns[i][j][k][300:,1]))
            lrs2_temp = lr2(1,1,np.mean(Huns[i][j][k][300:,0]),deltas2,np.mean(Huns[i][j][k][300:,1])) 
            lr_est = np.concatenate((lrs2_temp,lrs1_temp))
            rmse_temp.append(rmse(lrref, lr_est))
        Huns_rmse.append(rmse_temp)

for i in range(len(Huns_rmse)):
    for j in range(len(Huns_rmse[i])):
        if math.isnan(Huns_rmse[i][j])==True:
            Huns_rmse_temp = np.delete(np.asarray(Huns_rmse)[:,j],i)
            summ = np.sum(Huns_rmse_temp)
            while math.isnan(summ) == True:
                nans = np.where(np.isnan(Huns_rmse_temp))[0][0]
                Huns_rmse_temp = np.delete(np.asarray(Huns_rmse_temp),nans)
                summ = np.sum(Huns_rmse_temp)
            Huns_rmse[i][j] = np.mean(Huns_rmse_temp)
            
        
Crzs = [TfHz1,TfHz2]
Crzs_rmse = []

Aoutliers = []
Tauoutliers = []
for i in range(len(Crzs)):
    for j in range(len(Crzs[i])):
        rmse_temp = []
        for k in range(len(Crzs[i][j])):
            lrs1_temp = lr1(1,1,np.mean(Crzs[i][j][k][300:,0]),deltas,np.mean(Crzs[i][j][k][300:,1]))
            lrs2_temp = lr2(1,1,np.mean(Crzs[i][j][k][300:,0]),deltas2,np.mean(Crzs[i][j][k][300:,1])) 
            lr_est = np.concatenate((lrs2_temp,lrs1_temp))
            rmse_temp.append(rmse(lrref, lr_est))
            #if j > 1 and j < 5:
             #   print('A:',np.mean(Crzs[i][j][k][300:,0]))
              #  print('Tau: ',np.mean(Crzs[i][j][k][300:,1]))        
               # Aoutliers.append(np.mean(Crzs[i][j][k][300:,0]))
                #Tauoutliers.append(np.mean(Crzs[i][j][k][300:,1]))
        Crzs_rmse.append(rmse_temp)

for i in range(len(Crzs_rmse)):
    for j in range(len(Crzs_rmse[i])):
        if math.isnan(Crzs_rmse[i][j])==True:
            Crzs_rmse_temp = np.delete(np.asarray(Crzs_rmse)[:,j],i)
            summ = np.sum(Crzs_rmse_temp)
            while math.isnan(summ) == True:
                nans = np.where(np.isnan(Crzs_rmse_temp))[0][0]
                Crzs_rmse_temp = np.delete(np.asarray(Crzs_rmse_temp),nans)
                summ = np.sum(Crzs_rmse_temp)
            Crzs_rmse[i][j] = np.mean(Crzs_rmse_temp)
'''
plt.figure()
plt.scatter(Aoutliers,Tauoutliers)
plt.plot([0.005],[0.02],'ro',label='True value')
plt.xlabel('A')
plt.ylabel('Tau')
plt.legend()
plt.show()
'''

Tens_rmse = np.asarray(Tens_rmse).flatten()
Tws_rmse = np.asarray(Tws_rmse).flatten()
Fts_rmse = np.asarray(Fts_rmse).flatten()
Huns_rmse = np.asarray(Huns_rmse).flatten()
Crzs_rmse = np.asarray(Crzs_rmse).flatten()

labels = [] #### 0 = 10hz , 1 =  20hz, 2 = 50hz, 3 = 100hz, 4= 250hz !!!!!!!!!!!!!

for i in range(len(Tens_rmse)):
    labels.append(0)
for i in range(len(Tws_rmse)):
    labels.append(1)
for i in range(len(Fts_rmse)):
    labels.append(2)
for i in range(len(Huns_rmse)):
    labels.append(3)
#for i in range(len(Crzs_rmse)):
#    labels.append(4)

labels = np.asarray(labels)

mses = np.hstack((Tens_rmse,Tws_rmse,Fts_rmse,Huns_rmse))
weights = []
count = 0
for i in range(len(mses)):
    weights.append(0.5+count*1)
    count += 1
    if count == 8:
        count = 0
#matplotlib.rcParams.update({'font.size': 13})
#plt.rc('axes', labelsize=16)  

#plt.rcParams.update(plt.rcParamsDefault)
matplotlib.rcParams.update({'font.size': 15})
plt.rc('axes', labelsize=18)  



data = np.transpose(np.asarray([mses,weights,labels]))
df = pd.DataFrame(data, columns =['RMSE', 'Initial weight','Label'])
ax = sns.lineplot(data=df, x="Initial weight", y="RMSE",hue="Label")#,palette=['orangered','chartreuse','royalblue','gold'])#,'royalblue'])
ax.legend(['10Hz','20Hz','50Hz','100Hz'],loc='upper left')#,'Randomised Frequency','Optimal [10-100hz] grid','Dales Law'])
ax.title.set_text('10s trial inference - New learning rule')
ax.set_yscale('log')
