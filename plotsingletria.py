#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:35:37 2021

@author: emilam
"""

import numpy as np              
import matplotlib.pyplot as plt 

emsb = np.load('1msbase3.npy')
emsf = np.load('1msfreq3.npy')
fmsb = np.load('5msbase3.npy')
fmsf = np.load('5msfreq3.npy')
preb = np.load('prebase3.npy')
postb = np.load('postbase3.npy')
pref= np.load('prefreq3.npy')
postf=np.load('postfreq3.npy')
wb=np.load('weightbase3.npy')
wf=np.load('weightfreq3.npy')
    
frbase = []
frf = []
for i in range(120):
    counter = 0
    counter2 = 0
    for j in range(i*500,(i+1)*500):
        if postb[j] == 1:
            counter += 1
        if postf[j] == 1:
            counter2 += 1
    frbase.append(counter)
    frf.append(counter2)
    
    
    
    
    
base1msmeans = []
base1msstds = []
base5msmeans = []
base5msstds = []
f1msmeans = []
f1msstds = []
f5msmeans = []
f5msstds = []
for i in range(120):
    base1msmeans.append(np.mean(emsb[i][300:],axis=0))
    base1msstds.append(np.sqrt(np.var(emsb[i][300:],axis=0)))
    f1msmeans.append(np.mean(emsf[i][300:],axis=0))
    f1msstds.append(np.sqrt(np.var(emsf[i][300:],axis=0)))
for i in range(24):
    base5msmeans.append(np.mean(fmsb[i][300:],axis=0))
    base5msstds.append(np.sqrt(np.var(fmsb[i][300:],axis=0)))
    f5msmeans.append(np.mean(fmsf[i][300:],axis=0))
    f5msstds.append(np.sqrt(np.var(fmsf[i][300:],axis=0)))

x = np.linspace(1,120,120)
'''
#ticksss = ['60','120','180','240','300']
plt.figure()
plt.title('Single trial A inference - 1ms trials - Baseline firing')
plt.xlabel('Trial')
plt.ylabel('A estimation')
plt.ylim([0,0.2])
plt.xlim([0,121])
#plt.xticks(x,labels = ticksss)
for i in range(120):
    plt.errorbar(x[i], base1msmeans[i][0], yerr = base1msstds[i][0],marker = 'o')        
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
    
plt.figure()
plt.title('Single trial A inference - 5ms trials - Baseline firing')
plt.xlabel('Trial')
plt.ylabel('A estimation')
plt.ylim([0,0.2])
plt.xlim([0,25])
#plt.xticks(x,labels = ticksss)
for i in range(24):
    plt.errorbar(x[i], base5msmeans[i][0], yerr = base5msstds[i][0],marker = 'o')        
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
    
plt.figure()
plt.title('Single trial A inference - 1ms trials - 20 Hz additional stimuli')
plt.xlabel('Trial')
plt.ylabel('A estimation')
plt.ylim([0,0.2])
plt.xlim([0,121])
#plt.xticks(x,labels = ticksss)
for i in range(120):
    plt.errorbar(x[i], f1msmeans[i][0], yerr = f1msstds[i][0],marker = 'o')        
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
    
plt.figure()
plt.title('Single trial A inference - 5ms trials - 20 Hz additional firing')
plt.xlabel('Trial')
plt.ylabel('A estimation')
plt.ylim([0,0.2])
plt.xlim([0,25])
#plt.xticks(x,labels = ticksss)
for i in range(24):
    plt.errorbar(x[i], f5msmeans[i][0], yerr = f5msstds[i][0],marker = 'o')        
plt.axhline(0.005,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
    
plt.figure()
plt.title('Single trial Tau inference - 1ms trials - Baseline firing')
plt.xlabel('Trial')
plt.ylabel('Tau estimation')
plt.ylim([0,0.2])
plt.xlim([0,121])
#plt.xticks(x,labels = ticksss)
for i in range(120):
    plt.errorbar(x[i], base1msmeans[i][1], yerr = base1msstds[i][1],marker = 'o')        
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
    
plt.figure()
plt.title('Single trial Tau inference - 5ms trials - Baseline firing')
plt.xlabel('Trial')
plt.ylabel('Tau estimation')
plt.ylim([0,0.2])
plt.xlim([0,25])
#plt.xticks(x,labels = ticksss)
for i in range(24):
    plt.errorbar(x[i], base5msmeans[i][1], yerr = base5msstds[i][1],marker = 'o')        
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
    
plt.figure()
plt.title('Single trial Tau inference - 1ms trials - 20 Hz additional stimuli')
plt.xlabel('Trial')
plt.ylabel('Tau estimation')
plt.ylim([0,0.2])
plt.xlim([0,121])
#plt.xticks(x,labels = ticksss)
for i in range(120):
    plt.errorbar(x[i], f1msmeans[i][1], yerr = f1msstds[i][1],marker = 'o')        
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()
    
plt.figure()
plt.title('Single trial Tau inference - 5ms trials - 20 Hz additional firing')
plt.xlabel('Trial')
plt.ylabel('Tau estimation')
plt.ylim([0,0.2])
plt.xlim([0,25])
#plt.xticks(x,labels = ticksss)
for i in range(24):
    plt.errorbar(x[i], f5msmeans[i][1], yerr = f5msstds[i][1],marker = 'o')        
plt.axhline(0.02,color='r',linestyle='--',label='True Value')
plt.legend()
plt.show()

fig, ax1 = plt.subplots(figsize=(6.4,5.2))
color = 'tab:red'
ax1.set_xlabel('Time [t]')
ax1.set_ylabel('Synaptic strength', color=color)
ax1.plot(np.linspace(0,120,len(wf)),wf, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
ax2.plot(np.linspace(1,120,120),frf, 'b-',color=color)
ax2.tick_params(axis='y', labelcolor=color)
    
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.subplots_adjust(top=0.92)
plt.title('Firing rate and weight connectivity')
#plt.savefig('stockc')
plt.show()
'''            
            
fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 1s trials A estimation - Baseline vs Stimulus', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(120):
                ax.errorbar(x[i], base1msmeans[i][0], yerr = base1msstds[i][0],marker = 'o')        
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('A estimation')
            ax.set_title('Baseline stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(120):
                ax.errorbar(x[i], f1msmeans[i][0], yerr = f1msstds[i][0],marker = 'o')        
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('20 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wb)),wb, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frbase, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wf)),wf, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frf, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()


x = np.linspace(5,120,24)
fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 5s trials A estimation - Baseline vs Stimulus', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(24):
                ax.errorbar(x[i], base5msmeans[i][0], yerr = base5msstds[i][0],marker = 'o')        
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('A estimation')
            ax.set_title('Baseline stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(24):
                ax.errorbar(x[i], f5msmeans[i][0], yerr = f5msstds[i][0],marker = 'o')        
            ax.axhline(0.005,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('20 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wb)),wb, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frbase, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wf)),wf, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frf, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()

x = np.linspace(1,120,120)
fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 1s trials Tau estimation - Baseline vs Stimulus', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(120):
                ax.errorbar(x[i], base1msmeans[i][1], yerr = base1msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('Tau estimation')
            ax.set_title('Baseline stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(120):
                ax.errorbar(x[i], f1msmeans[i][1], yerr = f1msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            #.set_xlabel('A estimation')
            ax.set_title('20 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wb)),wb, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frbase, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wf)),wf, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frf, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()

x = np.linspace(5,120,24)
fig, axes = plt.subplots(2,2, figsize=(8,8))
j = 0
fig.suptitle('Single 5s trials Tau estimation - Baseline vs Stimulus', fontsize=20)
for ax_row in axes:
    for ax in ax_row:    
        if j == 0:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(24):
                ax.errorbar(x[i], base5msmeans[i][1], yerr = base5msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            ax.set_ylabel('Tau estimation')
            ax.set_title('Baseline stimulus')
            ax.legend()
            #plt.show()
        if j == 1:
            #ax.xlabel('Trial')
            #ax.ylabel('A estimation')
            #ax.ylim([0,0.2])
            #ax.xlim([0,121])
            #plt.xticks(x,labels = ticksss)
            for i in range(24):
                ax.errorbar(x[i], f5msmeans[i][1], yerr = f5msstds[i][1],marker = 'o')        
            ax.axhline(0.02,color='r',linestyle='--',label='True Value')
            #ax.set_xlabel('A estimation')
            ax.set_title('20 Hz additional stimulus')
            ax.legend()
            #plt.show()
        if j == 2:
        # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wb)),wb, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            #ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frbase, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        if j == 3:
            ax2 = ax.twinx()
            color = 'tab:red'
            ax.set_xlabel('Time [t]')
            #ax.set_ylabel('Synaptic strength', color=color)
            ax.plot(np.linspace(0,120,len(wf)),wf, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            # plot some data on each axis.
            color = 'tab:blue'
            ax2.set_ylabel('Postsynaptic firing rate', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.linspace(1,120,120),frf, 'b-',color=color)
            ax2.tick_params(axis='y', labelcolor=color)    
        j += 1
plt.tight_layout()
plt.show()
