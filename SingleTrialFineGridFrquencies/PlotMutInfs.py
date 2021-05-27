#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 13:08:17 2021

@author: emilam
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import UtilitiesMaster as ut

plt.style.use('default')
frequencies = [10,20,30,40,50,60,70,80,100,125,167,250]

'''
### TTRIAL 2 ###
mutInfs2 = np.load('MutinfsTrial2.npy')*(-1)
plt.figure()
plt.title('Trial 2')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs2)
plt.show()

#### TRIAL 3 ###
mutInfs3 = np.load('MutinfsTrial3.npy')*(-1)
plt.figure()
plt.title('Trial 3')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs3)
plt.show()


### TRIAL 4 ####
mutInfs4 = np.load('MutinfsTrial4.npy')*(-1)
plt.figure()
plt.title('Trial 4')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs4)
plt.show()


### TRIAL 5 ###
mutInfs5 = np.load('MutinfsTrial5.npy')*(-1)

plt.figure()
plt.title('Trial 5')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs5)
plt.show()
'''

plt.rc('axes', labelsize=15)  
mutinfs = np.hstack((mutInfs2/max(mutInfs2),mutInfs3/max(mutInfs3),mutInfs4/max(mutInfs4),mutInfs5/max(mutInfs5)))
frequencies = [10,20,30,40,50,60,70,80,100,125,167,250]
frqs = np.asarray([frequencies,frequencies,frequencies,frequencies]).flatten()
labels = np.asarray([np.full(len(frequencies),2),np.full(len(frequencies),3),np.full(len(frequencies),4),np.full(len(frequencies),5)]).flatten()
labels2 = ['2','3', '4', '5']
data = np.transpose(np.asarray([mutinfs,frqs,labels]))
df = pd.DataFrame(data, columns =['Normalised mutual information', 'Frequency','Trial'])
ax = sns.barplot(data=df, x="Frequency", y="Normalised mutual information",hue="Trial",palette=['bisque','navajowhite','sandybrown','orange'])#,'royalblue'])
#ax.legend(['Trial 2','Trial 3','Trial 4','Trial 5'])
h, l = ax.get_legend_handles_labels()
ax.legend(h, labels2, title="Trial")
ax.set_ylim([0.9,1.002])
ax.set_yticks([0.9,0.95,1])
ax.set_xticklabels(['10','20','30','40','50','60','70','80','100','125','167','250'])

'''
x = np.arange(len(frequencies))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 3*width/4, mutInfs2/max(mutInfs2), width, label='Trial 2')
rects2 = ax.bar(x - width/2, mutInfs3/max(mutInfs3), width, label='Trial 3')
rects3 = ax.bar(x + width/2, mutInfs4/max(mutInfs4), width, label='Trial 4')
rects4 = ax.bar(x + 3*width/4, mutInfs5/max(mutInfs5), width, label='Trial 5')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Normalised mutual information')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(frequencies)
ax.legend()
ax.set_ylim([0.8,1.002])

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)
ax.bar_label(rects4, padding=3)

fig.tight_layout()

plt.show()
'''
'''
plt.figure()
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs2/max(mutInfs2),label='Trial 2')
plt.plot(frequencies,mutInfs3/max(mutInfs3),label='Trial 3')
plt.plot(frequencies,mutInfs4/max(mutInfs4),label='Trial 4')
plt.plot(frequencies,mutInfs5/max(mutInfs5),label='Trial 5')
plt.legend()
plt.yscale('log')
'''
## TRIAL7 ##

#candidates : trial7,trial9, trial 13,trial 11_2, trial 13_2

mutInfs7 = np.load('MutinfsTrial7.npy')*(-1)
'''
plt.figure()
plt.title('Trial 7')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs7)
plt.show()
'''
## TRIAL8 ##
'''
mutInfs8 = np.load('MutinfsTrial8.npy')
plt.figure()
plt.title('Trial 8')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs8)
plt.show()
'''
## TRIAL 9 ##

mutInfs9 = np.load('MutinfsTrial9.npy')*(-1)

'''
plt.figure()
plt.title('Trial 9')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs9)
plt.show()

## TRIAL 10 ##
mutInfs10 = np.load('MutinfsTrial10_2.npy')
plt.figure()
plt.title('Trial 10')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs10)
plt.show()
## TRIAL 11 ##
'''
mutInfs11 = np.load('MutinfsTrial11_2.npy')*(-1)
'''
plt.figure()
plt.title('Trial 11')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs11)
plt.show()

## TRIAL 12 ##
mutInfs12 = np.load('MutinfsTrial12_2.npy')
plt.figure()
plt.title('Trial 12')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs12)
plt.show()
'''
## TRIAL 13 ##

mutInfs13 = np.load('MutinfsTrial13.npy')*(-1)
plt.figure()
plt.title('Trial 13')
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
plt.plot(frequencies,mutInfs13)
plt.show()

mutInfs13_2 = np.load('MutinfsTrial13_2.npy')*(-1)
plt.rc('axes', labelsize=15)  
mutinfs = np.hstack((mutInfs7/max(mutInfs7),mutInfs9/max(mutInfs9),mutInfs11/max(mutInfs11),mutInfs13/max(mutInfs13)))
frequencies = [10,20,30,40,50,60,70,80,100,125,167,250]
frqs = np.asarray([frequencies,frequencies,frequencies,frequencies]).flatten()
labels = np.asarray([np.full(len(frequencies),2),np.full(len(frequencies),3),np.full(len(frequencies),4),np.full(len(frequencies),5)]).flatten()
labels2 = ['7','9', '11', '13']
data = np.transpose(np.asarray([mutinfs,frqs,labels]))
df = pd.DataFrame(data, columns =['Normalised mutual information', 'Frequency','Trial'])
ax = sns.barplot(data=df, x="Frequency", y="Normalised mutual information",hue="Trial",palette=['bisque','navajowhite','sandybrown','orange'])#,'royalblue'])
#ax.legend(['Trial 2','Trial 3','Trial 4','Trial 5'])
h, l = ax.get_legend_handles_labels()
ax.legend(h, labels2, title="Trial",loc=[0.33,0.70])
ax.set_ylim([0.9961,1.00])
ax.set_yticks([0.9970,0.9985,1])
ax.set_xticklabels(['10','20','30','40','50','60','70','80','100','125','167','250'])
'''
mutInfs9 = np.load('MutinfsTrial9_2.npy')

mutInfs9=mutInfs9/(min(mutInfs9))
tickss=[1,2,3,4,5,6,7,8,9,10,11,12]
plt.figure()
plt.xlabel('Frequency')
plt.ylabel('Mutual information')
#plt.bar(x=[1,2,3,4,5,6,7,8,9,10,11,12],height = mutInfs9)
plt.plot(tickss,mutInfs9)
plt.xticks(ticks=tickss,labels=['10','20','30','40','50','60','70','80','100','125','167','250'])
#plt.ylim([8,10.5])
plt.yscale('log')
'''