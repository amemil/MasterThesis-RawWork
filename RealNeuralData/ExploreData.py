#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:29:09 2021

@author: emilam
"""
import numpy as np
from mat4py import loadmat
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib
#plt.style.use('seaborn-darkgrid')
plt.style.use('default')

#data18105 = loadmat('camkii10_180105.spikes.cellinfo.mat')
#sessioninfo105 = loadmat('camkii10_180105.sessionInfo.mat')
#stiminfo = loadmat('stim_info180105.mat')
#cellparams = loadmat('camkii10_180105_CellParams.mat')
#eventtree = loadmat('camkii10_180105_EventTree.mat')
#pulse = loadmat('TTL_pulse180105.mat')
#position = loadmat('position_info180105.mat')
#ripmod  = loadmat('rip_mod180105.mat')
#ledposition = loadmat('LED_position-01-180105.mat')
matplotlib.rcParams.update({'font.size': 15})
#jux = loadmat('JuxtaGroundTruth.mat')

ses1spikes = jux['ses']['times'][5]
ses1stim=jux['ses']['stimTimes'][5]
ses1spikeidraw = jux['ses']['ID'][5]
ses1spikeid = []
for i in range(len(ses1spikeidraw)):
    ses1spikeid.append(ses1spikeidraw[i][-1])
spiketrains = []
for i in range(len(np.unique(np.asarray(ses1spikeid)))):
    spiketrains.append([])
for i in range(len(ses1spikes)):
    spiketrains[ses1spikeid[i]-2].append(ses1spikes[i])
for i in range(len(spiketrains)):
    spiketrains[i] = np.asarray(spiketrains[i]).flatten()

def cicc(lags,significance,n):
    return stats.norm.interval(significance,0,np.sqrt(1/(n-abs(lags))))

'''
#for i in range(18):
    ses1spikes = jux['ses']['times'][i]
    ses1stim=jux['ses']['stimTimes'][i]
    ses1spikeidraw = jux['ses']['ID'][i]
    ses1spikeid = []
    for i in range(len(ses1spikeidraw)):
        ses1spikeid.append(ses1spikeidraw[i][-1])
    spiketrains = []
    for i in range(len(np.unique(np.asarray(ses1spikeid)))):
        spiketrains.append([])
    for i in range(len(ses1spikes)):
        spiketrains[ses1spikeid[i]-2].append(ses1spikes[i])
    for i in range(len(spiketrains)):
        spiketrains[i] = np.asarray(spiketrains[i]).flatten()

    def cicc(lags,significance,n):
        return stats.norm.interval(significance,0,np.sqrt(1/(n-abs(lags))))
    
session 7, ind 26
session 14 ind 18
'''
## 275-355, 355-435, 435-515,515-595,595-675, 675-755, 755-835,835-915,915-995,995-1075

## 35-115,115-195,195-275,1175-1255,1255-1335,1335-1415,1415-1495,1495-1575,1575-1655,1655-1735

# stim periods: 275-1160, 1713-2019 (utenom 1837-1850)
# non-stim : 0-275, 1160-1713

startstim = 0
stopstim = 2000

    

pre = spiketrains[-1][(np.where((spiketrains[-1] > startstim) & (spiketrains[-1] < stopstim)))]


post = spiketrains[32][(np.where((spiketrains[32] > startstim) & (spiketrains[32] < stopstim)))]
srp = []
srpre = []

postint = post.astype(int)
preint = pre.astype(int)
for j in range(int(max(spiketrains[32]))):
    srp.append(np.count_nonzero(postint==j))        
for j in range(int(max(spiketrains[-1]))):
    srpre.append(np.count_nonzero(preint==j))
    
    
preint = pre*1000    
postint = post*1000
    
postint = postint.astype(int)
preint = preint.astype(int)


start = min(preint[0],postint[0])
end = max(preint[-1],postint[-1])
binsize = 1
bins = int((end-start)/binsize)
timesteps = np.linspace(start,end-binsize,bins)
s1,s2 = [],[]
#s1,s2 = np.zeros(int(bins/2)),np.zeros(int(bins/2))
for k in range(0,bins):
    if (timesteps[k] in preint):#: or timesteps[i]+1 in st2pos):
        s1.append(1)
    else:
        s1.append(0)
    if (timesteps[k] in postint):# or timesteps[i]+1 in st5pos):
        s2.append(1)
    else:
        s2.append(0)
    
#np.save('Pre20secLsStim_'+str(i+1),s1)
#np.save('Post20secLsStim_'+str(i+1),s2)
    
'''
for i in range(40):
    

    startstim = 280+((i)*20)
    stopstim = 300+((i)*20)
    
    print('['+str(startstim)+', '+str(stopstim)+']')
    

    pre = spiketrains[-1][(np.where((spiketrains[-1] > startstim) & (spiketrains[-1] < stopstim)))]


    post = spiketrains[32][(np.where((spiketrains[32] > startstim) & (spiketrains[32] < stopstim)))]
    srp = []
    srpre = []

    postint = post.astype(int)
    preint = pre.astype(int)
    for j in range(int(max(spiketrains[32]))):
        srp.append(np.count_nonzero(postint==j))        
    for j in range(int(max(spiketrains[-1]))):
        srpre.append(np.count_nonzero(preint==j))
    
    preint = pre*1000
    postint = post*1000
    
    postint = postint.astype(int)
    preint = preint.astype(int)


    start = min(preint[0],postint[0])
    end = max(preint[-1],postint[-1])
    binsize = 1
    bins = int((end-start)/binsize)
    timesteps = np.linspace(start,end-binsize,bins)
    s1,s2 = [],[]
#s1,s2 = np.zeros(int(bins/2)),np.zeros(int(bins/2))
    for k in range(0,bins,2):
        if (timesteps[k] in preint) or (timesteps[k]+1 in preint):#: or timesteps[i]+1 in st2pos):
            s1.append(1)
        else:
            s1.append(0)
        if (timesteps[k] in postint) or (timesteps[k]+1 in postint):# or timesteps[i]+1 in st5pos):
            s2.append(1)
        else:
            s2.append(0)
    
    np.save('Pre20secLsStim_'+str(i+1),s1)
    np.save('Post20secLsStim_'+str(i+1),s2)

'''

maxlag = 10
lags = np.linspace(-maxlag,maxlag,2*maxlag+1)
#ccov = plt.xcorr(s1 - s1.mean(), s2 - s2.mean(),maxlags=10,normed=True)
#ccor = (ccov[1]) / (len(s1) * s1.std() * s2.std())
ci = cicc(lags,0.99,len(s1))

plt.figure()
#plt.title('Cross-correlation candidate neuron pair (Stimulated period)')#+str(interesting[i]))
plt.xticks([-10,-8,-6,-4,-2,0,2,4,6,8,10],labels=['-10','-8','-6','-4','-2','0','2','4','6','8','10'])
plt.xcorr(np.asarray(s1) - np.asarray(s1).mean(), np.asarray(s2) - np.asarray(s2).mean(),maxlags=10,normed=True)
plt.plot(lags,ci[1],'r--',label='99% CI under $H_0$')
plt.plot(lags,ci[0],'r--')#,label='99% CI under $H_0$')
#plt.ylim((-0.035,0.035))
#plt.xticks(x,labels = ms)
plt.xlabel('Timelag (ms)')
plt.legend(loc=1)
plt.show()

'''
plt.figure()
plt.title('Observed firing rate presynaptic (stimulated) neuron')
plt.xlabel('Time [s]')
plt.ylabel('Firing rate [spikes / sec]')
plt.plot(np.linspace(1,len(srpre),len(srpre)),srpre,label = 'Observed rate')
for i in range(len(ses1stim)):
    if i == 0:
        plt.plot(np.linspace(ses1stim[i][0],ses1stim[i][1],100),np.ones(100),'rx',label = 'Stimulation')
    else:
        plt.plot(np.linspace(ses1stim[i][0],ses1stim[i][1],100),np.ones(100),'rx')
plt.legend()
plt.show()


plt.figure()
plt.title('Observed firing rate postsynaptic neuron')
plt.xlabel('Time [s]')
plt.ylabel('Firing rate [spikes / sec]')
plt.plot(np.linspace(1,len(srp),len(srp)),srp,label = 'Observed rate')
#for i in range(len(ses1stim)):
#    if i == 0:
#        plt.plot(np.linspace(ses1stim[i][0],ses1stim[i][1],100),np.ones(100),'ro',label = 'Stimulation')
#    else:
#        plt.plot(np.linspace(ses1stim[i][0],ses1stim[i][1],100),np.ones(100),'ro')
plt.show()
'''
'''
lineSizes= [0.4, 0.4]
plt.figure()
plt.title('Spike Trains of selected neurons - pre v index '+str(i))
plt.xlabel('Time (seconds)')
plt.xlim([startstim,stopstim])
plt.ylim([-0.5,1.9])
plt.ylabel('Neuron')
plt.eventplot([pre,post],linelengths=lineSize)#,colors= colorCodes)
#plt.axvline(3914,color='g',linestyle='--',alpha=0.8)
#plt.axvline(4034,color='g',linestyle='--',alpha=0.8,label='Possible stimulation time')
plt.legend(loc=('upper center'))
plt.show()

for i in range(len(spiketrains)-1):
    
    post = spiketrains[i][(np.where((spiketrains[i] > startstim) & (spiketrains[i] < stopstim)))]

    lineSize = [0.4, 0.4]
    plt.figure()
    plt.title('Spike Trains of selected neurons - pre v index '+str(i))
    plt.xlabel('Time (seconds)')
    plt.xlim([startstim,stopstim])
    plt.ylim([-0.5,1.9])
    plt.ylabel('Neuron')
    plt.eventplot([pre,post],linelengths=lineSize)#,colors= colorCodes)
#plt.axvline(3914,color='g',linestyle='--',alpha=0.8)
#plt.axvline(4034,color='g',linestyle='--',alpha=0.8,label='Possible stimulation time')
    plt.legend(loc=('upper center'))
    plt.show()

interesting = [11]

for i in range(len(interesting)):
    post = spiketrains[interesting[i]][(np.where((spiketrains[interesting[i]] > startstim) & (spiketrains[interesting[i]] < stopstim)))]

    prems = pre*1000

    postms = post * 1000

    prems = prems.astype(int)

    postms = postms.astype(int)


    
    start = min(prems[0],postms[0])
    end = max(prems[-1],postms[-1])
    binsize = 1
    bins = int((end-start)/binsize)
    timesteps = np.linspace(start,end-binsize,bins)

    s1,s2 = np.zeros(bins),np.zeros(bins)
    for j in range(bins):
        if (timesteps[j] in prems):#: or timesteps[i]+1 in st2pos):
            s1[j] = 1
        if (timesteps[j] in postms):# or timesteps[i]+1 in st5pos):
            s2[j] = 1
        
    maxlag = 10
    lags = np.linspace(-maxlag,maxlag,2*maxlag+1)
    #ccov = plt.xcorr(s1 - s1.mean(), s2 - s2.mean(),maxlags=10,normed=True)
    #ccor = (ccov[1]) / (len(s1) * s1.std() * s2.std())
    ci = cicc(lags,0.99,len(s1))

    plt.figure()
    plt.title('Cross-correlation candidate neuron pair (Stimulated period)')#+str(interesting[i]))
    plt.xcorr(s1 - s1.mean(), s2 - s2.mean(),maxlags=10,normed=True)
    plt.plot(lags,ci[1],'r--',label='99% CI under $H_0$')
    plt.plot(lags,ci[0],'r--')#,label='99% CI under $H_0$')
    #plt.ylim((-0.035,0.035))
    #plt.xticks(x,labels = ms)
    plt.xlabel('Timelag (ms)')
    plt.legend(loc=1)
    plt.show()

for j in range(51):
    st1 = np.concatenate(data18105['spikes']['times'][j])
    stim = stiminfo["stim"]["ts"][1]
    rates_stim = []
    rates_nonstim = []

    for i in range(len(stim[0])):
        rates_stim.append(np.count_nonzero(((st1 > stim[0][i][0]) & (st1 < stim[0][i][1]+1))==True)/(stim[0][0][1]+1-stim[0][0][0]))
        a = np.random.randint(2500,6500)
        b = np.random.randint(11500,16500)
        c = np.random.choice([a,b])
        #print(c)
        rates_nonstim.append(np.count_nonzero(((st1 > c) & (st1 < c+1))==True))


    rates_stim_mean = np.mean(rates_stim)
    rates_nonstim_mean = np.mean(rates_nonstim)
    rates_stim_var = np.sqrt(np.var(rates_stim))
    rates_nonstim_var = np.sqrt(np.var(rates_nonstim))
    
    means = [rates_stim_mean,rates_nonstim_mean]
    stds = [rates_stim_var,rates_nonstim_var]
    x = [1,2]
    ticksss = ['Stimulated','Non-stimulated']
    plt.figure()
    plt.title('Spikerates')
    plt.ylabel('Rate')
    #plt.ylim([0,0.025])
    plt.xlim([0,3])
    plt.xticks(x,labels = ticksss)
    for i in range(2):
        plt.errorbar(x[i], means[i], yerr = stds[i],marker = 'o')
        #plt.axhline(0.005,color='r',linestyle='--',label='True Value')
    plt.legend()
    plt.show()

for j in range(51):
    st1 = np.concatenate(data18105['spikes']['times'][j])
    st1 = st1.astype(int)+1
    start = int(st1[0])+1
    end = int(st1[-1])+1
    binsize = 1
    bins = int((end-start)/binsize)
    timesteps = np.linspace(start,end-binsize,bins)

    s1 = np.zeros(bins)
    for i in range(bins):
        s1[i] = np.count_nonzero(st1 == i+1)


    plt.figure()
    plt.title('Neuron firing rate')
    plt.xlabel('Time (seconds)')
    plt.ylabel('#Spikes last second')
    plt.plot(np.linspace(1,end,bins),s1)
    for i in range(len(stim[0])):
        if i == 0:
            plt.plot(stim[0][i][0],1,'ro',label='stimuli')
        else:
            plt.plot(stim[0][i][0],1,'ro')
    plt.legend()
    plt.show()

data18102 = loadmat('camkii10_180102.spikes.cellinfo.mat')
data18104 = loadmat('camkii10_180104.spikes.cellinfo.mat')
data18220 = loadmat('camkii13_181220.spikes.cellinfo.mat')
data18103 = loadmat('camkii10_180103.spikes.cellinfo.mat')
data18105 = loadmat('camkii10_180105.spikes.cellinfo.mat')
data18125 = loadmat('camkii10_180125.spikes.cellinfo.mat')
data18126 = loadmat('camkii10_180126.spikes.cellinfo.mat')
data18221 = loadmat('camkii13_181221.spikes.cellinfo.mat')
data18228 = loadmat('camkii13_181228.spikes.cellinfo.mat')
data18229 = loadmat('camkii13_181229.spikes.cellinfo.mat')
data18230 = loadmat('camkii13_181230.spikes.cellinfo.mat')
data18231 = loadmat('camkii13_181231.spikes.cellinfo.mat')
#data3= loadmat('camkii10_180123_CellParams.mat')
data19109 = loadmat('camkii13_190109.spikes.cellinfo.mat')
data19110 = loadmat('camkii13_190110.spikes.cellinfo.mat')
data19112 = loadmat('camkii13_190112.spikes.cellinfo.mat')
data19113 = loadmat('camkii13_190113.spikes.cellinfo.mat')
data19114 = loadmat('camkii13_190114.spikes.cellinfo.mat')
data19115 = loadmat('camkii13_190115.spikes.cellinfo.mat')

#data18102 = loadmat('camkii10_180102.spikes.cellinfo.mat')
#st1 = np.concatenate(data18102['spikes']['times'][13])
#st2 = np.concatenate(data18102['spikes']['times'][1])
#st3 = np.concatenate(data18102['spikes']['times'][14])
#st4 = np.concatenate(data18102['spikes']['times'][8])
#st5 = np.concatenate(data18102['spikes']['times'][9])
#st6 = np.concatenate(data18102['spikes']['times'][21])
#st1 = st1.astype(int)
#st2 = st2.astype(int)

#data19109 = loadmat('camkii13_190109.spikes.cellinfo.mat')
#st1 = np.concatenate(data19109['spikes']['times'][0])
#st2 = np.concatenate(data19109['spikes']['times'][4])
#st3 = np.concatenate(data19109['spikes']['times'][12])
#st4 = np.concatenate(data19109['spikes']['times'][14])
#st5 = np.concatenate(data19109['spikes']['times'][16])
#st6 = np.concatenate(data19109['spikes']['times'][19])


#st1 = np.concatenate(data19110['spikes']['times'][3]).round(1)
#st2 = np.concatenate(data19110['spikes']['times'][11]).round(1)
#st3 = np.concatenate(data19110['spikes']['times'][23]).round(1)
#st4 = np.concatenate(data19110['spikes']['times'][24]).round(1)

#st1 = np.concatenate(data19113['spikes']['times'][2])
#st2 = np.concatenate(data19113['spikes']['times'][3])
#st3 = np.concatenate(data19113['spikes']['times'][5])
#st4 = np.concatenate(data19113['spikes']['times'][21])
#st5 = np.concatenate(data19113['spikes']['times'][24])
#st6 = np.concatenate(data19113['spikes']['times'][11])

#st1 = np.concatenate(data18102['spikes']['times'][0])
#st2 = np.concatenate(data18102['spikes']['times'][1])
#st3 = np.concatenate(data18102['spikes']['times'][8])
#st4 = np.concatenate(data18102['spikes']['times'][9])
#st5 = np.concatenate(data18102['spikes']['times'][10])#*1000
#st6 = np.concatenate(data18102['spikes']['times'][14])
#st7 = np.concatenate(data18102['spikes']['times'][15])
#st8 = np.concatenate(data18102['spikes']['times'][20])
#st9 = np.concatenate(data18102['spikes']['times'][21])#*1000

#st1 = np.concatenate(data18231['spikes']['times'][2])
#st2 = np.concatenate(data18231['spikes']['times'][4])
#st3 = np.concatenate(data18231['spikes']['times'][7])
#st4 = np.concatenate(data18231['spikes']['times'][12])
#st5 = np.concatenate(data18231['spikes']['times'][15])
#st6 = np.concatenate(data18231['spikes']['times'][16]).round(1)
#st6 = np.concatenate(data18231['spikes']['times'][17])
#st8 = np.concatenate(data18231['spikes']['times'][20]).round(1)
#st9 = np.concatenate(data18231['spikes']['times'][21]).round(1)
#st10 = np.concatenate(data18231['spikes']['times'][30]).round(1)
#st11 = np.concatenate(data18231['spikes']['times'][32]).round(1)
#spiketrains = [st1[7769:9548],st2[1625:1686],st3[1662:1785],st4[1527:1557]]#,st5,st6]

#end = max(st1[-1],st2[-1])#,st7[-1],st8[-1],st9[-1],st10[-1],st11[-1]))

#st5 = st5.round()
#st9 = st9.round()
data18231 = loadmat('camkii13_181231.spikes.cellinfo.mat')

st2 = np.concatenate(data18231['spikes']['times'][8])
st5 = np.concatenate(data18231['spikes']['times'][10])
st1 = np.concatenate(data18231['spikes']['times'][1])
st3 = np.concatenate(data18231['spikes']['times'][9])
st4 = np.concatenate(data18231['spikes']['times'][0])
st6 = np.concatenate(data18231['spikes']['times'][23])

st2pos = st2[(np.where((st2 > 3900) & (st2 < 4035)))]

st5pos = st5[(np.where((st5 > 3900) & (st5 < 4035)))]



st2pos = st2pos*1000

st5pos = st5pos * 1000

st2pos = st2pos.astype(int)

st5pos = st5pos.astype(int)

def cicc(lags,significance,n):
    return stats.norm.interval(significance,0,np.sqrt(1/(n-abs(lags))))


start = min(st2pos[0],st5pos[0])
end = max(st2pos[-1],st5pos[-1])
binsize = 1
bins = int((end-start)/binsize)
timesteps = np.linspace(start,end-binsize,bins)

s1,s2 = np.zeros(bins),np.zeros(bins)
for i in range(bins):
    if (timesteps[i] in st2pos):#: or timesteps[i]+1 in st2pos):
        s1[i] = 1
    if (timesteps[i] in st5pos):# or timesteps[i]+1 in st5pos):
        s2[i] = 1

#    if timesteps[i] in st3:
#        s3[i] = 1
#    if timesteps[i] in st4:
#        s4[i] = 1
#    if timesteps[i] in st5:
#        s5[i] = 1
#    if timesteps[i] in st6:
#        s6[i] = 1
    #if timesteps[i] in st7:
    #    s7[i] = 1
    #if timesteps[i] in st8:
    #    s8[i] = 1
    #if timesteps[i] in st9:
    #    s9[i] = 1
    #if timesteps[i] in st10:
    #    s10[i] = 1
    #if timesteps[i] in st11:
    #    s11[i] = 1

maxlag = 10
lags = np.linspace(-maxlag,maxlag,2*maxlag+1)
#ccov = plt.xcorr(s1 - s1.mean(), s2 - s2.mean(),maxlags=10,normed=True)
#ccor = (ccov[1]) / (len(s1) * s1.std() * s2.std())
ci = cicc(lags,0.99,len(s1))

plt.figure()
plt.title('Cross-correlation for the first 10 seconds')
plt.xcorr(s1 - s1.mean(), s2 - s2.mean(),maxlags=10,normed=True)
plt.plot(lags,ci[1],'r--',label='99% CI under $H_0$')
plt.plot(lags,ci[0],'r--')#,label='99% CI under $H_0$')
plt.ylim((-0.035,0.035))
#plt.xticks(x,labels = ms)
plt.xlabel('Timelag (ms)')
plt.legend(loc=1,fancybox = True)
plt.show()
'''
'''
startsec = 3880
endsec = 4050
spiketrains = [st1[(np.where((st1 > startsec) & (st1 < endsec)))],st2[(np.where((st2 > startsec) & (st2 < endsec)))]\
               ,st3[(np.where((st3 > startsec) & (st3 < endsec)))],st4[(np.where((st4 > startsec) & (st4 < endsec)))]\
                   ,st5[(np.where((st5 > startsec) & (st5 < endsec)))],st6[(np.where((st6 > startsec) & (st6 < endsec)))]]
lineSize = [0.4, 0.4, 0.4, 0.4, 0.4,0.4]
spiketrain_halfs = []
for i in range(len(spiketrains)):
    temp = np.zeros(int(len(spiketrains[i])/4)+1)
    for j in range(len(spiketrains[i])):
        if j%4 == 0:
            temp[int(j/4)] = spiketrains[i][j]
    spiketrain_halfs.append(temp)
    
    
colorCodes = np.array([[0.3, 0.3, 0.4],
                        [0, 0, 1],
                       [0.3, 0.3, 0.4],
                        [0.3, 0.3, 0.4],
                        [0, 0, 1],
                        [0.3, 0.3, 0.4]])
plt.figure()
plt.title('Spike Trains of selected neurons')
plt.xlabel('Time (seconds)')
plt.xlim([startsec,endsec])
plt.ylim([-0.5,5.9])
plt.ylabel('Neuron')
plt.eventplot(spiketrain_halfs,linelengths=lineSize,colors= colorCodes)
plt.axvline(3914,color='g',linestyle='--',alpha=0.8)
plt.axvline(4034,color='g',linestyle='--',alpha=0.8,label='Possible stimulation time')
plt.legend(loc=('upper center'))
plt.show()
'''
'''
for i in range(4000,6000,100):
    startsec = i
    endsec = i + 100
    spiketrains = [st1[(np.where((st1 > startsec) & (st1 < endsec)))],st2[(np.where((st2 > startsec) & (st2 < endsec)))]\
                   ,st3[(np.where((st3 > startsec) & (st3 < endsec)))],st4[(np.where((st4 > startsec) & (st4 < endsec)))]\
                       ,st5[(np.where((st5 > startsec) & (st5 < endsec)))],st6[(np.where((st6 > startsec) & (st6 < endsec)))]]
    lineSize = [0.4, 0.4, 0.4, 0.4, 0.4,0.4]
    plt.figure()
    plt.title('Spike Trains of selected neurons')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Neuron')
    plt.eventplot(spiketrains,linelengths=lineSize)#,colors= colorCodes)
    plt.show()
colorCodes = np.array([[0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 0],
                        [0, 1, 1]])
lineSize = [0.4, 0.4, 0.4, 0.4, 0.4,0.4]
'''

'''
Potential interesting connections so far?
Data18102: ind 1v9 (between seconds 1040 and 1160) PossiblePair?
Data18231 ind 0v10 between seconds 1900 and 2020 or 2660-2800 PossiblePair2?
Data18231 ind 8v10 between 3900-4035s PossiblePair3?
Data1909 : ind0v4 1360-1480s
'''

'''
Ekempler på IKKE korrelerte:
Data19109 index 2v5
Data18231: index 7v17, index 15v16
'''
'''
# DATA: 18231 ind 8v10!
test1 = st3[(np.where((st3 > 3900) & (st3 < 4035)))]
test2 = st4[(np.where((st4 > 3900) & (st4 < 4035)))]
test1halv = np.zeros(int(len(test1)/4)+1)
test2halv = np.zeros(int(len(test2)/4))
for i in range(len(test1)):
    if i%4 == 0:
        test1halv[int(i/4)] = test1[i]
        
for i in range(len(test2)):
    if i%4 == 0:
        test2halv[int(i/4)] = test2[i]
testspikes = [test1halv,test2halv]
plt.figure()
plt.title('Spike Trains of selected neurons')
plt.xlabel('Time (seconds)')
plt.ylabel('Neuron')
plt.axvline(3970,color='r',linestyle='--',label='Possible stimulus')
plt.eventplot(testspikes,linelengths=[0.2,0.2])#,colors= colorCodes)
plt.yticks([0,1],labels=['1','2'])
plt.legend()
plt.show()
'''  


