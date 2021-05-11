import numpy as np              
import matplotlib.pyplot as plt 
import math
#from tqdm import tqdm
from scipy.stats import gamma
from scipy.stats import multivariate_normal
from numba import njit
@njit

def learning_rule(s1,s2,Ap,Am,taup,taum,t,i,binsize): 
    '''
    s1,s2 : binary values for the different time bins for neuron 1 and 2 respectively, 1:spike, 0:no spike
    i : current iteration/timebin for the numerical approximation
    '''
    l = i - np.int(np.ceil(10*taup / binsize))
    return s2[i-1]*np.sum(s1[max([l,0]):i]*Ap*np.exp((t[max([l,0]):i]-max(t))/taup)) - s1[i-1]*np.sum(s2[max([l,0]):i]*Am*np.exp((t[max([l,0]):i]-max(t))/taum))

def logit(x):
    return np.log(x/(1-x))

def inverse_logit(x):
    return np.exp(x)/(1+np.exp(x))


class SimulatedData():
    '''
    Ap, Am, tau : learning rule parameters
    b1,b2 : background noise constants for neuron 1 and neuron 2, determnining their baseline firing rate
    w0 : start value for synapse strength between neuron 1 and 2. 
    '''
    def __init__(self,Ap=0.005, tau=0.02, std=0.001,b1=-2.0, b2=-2.0, w0=1.0,sec = 120, binsize = 1/200.0,freq = 50):
        self.Ap = Ap
        self.tau = tau
        self.std = std
        self.Am = 1.05*self.Ap
        self.b1 = b1
        self.b2 = b2
        self.w0 = w0
        self.sec = sec
        self.binsize = binsize
        self.freq = freq
    
    def set_Ap(self,Ap):
        self.Ap = Ap
    def set_tau(self,tau):
        self.tau = tau
    def set_std(self,std):
        self.std = std
    def set_b1(self,b1):
        self.b1 = b1
    def set_b2(self,b2):
        self.b2 = b2
    def set_w0(self,w0):
        self.w0 = w0
    def set_sec(self,sec):
        self.sec = sec
    def set_binsize(self,binsize):
        self.binsize = binsize
    
    def get_Ap(self):
        return self.Ap
    def get_tau(self):
        return self.tau
    def get_std(self):
        return self.std
    def get_b1(self):
        return self.b1
    def get_b2(self):
        return self.b2
    def get_w0(self):
        return self.w0
    def get_sec(self):
        return self.sec
    def get_binsize(self):
        return self.binsize
        
    def create_data(self):
        iterations = np.int(self.sec/self.binsize)
        t,W,s1,s2 = np.zeros(iterations),np.zeros(iterations),np.zeros(iterations),np.zeros(iterations)
        W[0] = self.w0
        s1[0] = np.random.binomial(1,inverse_logit(self.b1))
        for i in range(1,iterations):
            lr = learning_rule(s1,s2,self.Ap,self.Am,self.tau,self.tau,t,i,self.binsize)
            W[i] = W[i-1] + lr + np.random.normal(0,self.std) 
            s2[i] = np.random.binomial(1,inverse_logit(W[i]*s1[i-1]+self.b2))
            s1[i] = np.random.binomial(1,inverse_logit(self.b1)) 
            t[i] = self.binsize*i
        self.s1 = s1 
        self.s2 = s2
        self.t = t
        self.W = W
        
    def create_freq_data(self):
        iterations = np.int(self.sec/self.binsize)
        t,W,s1,s2 = np.zeros(iterations),np.zeros(iterations),np.zeros(iterations),np.zeros(iterations)
        W[0] = self.w0
        s1[0] = 1
        for i in range(1,iterations):
            lr = learning_rule(s1,s2,self.Ap,self.Am,self.tau,self.tau,t,i,self.binsize)
            W[i] = W[i-1] + lr + np.random.normal(0,self.std) 
            s2[i] = np.random.binomial(1,inverse_logit(W[i]*s1[i-1]+self.b2))
            s1[i] = [np.random.binomial(1,inverse_logit(self.b1)),1][i % int((1/self.binsize)/self.freq) == 0]
            t[i] = self.binsize*i
        self.s1 = s1
        self.s2 = s2
        self.t = t
        self.W = W
    
    def get_data(self):
        return self.s1,self.s2,self.t,self.W
        
    def plot_weight_trajectory(self):
        plt.figure()
        plt.title('Weight trajectory - 250 freq')
        plt.plot(self.t,self.W)
        plt.xlabel('Time')
        plt.ylabel('Weight')
        plt.show()
        
class ParameterInference():
    '''
    Class for estimating b1,b2,w0,Ap,Am,tau from SimulatedData, given data s1,s2.
    '''
    def __init__(self,s1,s2,P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                 , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=120\
                     ,binsize = 1/200.0,taufix = 0.02,Afix = 0.005,b1est = -3.1,b2est = -3.1,w0est = 1):
        self.s1 = s1
        self.s2 = s2
        self.infstd = infstd
        self.P = P
        self.Usim = Usim
        self.Ualt = Ualt
        self.it = it
        self.N = N
        self.shapes_prior = shapes_prior
        self.rates_prior = rates_prior
        self.sec = sec
        self.binsize = binsize
        self.Afix = Afix
        self.taufix = taufix
        self.b1est = b1est
        self.b2est = b2est
        self.w0est = w0est
    
    def get_sec(self):
        return self.sec
    
    def set_std(self,std):
        self.std = std
    def set_P(self,P):
        self.P = P
    def set_s1(self,s1):
        self.s1 = s1
    def set_s2(self,s2):
        self.s2 = s2
    def set_sec(self,sec):
        self.sec = sec
    def set_shapes_prior(self,shapes_prior):
        self.shapes_prior = shapes_prior
    def set_rates_prior(self,rates_prior):
        self.rates_prior = rates_prior
    def set_w0est(self,w0est):
        self.w0est=w0est
    
    
    def b1_estimation(self):
        self.b1est = logit(np.sum(self.s1)/len(self.s1))
        return self.b1est

    def normalize(self,vp):
        return vp/np.sum(vp)

    def perplexity_func(self,vp_normalized):
        h = -np.sum(vp_normalized*np.log(vp_normalized))
        return np.exp(h)/self.P

    def resampling(self,vp_normalized,wp):
        wp_new = np.copy(wp)
        indexes = np.linspace(0,self.P-1,self.P)
        resampling_indexes = np.random.choice(indexes,self.P,p=vp_normalized)
        for i in range(self.P):
            wp_new[i] = np.copy(wp[resampling_indexes.astype(int)[i]])
        return wp_new

    def likelihood_step(self,s1prev,s2next,wcurr):
        return inverse_logit(wcurr*s1prev + self.b2est)**(s2next) * (1-inverse_logit(wcurr*s1prev + self.b2est))**(1-s2next)
    
    def parameter_priors(self):
        #print(self.shapes_prior)
        #print(self.rates_prior)
        return np.array([(np.random.gamma(self.shapes_prior[i],1/self.rates_prior[i])) for i in range(self.N)])
    
    def parameter_priors_ns(self,mean,cov):
        return multivariate_normal.rvs(mean,cov,1)
    
    def proposal_step(self,shapes,theta):
        return np.array([(np.random.gamma(shapes[i],theta[i]/shapes[i])) for i in range(self.N)])
    
    def adjust_variance(self,theta,shapes):
        means = theta[-self.Usim:].mean(0)
        var_new = np.array([0,0])
        u_temp = self.Usim
        while (any(i == 0 for i in var_new)):
            var_new = theta[-u_temp:].var(0)*(2.4**2)
            #print(var_new)
            u_temp += 50
            if u_temp > self.it:
                return shapes, np.array([(np.random.gamma(shapes[i],theta[-1][i]/shapes[i])) for i in range(self.N)])
        #print(var_new)
        #print('made it!')
        new_shapes = np.array([((means[i]**2) / var_new[i]) for i in range(self.N)])
        proposal = np.array([(np.random.gamma(new_shapes[i],theta[-1][i]/new_shapes[i])) for i in range(self.N)])
        return new_shapes,proposal
    
    def ratio(self,prob_old,prob_next,shapes,theta_next,theta_prior):
        spike_prob_ratio = prob_next / prob_old
        prior_ratio, proposal_ratio = 1,1
        for i in range(self.N):
            prior_ratio *= gamma.pdf(theta_next[i],a=self.shapes_prior[i],scale=1/self.rates_prior[i])/\
            gamma.pdf(theta_prior[i],a=self.shapes_prior[i],scale=1/self.rates_prior[i])
            proposal_ratio *= gamma.pdf(theta_prior[i],a=shapes[i],scale=theta_next[i]/shapes[i])/\
            gamma.pdf(theta_next[i],a=shapes[i],scale=theta_prior[i]/shapes[i])
        return spike_prob_ratio * prior_ratio * proposal_ratio
    
    def ratio_g(self,prob_old,prob_next,shapes,theta_next,theta_prior,mean,cov):
        spike_prob_ratio = prob_next / prob_old
        #print('spike_prob_ratio:',spike_prob_ratio)
        #print(spike_prob_ratio)
        proposal_ratio = 1
        prior_ratio = multivariate_normal.pdf(theta_next,mean,cov) / multivariate_normal.pdf(theta_prior,mean,cov)
        #print('next prior:', multivariate_normal.pdf(theta_next,mean,cov))
        #print('prev prior:', multivariate_normal.pdf(theta_prior,mean,cov))
        #print('prior ratio:', prior_ratio)
        for i in range(self.N):
            proposal_ratio *= gamma.pdf(theta_prior[i],a=shapes[i],scale=theta_next[i]/shapes[i])/\
            gamma.pdf(theta_next[i],a=shapes[i],scale=theta_prior[i]/shapes[i])
        return spike_prob_ratio * prior_ratio * proposal_ratio


    def scaled2_spike_prob(self,old,new):
        return np.exp(old - min(old,new)),np.exp(new - min(old,new))
    
    def b2_w0_estimation(self):
        '''
        Fisher scoring algorithm 
        Two in parallell, since w0 is estimated with a subset of the data
        '''
        s1short,s2short = self.s1[:int(10/(self.binsize))],self.s2[:int(10/(self.binsize))]
        beta,beta2 = np.array([0,0]),np.array([0,0])
        x,x2 = np.array([np.ones(len(self.s1)-1),self.s1[:-1]]),np.array([np.ones(len(s1short)-1),s1short[:-1]])
        i = 0
        score,score2 = np.array([np.inf,np.inf]), np.array([np.inf,np.inf])
        while(i < 1000 and any(abs(i) > 1e-10 for i in score) and any(abs(j) > 1e-10 for j in score2)):
            eta,eta2 = np.matmul(beta,x),np.matmul(beta2,x2) #linear predictor
            mu,mu2 = inverse_logit(eta),inverse_logit(eta2)
            score,score2 = np.matmul(x,self.s2[1:] - mu),np.matmul(x2,s2short[1:] - mu2)
            hessian_u,hessian_u2 = mu * (1-mu), mu2 *(1-mu2)
            hessian,hessian2 = np.matmul(x*hessian_u,np.transpose(x)),np.matmul(x2*hessian_u2,np.transpose(x2))
            delta,delta2 = np.matmul(np.linalg.inv(hessian),score),np.matmul(np.linalg.inv(hessian2),score2)
            beta,beta2 = beta + delta, beta2 + delta2
            i += 1
        self.b2est = beta[0]
        self.w0est = beta2[1]
        return self.b2est,self.w0est


    def particle_filter(self,A,tau):
        '''
        Particle filtering, (doesnt quite work yet, smth with weights vp)
        Possible to speed it up? 
        How to initiate w0 and vp?
        '''
        timesteps = np.int(self.sec/self.binsize)
        t = np.zeros(timesteps)
        wp = np.full((self.P,timesteps),np.float(self.w0est))
        vp = np.ones(self.P)
        log_posterior = 0
        for i in range(1,timesteps):
            v_normalized = self.normalize(vp)
            perplexity = self.perplexity_func(v_normalized)
            if perplexity < 0.66:
                wp = self.resampling(v_normalized,wp)
                vp = np.full(self.P,1/self.P)
                v_normalized = self.normalize(vp)
            lr = learning_rule(self.s1,self.s2,A,A*1.05,tau,tau,t,i,self.binsize) 
            ls = self.likelihood_step(self.s1[i-1],self.s2[i],wp[:,i-1])  
            vp = ls*v_normalized
            wp[:,i] = wp[:,i-1] + lr + np.random.normal(0,self.infstd,size = self.P)
            t[i] = i*self.binsize
            log_posterior += np.log(np.sum(vp)/self.P)
        #print(log_posterior)
        return wp,t,log_posterior
    
    def particle_filter_noise(self,A,tau,std):
        '''
        Particle filtering, (doesnt quite work yet, smth with weights vp)
        Possible to speed it up? 
        How to initiate w0 and vp?
        '''
        timesteps = np.int(self.sec/self.binsize)
        t = np.zeros(timesteps)
        wp = np.full((self.P,timesteps),np.float(self.w0est))
        vp = np.ones(self.P)
        log_posterior = 0
        for i in range(1,timesteps):
            v_normalized = self.normalize(vp)
            perplexity = self.perplexity_func(v_normalized)
            if perplexity < 0.66:
                wp = self.resampling(v_normalized,wp)
                vp = np.full(self.P,1/self.P)
                v_normalized = self.normalize(vp)
            lr = learning_rule(self.s1,self.s2,A,A*1.05,tau,tau,t,i,self.binsize) 
            ls = self.likelihood_step(self.s1[i-1],self.s2[i],wp[:,i-1])  
            vp = ls*v_normalized
            wp[:,i] = wp[:,i-1] + lr + np.random.normal(0,std,size = self.P)
            t[i] = i*self.binsize
            log_posterior += np.log(np.sum(vp)/self.P)
        return wp,t,log_posterior
    
    

    def standardMH(self):#,w0est,b1,b2):
        '''
        Monte Carlo sampling with particle filtering, Metropolis Hastings algorithm
        '''
        #self.w0est = w0est
        #self.b1est = b1
        #self.b2est = b2
        theta_prior = self.parameter_priors()
        #print(theta_prior)
        theta = np.array([theta_prior])
        shapes = np.copy(self.shapes_prior)
        _,_,old_log_post = self.particle_filter(theta_prior[0],theta_prior[1])
        for i in range(1,self.it):
            if (i % self.Usim == 0):
                shapes, theta_next = self.adjust_variance(theta,shapes)
            else:    
                theta_next = self.proposal_step(shapes,theta_prior)
            _,_,new_log_post = self.particle_filter(theta_next[0],theta_next[1])
            prob_old,prob_next = self.scaled2_spike_prob(old_log_post,new_log_post)
            r = self.ratio(prob_old,prob_next,shapes,theta_next,theta_prior)
            choice = np.int(np.random.choice([1,0], 1, p=[min(1,r),1-min(1,r)]))
            theta_choice = [np.copy(theta_prior),np.copy(theta_next)][choice == 1]
            #print('it:',i)
            #print('prior:',theta_prior)
            #print('next:',theta_next)
            #print('choice:',theta_choice)
            theta = np.vstack((theta, theta_choice))
            theta_prior = np.copy(theta_choice)
            old_log_post = [np.copy(old_log_post),np.copy(new_log_post)][choice == 1]
        return theta
        #self.smh_sample = theta
    
    #def get_standardMHsample(self):
    #    return self.smh_sample
    
    def standardMH_taufix(self):
        '''
        Monte Carlo sampling with particle filtering, Metropolis Hastings algorithm
        '''
        theta_prior = np.array([0.001])#self.parameter_priors()
        theta = np.array([theta_prior])
        shapes = np.copy(self.shapes_prior)
        _,_,old_log_post = self.particle_filter(theta_prior[0],self.taufix)
        for i in range(1,self.it):
            if (i % self.Usim == 0):
                shapes, theta_next = self.adjust_variance(theta,shapes)
            else:    
                theta_next = self.proposal_step(shapes,theta_prior)
            _,_,new_log_post = self.particle_filter(theta_next[0],self.taufix)
            prob_old,prob_next = self.scaled2_spike_prob(old_log_post,new_log_post)
            r = self.ratio(prob_old,prob_next,shapes,theta_next,theta_prior)
            choice = np.int(np.random.choice([1,0], 1, p=[min(1,r),1-min(1,r)]))
            theta_choice = [np.copy(theta_prior),np.copy(theta_next)][choice == 1]
            #print('prior:',theta_prior)
            #print('next:',theta_next)
            #print('choice:',theta_choice)
            theta = np.vstack((theta, theta_choice))
            theta_prior = np.copy(theta_choice)
            old_log_post = [np.copy(old_log_post),np.copy(new_log_post)][choice == 1]
        return theta
    
    def standardMH_afix(self):
        '''
        Monte Carlo sampling with particle filtering, Metropolis Hastings algorithm
        '''
        theta_prior = np.array([0.005])#self.parameter_priors()
        theta = np.array([theta_prior])
        shapes = np.copy(self.shapes_prior)
        _,_,old_log_post = self.particle_filter(self.Afix,theta_prior[0])
        for i in range(1,self.it):
            if (i % self.Usim == 0):
                shapes, theta_next = self.adjust_variance(theta,shapes)
            else:    
                theta_next = self.proposal_step(shapes,theta_prior)
            _,_,new_log_post = self.particle_filter(self.Afix,theta_next[0])
            prob_old,prob_next = self.scaled2_spike_prob(old_log_post,new_log_post)
            r = self.ratio(prob_old,prob_next,shapes,theta_next,theta_prior)
            choice = np.int(np.random.choice([1,0], 1, p=[min(1,r),1-min(1,r)]))
            theta_choice = [np.copy(theta_prior),np.copy(theta_next)][choice == 1]
            #print('prior:',theta_prior)
            #print('next:',theta_next)
            #print('choice:',theta_choice)
            theta = np.vstack((theta, theta_choice))
            theta_prior = np.copy(theta_choice)
            old_log_post = [np.copy(old_log_post),np.copy(new_log_post)][choice == 1]
        return theta
    
    def standardMH_mv(self,mean,cov):#,ir):
        '''
        Monte Carlo sampling with particle filtering, Metropolis Hastings algorithm
        '''
        theta_prior = self.parameter_priors()
        #print('prior:',theta_prior)
        theta = np.array([theta_prior])
        shapes = np.copy(self.shapes_prior)
        _,_,old_log_post = self.particle_filter(theta_prior[0],theta_prior[1])
        for i in range(1,self.it):
            if (i % self.Usim == 0):
                shapes, theta_next = self.adjust_variance(theta,shapes)
            else:    
                theta_next = self.proposal_step(shapes,theta_prior)             
            _,_,new_log_post = self.particle_filter(theta_next[0],theta_next[1])
            #print(i)
            #if ir == 6:
            #    print('it:', i)
            #    print('old:',old_log_post)
            #    print('new:',new_log_post)
            prob_old,prob_next = self.scaled2_spike_prob(old_log_post,new_log_post)
            r = self.ratio_g(prob_old,prob_next,shapes,theta_next,theta_prior,mean,cov)
            choice = np.int(np.random.choice([1,0], 1, p=[min(1,r),1-min(1,r)]))
            theta_choice = [np.copy(theta_prior),np.copy(theta_next)][choice == 1]
            #print('prior:',theta_prior)
            #print('next:',theta_next)
            #print('choice:',theta_choice)
            theta = np.vstack((theta, theta_choice))
            theta_prior = np.copy(theta_choice)
            old_log_post = [np.copy(old_log_post),np.copy(new_log_post)][choice == 1]
            #if ir == 5:
            #    print(theta_choice)
            #return theta
        return theta
        
    
    def MH_noise(self):
        theta_prior = self.parameter_priors()
        theta = np.array([theta_prior])
        shapes = np.copy(self.shapes_prior)
        _,_,old_log_post = self.particle_filter_noise(self.Afix,self.taufix,theta_prior[0])
        for i in range(1,self.it):
            if (i % self.Usim == 0):
                shapes, theta_next = self.adjust_variance(theta,shapes)
            else:    
                theta_next = self.proposal_step(shapes,theta_prior)
            _,_,new_log_post = self.particle_filter_noise(self.Afix,self.taufix,theta_next[0])
            #print('old:', theta_prior)
            #print('new:', theta_next)
            prob_old,prob_next = self.scaled2_spike_prob(old_log_post,new_log_post)
            r = self.ratio(prob_old,prob_next,shapes,theta_next,theta_prior)
            #print('r:',r)
            choice = np.int(np.random.choice([1,0], 1, p=[min(1,r),1-min(1,r)]))
            theta_choice = [np.copy(theta_prior),np.copy(theta_next)][choice == 1]
            #print('choice:',theta_choice)
            theta = np.vstack((theta, theta_choice))
            theta_prior = np.copy(theta_choice)
            old_log_post = [np.copy(old_log_post),np.copy(new_log_post)][choice == 1]
        return theta
    
    
    def adjust_variance_alternating(self,theta,par_ind,shapes):
        mean = np.zeros(self.N)
        var_new = np.zeros(self.N)
        theta_new = np.copy(theta[-1])
        u_temp = self.Ualt
        while (any(i == 0 for i in var_new)):
            for i in range(self.N):
                if i == 0:
                    mean[i] = theta[-u_temp+1::2].mean(0)[i]
                    var_new[i] = theta[-u_temp+1::2].var(0)[i]*(2.4**2)
                elif i == 1:
                    mean[i] = theta[-u_temp::2].mean(0)[i]
                    var_new[i] = theta[-u_temp::2].var(0)[i]*(2.4**2)
                else:
                    mean[i] = theta[-u_temp:].mean(0)[i]
                    var_new[i] = theta[-u_temp:].var(0)[i]*(2.4**2)
            u_temp+= 100
            if u_temp > self.it:
                return shapes, np.array([(np.random.gamma(shapes[i],theta[-1][i]/shapes[i])) for i in range(self.N)])
        new_shapes = np.array([((mean[i]**2) / var_new[i]) for i in range(self.N)])
        for i in par_ind:
            theta_new[i] = np.random.gamma(new_shapes[i],theta[-1][i]/new_shapes[i])
        return new_shapes,theta_new

    
    def proposal_step_alternating(self,shapes,theta,par_ind):
        theta_new = np.copy(theta)
        for i in par_ind:
            theta_new[i] = np.random.gamma(shapes[i],theta[i]/shapes[i])
        return theta_new

    def alternatingMH(self):
        '''
        Alternating MH sampling
        '''
        theta_prior = np.array([0.001,0.005])#self.parameter_priors()
        theta = np.array([theta_prior])
        shapes = np.copy(self.shapes_prior)
        par_ind = np.linspace(0,self.N-1,self.N).astype(int)
        _,_,old_log_post = self.particle_filter(theta_prior[0],theta_prior[1])
        for i in range(1,self.it):
            ex = [1,0][i % 2 == 0]
            par_ind_temp = np.delete(par_ind,ex)
            if (i % self.Ualt == 0):
                shapes, theta_next = self.adjust_variance_alternating(theta,par_ind_temp,shapes)
            else:    
                theta_next = self.proposal_step_alternating(shapes,theta_prior,par_ind_temp)
            _,_,new_log_post = self.particle_filter(theta_next[0],theta_next[1])
            prob_old,prob_next = self.scaled2_spike_prob(old_log_post,new_log_post)
            r = self.ratio(prob_old,prob_next,shapes,theta_next,theta_prior)
            choice = np.int(np.random.choice([1,0], 1, p=[min(1,r),1-min(1,r)]))
            theta_choice = [np.copy(theta_prior),np.copy(theta_next)][choice == 1]
            #print('prior:',theta_prior)
            #print('next:',theta_next)
            #print('choice:',theta_choice)
            theta = np.vstack((theta, theta_choice))
            theta_prior = np.copy(theta_choice)
            old_log_post = [np.copy(old_log_post),np.copy(new_log_post)][choice == 1]
        return theta

### functions for optimal experimental design


class ExperimentDesign():
    def __init__(self,freqs_init=np.array([20,50,100,200]),maxtime=120,trialsize=5\
                 ,Ap=0.005, tau=0.02, genstd=0.0001,b1=-3.1, b2=-3.1, w0=1.0,binsize = 1/500.0,reals = 20, longinit = 60,s1init = 1,s2init =1,Winit =1):
        self.maxtime = maxtime
        self.freqs_init = freqs_init
        self.Ap = Ap
        self.tau = tau
        self.genstd = genstd
        self.trialsize = trialsize
        self.Am = 1.05*self.Ap
        self.binsize = binsize
        self.b1 = b1
        self.b2 = b2
        self.w0 = w0
        self.b2est = b2
        self.b1est = b1
        self.w0est = w0
        #self.s1 = np.zeros(np.int(self.maxtime/self.binsize))
        #self.s2 = np.zeros(np.int(self.maxtime/self.binsize))
        #self.W = np.zeros(np.int(self.maxtime/self.binsize))
        #self.test= test
        self.reals = reals
        self.longinit = longinit
        self.s1init = s1init
        self.s2init = s2init
        self.Winit = Winit
    
    
    def NormEntropy(self,sigma):
        return 0.5 * np.log(np.linalg.det(2*np.pi*np.exp(1)*sigma))

    
    def datasim(self,freq,a,tau,init,optim,l):
        iterations = [np.int(self.trialsize/self.binsize),np.int(self.longinit/self.binsize)][l==True]
        #print(iterations)
        t = np.zeros(iterations)
        s1,s2,W = np.zeros(iterations),np.zeros(iterations),np.zeros(iterations)
        s1[0] = 1
        if init == True:
            W[0] = self.w0
        else: 
            W[0] = self.W[-1]
        #W[0] = [self.W[-1],self.w0][init == True]
        #if init == True:
        #    W[0] = self.w0
        #else:    
        #    W[0] = self.W[-1]    
        for i in range(1,iterations):
            #print(i)
            lr = learning_rule(s1,s2,a,1.05*a,tau,tau,t,i,self.binsize)
            #print(lr)
            W[i] = W[i-1] + lr + np.random.normal(0,self.genstd) 
            s2[i] = np.random.binomial(1,inverse_logit(W[i]*s1[i-1]+self.b2))
            s1[i] = [np.random.binomial(1,inverse_logit(self.b1)),1][i % int((1/self.binsize)/freq) == 0]
            t[i] = self.binsize*i
        if optim == False:
            if init == True:
                self.s1,self.s2,self.W = s1,s2,W
            else:
                self.s1 = np.hstack((self.s1,s1))
                self.s2 = np.hstack((self.s2,s2))
                self.W = np.hstack((self.W,W))
        else:
            return s1,s2,W
        
    def datasim_const(self,a,tau,init=False,optim = False,l = False):
        iterations = [np.int(self.trialsize/self.binsize),np.int(self.longinit/self.binsize)][l==True]
        s1,s2,W = np.zeros(iterations),np.zeros(iterations),np.zeros(iterations)
        if init == True:
            W[0] = self.w0
        else: 
            W[0] = self.W[-1]
        #W[0] = [self.W[-1],self.w0][init == True]
        t = np.zeros(iterations)
        for i in range(1,iterations):
            lr = learning_rule(s1,s2,a,1.05*a,tau,tau,t,i,self.binsize)
            W[i] = W[i-1] + lr + np.random.normal(0,self.genstd) 
            s2[i] = np.random.binomial(1,inverse_logit(W[i]*s1[i-1]+self.b2))
            s1[i] = np.random.binomial(1,inverse_logit(self.b1))
            t[i] = self.binsize*i
        if optim == False:
            if init == True:
                self.s1,self.s2,self.W = s1,s2,W
            else:
                self.s1 = np.hstack((self.s1,s1))
                self.s2 = np.hstack((self.s2,s2))
                self.W = np.hstack((self.W,W))
        else:
            return s1,s2,W
        
    
    def adjust_proposal(self,means,sample):
        new_shapes = np.array([((means[i]**2) / np.var(sample[300:,i])) for i in range(2)])
        new_rates = np.array([((new_shapes[i]) / means[i]) for i in range(2)])
        return new_shapes,new_rates
    
    def freq_optimiser(self,means,cov,init,optim,l,inference):
        entropies = []
        for j in range(len(self.freqs_init)):
            entropies_temp = []
            for k in range(self.reals):
                s1temp,s2temp,_ = self.datasim(self.freqs_init[j],means[0],means[1],init = init, optim = optim,l = l)
                inference.set_s1(s1temp)
                inference.set_s2(s2temp)
                sample_temp = inference.standardMH_mv(means,cov)
                cov_temp = np.cov(np.transpose(sample_temp[300:,:]))
                entropies_temp.append(self.NormEntropy(cov_temp))
                #print(entropies_temp)
            entropies_temp_clean = [x for x in entropies_temp if math.isnan(x) == False]
            entropies.append(np.mean(entropies_temp_clean))
            #print(entropies)
        return self.freqs_init[np.where(entropies == np.amin(entropies))[0][0]],entropies
    
    def onlineDesign_wh(self,nofreq = False, constant = False, random = False, optimised = True):
        freq_const = self.freqs_init[0]
        optimal_freqs = []
        trials = np.int(self.maxtime / self.trialsize)
        init = False
        self.s1 = self.s1init
        self.s2 = self.s2init
        self.W = self.Winit
        #self.datasim(freq_const,self.Ap,self.tau,init = init, optim = False,l= False)
        inference_whole = ParameterInference(self.s1,self.s2,P = 50, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                        , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=self.trialsize\
                                            ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
        
        sample = inference_whole.standardMH()
        posts = [sample]
        means, cov = [np.mean(sample[300:,0]),np.mean(sample[300:,1])], np.cov(np.transpose(sample[300:,:]))
        ests, entrs = np.array([means]), np.array([self.NormEntropy(cov)])
        new_shapes, new_rates = self.adjust_proposal(means,sample)
        if optimised == True:
            inference_optim = ParameterInference(1,1,P = 50, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                                 , shapes_prior = new_shapes, rates_prior = new_rates,sec=self.trialsize\
                                                     ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
            mutinfs = []
        for i in range(trials):
            if optimised == True:
                inference_optim.set_w0est(self.W[-1])
                opts_temp,mutinfs_temp = self.freq_optimiser(means,cov,init = init, optim = True,l=False,inference = inference_optim)
                optimal_freqs.append(opts_temp)
                mutinfs.append(mutinfs_temp)
                self.datasim(optimal_freqs[-1],self.Ap,self.tau,init=init,optim = False,l=False)
            elif random == True:
                freq_temp = np.random.choice(self.freqs_init)
                self.datasim(freq_temp,self.Ap,self.tau,init=init,optim = False,l=False)
            elif constant == True:
                self.datasim(freq_const,self.Ap,self.tau,init=init,optim = False,l=False)
            elif nofreq == True:
                self.datasim_const(self.Ap,self.tau,init=init,optim = False,l=False)
            inference_whole.set_s1(self.s1)
            inference_whole.set_s2(self.s2)
            inference_whole.set_sec(np.int(len(self.s1)*self.binsize))
            sample = inference_whole.standardMH()
            posts.append(sample)
            means = [np.mean(sample[300:,0]),np.mean(sample[300:,1])]
            cov = np.cov(np.transpose(sample[300:,:]))
            ests = np.vstack((ests, means))
            entrs = np.vstack((entrs,self.NormEntropy(cov)))
            new_shapes, new_rates = self.adjust_proposal(means,sample)
            if optimised == True:
                inference_optim.set_shapes_prior(new_shapes)
                inference_optim.set_rates_prior(new_rates)
        if optimised == True:
            return ests,entrs,optimal_freqs,mutinfs,self.W,posts
        else:
            return ests,entrs,optimal_freqs,self.W,posts
            
        
    def onlineDesign_initdata(self,nofreq = False, constant = False, random = False, optimised = True):
        optimal_freqs = []
        trials = np.int(self.maxtime / self.trialsize)
        init = True
        s1init,s2init,_=self.datasim(self.freqs_init[0],self.Ap,self.tau,init = init, optim = True,l = True)
        inference_long = ParameterInference(s1init,s2init,P = 50, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                        , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=self.longinit\
                                            ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
        sample = inference_long.standardMH()
        means,cov = [np.mean(sample[300:,0]),np.mean(sample[300:,1])],np.cov(np.transpose(sample[300:,:]))
        #means = [self.test.mean(axis = 0)[0],self.test.mean(axis = 0)[1]]
        #cov = np.cov(np.transpose(self.test))
        entrs = np.array([self.NormEntropy(cov)])
        ests = np.array([means])
        new_shapes, new_rates = self.adjust_proposal(means,sample)
        if optimised == True:
            inference_optim = ParameterInference(1,1,P = 50, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                                 , shapes_prior = new_shapes, rates_prior = new_rates,sec=self.trialsize\
                                                     ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
        inference_whole = ParameterInference(1,1,P = 50, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                        , shapes_prior = new_shapes, rates_prior = new_rates,sec=self.trialsize\
                                            ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
        for i in range(trials):
            if optimised == True:
                if init == False:
                    inference_optim.set_w0est(self.W[-1])
                optimal_freqs.append(self.freq_optimiser(means,cov,init = init, optim = True,l=False,inference = inference_optim))
                self.s1,self.s2,self.W=self.datasim(optimal_freqs[-1],self.Ap,self.tau,init=init,optim = True,l=False)
            elif constant == True:
                self.s1,self.s2,self.W=self.datasim(self.freqs_init[0],self.Ap,self.tau,init=init,optim = True,l=False)
            elif random == True:
                freq_temp = np.random.choice(self.freqs_init)
                self.s1,self.s2,self.W=self.datasim(freq_temp,self.Ap,self.tau,init=init,optim = True,l=False)
            elif nofreq == True:
                self.s1,self.s2,self.W=self.datasim_const(self.Ap,self.tau,init=init,optim = True,l=False)
            inference_whole.set_s1(self.s1)
            inference_whole.set_s2(self.s2)
            if init == False:
                inference_whole.set_w0est(self.W[-1])
            sample = inference_whole.standardMH_mv(means,cov)
            if i == 0:
                init = False
            means, cov = [np.mean(sample[300:,0]),np.mean(sample[300:,1])], np.cov(np.transpose(sample[300:,:]))
            ests = np.vstack((ests, means))
            entrs = np.vstack((entrs,self.NormEntropy(cov)))
            new_shapes, new_rates = self.adjust_proposal(means,sample)
            if optimised == True:
                inference_optim.set_shapes_prior(new_shapes)
                inference_optim.set_rates_prior(new_rates)
            inference_whole.set_shapes_prior(new_shapes)
            inference_whole.set_rates_prior(new_rates)
        return ests,entrs,optimal_freqs     
            
            
                        
            
                    
        

if __name__ == "__main__":  
    '''
    data=SimulatedData(Ap=0.005, tau=0.02, std=0.0001,b1=-3.1, b2=-3.1, w0=1.0,sec = 120, binsize = 1/500.0,freq = 50)
    data.create_data()
    s1,s2,_,W=data.get_data()
    data.plot_weight_trajectory()
    data2 = SimulatedData(Ap=0.005, tau=0.02, std=0.0001,b1=-3.1, b2=-3.1, w0=1.0,sec = 120, binsize = 1/500.0,freq = 50)
    data2.create_freq_data()
    s12,s22,_,W2 = data2.get_data()
    
    #data.create_data()

    #np.random.seed(5) 
    '''
    #a = np.ones((2,2))
    #print(np.linalg.norm(a,axis=1))
    #for i in range(20):
    #    design = ExperimentDesign(freqs_init=np.array([20,50,100]),maxtime=60,trialsize=5\
    #                              ,Ap=0.005, tau=0.02, genstd=0.0001,b1=-3.1, b2=-3.1, w0=1.0,binsize = 1/500.0,reals = 1,longinit = 60)
    #    s1,s2,W = design.datasim_const(0.005,0.02,init=True,optim = True,l = False)
    #    np.save('s1init_'+str(i+1),s1)
    #    np.save('s2init_'+str(i+1),s2)
    #    np.save('Winit_'+str(i+1),W)
    #    print(W[-1])
    design = ExperimentDesign(freqs_init=np.array([20,50,100]),maxtime=60,trialsize=5\
                                  ,Ap=0.005, tau=0.02, genstd=0.0001,b1=-3.1, b2=-3.1, w0=1.0,binsize = 1/500.0,reals = 1,longinit = 60)
    s1,s2,W = design.datasim(200,0.005,0.02,True,True,False)
    #ests,entr,opts = design.onlineDesign_wh(nofreq =False,constant = True, random = False, optimised = False)
    '''
    theta_1ms_baseline = [] 
    theta_1ms_freq = []
    theta_5ms_baseline = []
    theta_5ms_freq = []
    #for i in range(120):
    #    b_1ms_A.append(np.mean(theta_1ms_baseline[i][300:][0]))
    for i in range(120):    
        inference1 = ParameterInference(s1[i*500:(i+1)*500],s2[i*500:(i+1)*500],P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                        , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=1\
                                            ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
        theta_1ms_baseline.append(inference1.standardMH(W[i*500],-3.1,-3.1))
        inference2 = ParameterInference(s12[i*500:(i+1)*500],s22[i*500:(i+1)*500],P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                        , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=1\
                                            ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
        theta_1ms_freq.append(inference2.standardMH(W2[i*500],-3.1,-3.1))
    for j in range(24):    
        inference3 = ParameterInference(s1[j*2500:(j+1)*2500],s2[j*2500:(j+1)*2500],P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                        , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=5\
                                            ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
        theta_5ms_baseline.append(inference3.standardMH(W[j*2500],-3.1,-3.1))
        inference4 = ParameterInference(s12[j*2500:(j+1)*2500],s22[j*2500:(j+1)*2500],P = 100, Usim = 100, Ualt = 200,it = 1500, infstd=0.0001, N = 2\
                                        , shapes_prior = np.array([4,5]), rates_prior = np.array([50,100]),sec=5\
                                            ,binsize = 1/500.0,taufix = 0.02,Afix = 0.005)
        theta_5ms_freq.append(inference4.standardMH(W2[j*2500],-3.1,-3.1))
    ''' 
    
