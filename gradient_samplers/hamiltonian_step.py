'''
Created on Feb 13, 2011

@author: johnsalvatier
'''
import numpy as np
import pymc as pm
from multistep import MultiStep

class HMCStep(MultiStep):
    
    def __init__(self, stochastics,covariance, leapfrog_size = .3, leapfrog_n = 7, verbose = 0, tally = True  ):
        MultiStep.__init__(self, stochastics, verbose, tally)
        
        self.covariance = covariance
        self.inv_covariance = np.linalg.inv(covariance)
        self.leapfrog_size = leapfrog_size
        self.leapfrog_n = leapfrog_n 
        self.zero = np.zeros(self.dimensions)
        
    
    def step(self):
        startp = self.logps
        
        p = np.random.multivariate_normal(mean = self.zero ,cov = self.inv_covariance)
        start_p = p
        
        p = p - (self.leapfrog_size/2) * (-self.logp_grads)
        
        for i in range(self.leapfrog_n): 
            
            self.consider(self.vectors + self.leapfrog_size * np.dot(self.covariance, p))
            
            if i != self.leapfrog_n - 1:
                p = p - self.leapfrog_size * (-self.logp_grads)
             
        p = p - (self.leapfrog_size/2) * (-self.logp_grads)   
        
        p = -p 
            
        log_metrop_ratio = (-startp) - (-self.logps) + self.kenergy(start_p) - self.kenergy(p)
        
        if (np.isfinite(log_metrop_ratio) and 
            np.log(np.random.uniform()) < log_metrop_ratio):
            
            self.accept()
        else: 
            self.reject() 
            
    
    def kenergy (self, x):
        return .5 * np.dot(x,np.dot(self.covariance, x))
        