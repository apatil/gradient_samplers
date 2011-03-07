'''
Created on Feb 13, 2011

@author: johnsalvatier
'''
import numpy as np
import pymc as pm
import multistep
import find_mode as fm
import approx_hess as ah

__all__ = ['HMCStep','GPParentHMCStep']

class HMCStep(multistep.MultiStep, pm.Metropolis):
    """
    Hamiltonian/Hybrid Monte-Carlo (HMC) step method. Works well on continuous variables for which
    the gradient of the log posterior can be calculated.
    
    Based off Radford's review paper of the subject.
    
    Parameters
    ----------
    stochastics : single or iterable of stochastics
        the stochastics that should use this HMCStep
    step_size_scaling : float
        a scaling factor for the step sizes. If more than 1 value then treated as an interval to
        randomize between.
    trajectory_length : float
        (roughly) how far each HMC step should travel (think in terms of standard deviations)
    covariance : (ndim , ndim) ndarray (where ndim is the total number of variables)
        covariance matrix for the HMC sampler to use.
        If None then will be estimated using the inverse hessian at the mode
    find_mode : bool
        whether to start the chain at the local minima of the distribution.
        If false, will start the simulation from the initial values of the stochastics
    
    
    Tuning advice:
     * General problems: try passing a better covariance matrix. For example,
        try doing a trial run and and then passing the empirical covariance matrix from
        that. 
     * optimal acceptance approaches .651 from above for high dimensional posteriors 
         (see Beskos 2010 esp. page 13). Target somewhat higher acceptance in pratice.
     * Low acceptance: try a lower step_size_scaling. 
     * Slow mixing: try significantly longer or shorter trajectory length (trajectories
        can double back).
     * Seems to sometimes get stuck in places for long periods: This is due to trajectory
         instability, try a smaller step size. Think of this as low acceptance in certain 
         areas. This is a sign that the sampler may give misleading results for small sample
         numbers in the areas with different stability limits (often the tails), so don't 
         ignore this if you care about those areas. Randomizing the step size may help.
     * See section 4.2 of Radford's paper for more advice.
     
    Relevant Literature: 
    
    A. Beskos, N. Pillai, G. Roberts, J. Sanz-Serna, A. Stuart. "Optimal tuning of the Hybrid Monte-Carlo Algorithm" 2010. http://arxiv.org/abs/1001.4460
    G. Roberts. "MCMC using Hamiltonian dynamics" out of "Handbook of Markov Chain Monte Carlo" 2010. http://www.cs.utoronto.ca/~radford/ham-mcmc.abstract.html    
    """
    
    optimal_acceptance = .651 #Beskos 2010
    _tuning_info = ['acceptr']
    
    def __init__(self, stochastics, step_size_scaling = .25, trajectory_length = 2., verbose = 0, tally = True, masses = None):
        multistep.MultiStep.__init__(self, stochastics, verbose, tally)
        
        
        self._id = 'HMC'
        self.proposal_distribution='None'
                    
        self.gaussian_stochastics = filter(lambda x: isinstance(x,pm.MvNormalChol), self.stochastics)
        _, self.gaussian_dimensions = multistep.vectorize_stochastics(self.gaussian_stochastics)
        self.nongaussian_stochastics = list(set(self.stochastics) - set(self.gaussian_stochastics))
        _, self.nongaussian_dimensions = multistep.vectorize_stochastics(self.nongaussian_stochastics)
        self.slices, self.dimensions = multistep.vectorize_stochastics(self.stochastics)
        
        wg = []
        wng = []
        for s in self.stochastics:
            slice = self.slices[s.__name__]
            index_range = range(slice.start, slice.stop)
            if s in self.gaussian_stochastics:
                wg += index_range
            else:
                wng += index_range
        self.where_gaussian = (np.array(wg,dtype='int'),)
        self.where_nongaussian = (np.array(wng,dtype='int'),)
        
        # These can be updated by adaptation if desired.
        if self.nongaussian_dimensions>0:
            self.nongaussian_covariance = np.diag(1./masses) if masses is not None else np.eye(self.nongaussian_dimensions)
            self.nongaussian_cholesky = np.linalg.cholesky(self.nongaussian_covariance).copy('F')

        step_size = step_size_scaling * self.dimensions**(1/4.)
        
        if np.size(step_size) > 1:
            self.step_size_max, self.step_size_min = step_size
        else :
            self.step_size_max = self.step_size_min = step_size 
        
        self.trajectory_length = trajectory_length   
        self.zero = np.zeros(self.dimensions)
        
        self.acceptr = 0.
    
    reject = multistep.MultiStep.revert

    @property
    def nongaussian_logp_gradient(self):
        # The gradient of the logp, not including the Gaussian stochastics handled by self.
        # Reason: Those stochastics are in their own component of the Hamiltonian.
        # FIXME: Don't do all these set calls here.
        # FIXME: Use markov_blanket somehow, not markov_blanket_no_f. Maybe hand-wrap.
        return pm.utils.logp_gradient_of_set(set(self.stochastics), set(self.markov_blanket_no_f)-set(self.gaussian_stochastics))

    @property
    def gradients_vector(self):
        "HMCStep.gradients_vector does not take account of the prior logp's of the MvNormalChol's in the model."
        grad_logp = np.empty(self.dimensions)
        for stochastic, logp_gradient in self.nongaussian_logp_gradient.iteritems():
                
            grad_logp[self.slices[str(stochastic)]] = np.ravel(logp_gradient)  

        return grad_logp
    
    def scale(self,v,A,inv,transa):
        if self.gaussian_dimensions==0:
            return
        if inv:
            op=pm.flib.dtrsm_wrap
        else:
            op=pm.flib.dtrmm_wrap
        for s in self.gaussian_stochastics:
            op(A[s], v[self.slices[s.__name__]], side='L', transa=transa, uplo='L', alpha=1.)
        
    def init_momentum(self):
        # momentum scale proportional to inverse of parameter scale (basically sqrt(covariance))
        # Momentums corresponding to Gaussian parameters are understood to be scaled.
        L = self.get_L()
        p = np.empty(self.dimensions)
        p_nongaussian = np.random.normal(size=self.nongaussian_dimensions)
        p_gaussian = np.random.normal(size=self.gaussian_dimensions)
        # Compute kinetic energy before scaling momentum.
        ke = self.kenergy(p_nongaussian, p_gaussian)
        if self.nongaussian_dimensions>0:
            pm.flib.dtrsm_wrap(self.nongaussian_cholesky, p_nongaussian, transa='T', side='L', uplo='L', alpha=1.)
        p[self.where_nongaussian] = p_nongaussian
        p[self.where_gaussian] = p_gaussian
        self.scale(p,L,True,'T')
        return p, ke
    
    def get_L(self):
        return dict([(s, pm.utils.value(s.parents['sig'])) for s in self.gaussian_stochastics])
    
    def gaussian_step(self, p, q, step_size):
        L = self.get_L()
        self.scale(p,L,False,'T')
        self.scale(q,L,True,'N')
        p_gaussian = p[self.where_gaussian]
        q_gaussian = q[self.where_gaussian]
        r = np.sqrt(p_gaussian**2+q_gaussian**2)
        theta = np.arctan2(p_gaussian, q_gaussian)
        theta = theta + step_size
        p[self.where_gaussian] = r*np.cos(theta)
        q[self.where_gaussian] = r*np.sin(theta)
        self.scale(p,L,True,'T')
        self.scale(q,L,False,'N')
        return p,q
    
    def p_step(self, p, step_size):
        if self.nongaussian_dimensions > 0:
            return p - (step_size) * (-self.gradients_vector)
        else:
            return p
    def propose(self):
        self.record_starting_value()

        #randomize step size
        step_size = np.random.uniform(self.step_size_min, self.step_size_max)
        step_count = int(np.floor(self.trajectory_length / step_size))

        p, start_ke = self.init_momentum()

        #use the leapfrog method
        p = self.p_step(p,step_size*.5) # half momentum update

        for i in range(step_count): 
            
            #alternate full variable and momentum updates
            new_vector = self.vector
            if self.nongaussian_dimensions>0:
                nongaussian_delta = step_size * np.dot(self.nongaussian_covariance, p[self.where_nongaussian])
                new_vector[self.where_nongaussian] += nongaussian_delta
            
                # Need to inter here in case the Gaussians' distributions change.
                self.consider(new_vector)
            
            # One full update for the scaled Gaussian positions and momentums.
            # Requires four sets of triangular matrix multiplications
            if self.gaussian_dimensions>0:
                p,newer_vector = self.gaussian_step(p,new_vector,step_size)
                self.consider(newer_vector)

            # Make sure you haven't wandered into an illegal state
            try:
                self.logp_plus_loglike
            except pm.ZeroProbability:
                print 'Encountered illegal state while simulating, returning early.'
                return
            
            if i != step_count - 1:
                p = p - step_size * (-self.gradients_vector)
             
        p = self.p_step(p,step_size*.5)   # do a half step momentum update to finish off
        
        p = -p 
            
        # Rescale momentum and compute kinetic energy.
        L=self.get_L()
        self.scale(p,L,False,'T')    
        p_gaussian = p[self.where_gaussian]
        p_nongaussian = p[self.where_nongaussian]
        if self.nongaussian_dimensions>0:
            pm.flib.dtrmm_wrap(self.nongaussian_cholesky, p_nongaussian, side='L', transa='T', uplo='L', alpha=1.)
        ke = self.kenergy(p_nongaussian, p_gaussian)
        
        # This 'Hastings factor' makes the standard formula for Metropolis-Hastings acceptance work.    
        self._hastings_factor = start_ke - ke
        
    def hastings_factor(self):
        return self._hastings_factor
    
    def kenergy(self, p_nongaussian_s, p_gaussian_s):
        return .5*(np.dot(p_nongaussian_s,p_nongaussian_s)+np.dot(p_gaussian_s,p_gaussian_s))
        
    @staticmethod
    def competence(s):
        if pm.datatypes.is_continuous(s): 
            if False: # test ability to find gradient 
                return 2.5
        return 0

GPParentHMCStep = pm.gp.wrap_metropolis_for_gp_parents(HMCStep)