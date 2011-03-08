import pymc as pm
import pymc.gp as gp
from pymc.gp.cov_funs import matern
import numpy as np
import matplotlib.pyplot as pl
import gradient_samplers
from numpy.random import normal

x = np.arange(-1.,1.,.1)

def dinvlogit(ltheta):
    return np.exp(2.*ltheta)/(1.+np.exp(ltheta))**2

class InvLogit(pm.Deterministic):
    """
    P = InvLogit(name, ltheta[, doc, dtype=None, trace=True,
        cache_depth=2, plot=None])

    A Deterministic whose value is the inverse logit of parent ltheta.

    :Parameters:
      name : string
        The name of the variable.
      ltheta : number, array or variable
        The parent to which the inverse logit function should be
        applied.
      other parameters :
        See docstring of Deterministic.

    :SeeAlso:
      Deterministic, Lambda, Logit, StukelLogit, StukelInvLogit
    """
    def __init__(self, name, ltheta, doc='An inverse logit transformation', *args, **kwds):
        pm.Deterministic.__init__(self, eval=pm.utils.invlogit, name=name, parents={'ltheta': ltheta},
                                jacobians={'ltheta':dinvlogit}, jacobian_formats={'ltheta':'broadcast_operation'}, 
                                doc=doc, *args, **kwds)

def make_model():
    # Prior parameters of C
    # diff_degree = pm.Uniform('diff_degree', 1., 3)
    
    scalar_obs = True
    
    diff_degree = pm.Exponential('diff_degree',.1,value=.5, observed=scalar_obs)
    amp = pm.Lognormal('amp', mu=.4, tau=1., value=1, observed=scalar_obs)
    scale = pm.Lognormal('scale', mu=.5, tau=1., value=1, observed=scalar_obs)

    # The covariance dtrm C is valued as a Covariance object.
    @pm.deterministic
    def C(eval_fun = gp.matern.euclidean, diff_degree=diff_degree, amp=amp, scale=scale):
        return gp.NearlyFullRankCovariance(eval_fun, diff_degree=diff_degree, amp=amp, scale=scale)
        
    # Prior parameters of M
    a = pm.Normal('a', mu=1., tau=1., value=1, observed=scalar_obs)
    b = pm.Normal('b', mu=.5, tau=1., value=1, observed=scalar_obs)
    c = pm.Normal('c', mu=2., tau=1., value=1, observed=scalar_obs)

    # The mean M is valued as a Mean object.
    def linfun(x, a, b, c):
        # return a * x ** 2 + b * x + c
        return 0.*x + c
    @pm.deterministic
    def M(eval_fun = linfun, a=a, b=b, c=c):
        return gp.Mean(eval_fun, a=a, b=b, c=c)

    # The GP submodel
    fmesh = np.linspace(-np.pi/3.3,np.pi/3.3,4)
    sm = gp.GPSubmodel('sm',M,C,fmesh)
    sm.f_eval.value = 0*sm.f_eval.value
    
    @pm.potential(logp_partial_gradients = {'f': lambda f: 0*f})
    def fake(f=sm.f_eval):
        "Prevents DrawFromPrior being used for sm.f_eval"
        return 0
    
    # The data d is just array-valued. It's normally distributed about GP.f(obs_x).
    n_obs = [28,2,28,2]
    
    p = InvLogit('p',sm.f_eval)
    n = pm.Binomial('n',p=p,n=30,value=n_obs,observed=True)
    return locals()
    
M = pm.MCMC(make_model())

# print pm.utils.logp_gradient_of_set(set([M.sm.f_eval]), set([M.n]))

# # M.use_step_method(gradient_samplers.GPParentHMCStep, list(M.stochastics-set([M.sm.f])))
M.use_step_method(gradient_samplers.GPParentHMCStep, [M.sm.f_eval], step_size_scaling = .00001, trajectory_length=.1, keep_history=False)

sm = M.step_method_dict[M.sm.f_eval][0]
# M.isample(100)
# pl.clf()
# trace=M.trace('sm_f_eval')[:]
# pl.plot(trace[:,0], trace[:,1], 'k.')
# print sm.gradients_vector
# sm.step()
# history = np.array(sm.history)
# true_cov = list(sm.stochastics)[0].parents['sig'].parents['cb'].value[1]
# pl.clf()
# pl.plot(history[:,0], history[:,1])
# e,v = np.linalg.eigh(true_cov[:2,:2])