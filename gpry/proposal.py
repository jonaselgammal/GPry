from abc import ABCMeta, abstractmethod
from cobaya.cosmo_input.autoselect_covmat import get_best_covmat
from cobaya.tools import resolve_packages_path
from functools import partial
import scipy.stats
import numpy as np
from gpry.tools import generate_sampler_for_gp

class Proposer(metaclass=ABCMeta):

    @abstractmethod
    def get(random_state=None):
        """ Returns a random sample (given a certain random state) in the parameter space for the acquisition function to be optimized from """

class UniformProposer(Proposer):
    def __init__(self,bounds,n_d):
        proposal_pdf = scipy.stats.uniform(loc=bounds[:,0],scale=bounds[:,1])
        self.proposal_function = partial(proposal_pdf.rvs,size=n_d)
    def get(self,random_state=None):
        return self.proposal_function(random_state=random_state)

class MeanCovProposer(Proposer):
    def __init__(self,mean, cov):
        self.proposal_function = scipy.stats.multivariate_normal(mean=mean,cov=cov).rvs
    def get(self,random_state=None):
        return self.proposal_function(random_state=random_state)

class MeanAutoCovProposer(Proposer):
    def __init__(self,mean,model_info):
        cmat_dir = get_best_covmat(model_info, packages_path=resolve_packages_path())
        if np.any(d!=0 for d in cmat_dir['covmat'].shape):
          self.proposal_function = scipy.stats.multivariate_normal(mean=mean,cov=cmat_dir['covmat']).rvs
        else:
          self.proposal_function = model.prior.sample
    def get(self,random_state=None):
        return self.proposal_function(random_state=random_state)  

def SmallChainProposer(Proposer):
    def __init__(self, gpr, bounds, npoints=100, nsteps=10):
        self.samples = []
        surr_info, sampler = generate_sampler_for_gp(gpr, bounds, sampler="mcmc", add_options={'max_samples':100})
        self.sampler = sampler
        self.surr_model = get_model(surr_info)
        self.parnames = list(surr_info['params'])
        self.nsteps = 10
        self.gpr = gpr
    def get(self, random_state=None):
        if len(self.samples)>0:
            return self.samples.pop()
        else:
            self.resample().pop()
    def resample(self):
        this_i = choice(range(len(self.gpr.X_train)))
        this_X = np.copy(self.gpr.X_train[this_i])
        logpost = model.logposterior(this_X, temperature=self.temperature)
        self.sampler.current_point.add(this_X, logpost)
        # reset the number of samples and run
        self.sampler.collection.reset()
        self.sampler.run()
        points = self.sampler.products()["sample"][parnames].values[::-self.n_steps]
        self.samples = points
        return self.samples

class PartialProposer(Proposer):

    # Either sample from true_proposer, or give a random prior sample
    def __init__(self, bounds, n_d,true_proposer, random_proposal_fraction = 0.):

        if random_proposal_fraction > 1. or random_proposal_fraction < 0.:
          raise ValueError("Cannot pass a fraction outside of [0,1]. You passed {} for 'random_proposal_fraction'.".format(random_proposal_fraction))
        if not isinstance(true_proposer,Proposer):
          raise ValueError("The true proposer needs to be a valid proposer.")

        self.rpf = random_proposal_fraction
        self.random_proposer = UniformProposer(bounds, n_d)
        self.true_proposer = true_proposer

    def get(self,random_state=None):
        if np.random.random() > self.rpf:
            return self.true_proposer.get(random_state=random_state)
        else:
            return self.random_proposer.get(random_state=random_state)
              
