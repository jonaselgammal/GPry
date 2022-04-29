from abc import ABCMeta, abstractmethod
from cobaya.cosmo_input.autoselect_covmat import get_best_covmat
from cobaya.tools import resolve_packages_path
from functools import partial
import scipy.stats
import numpy.random.random as random_draw

class Proposer(metaclass=ABCMeta):

    @abstractmethod
    def get(random_state=None):
        """ Returns a random sample (given a certain random state) in the parameter space for the acquisition function to be optimized from """

class UniformProposer(Proposer):
    def __init__(self,bounds,n_d):
        self.proposal_function = partial(scipy.stats.uniform.rvs,loc=bounds[:,0],scale=bounds[:,1],size=n_d)
    def get(self,random_state=None):
        return self.proposal_function(random_state=random_state)
      
class MeanCovProposer(Proposer):
    def __init__(self,mean, cov):
        self.proposal_function = partial(scipy.stats.multivariate_normal.rvs,mean=mean,cov=cov)
    def get(self,random_state=None):
        return self.proposal_function(random_state=random_state)

class MeanAutoCovProposer(Proposer):
    def __init__(self,mean,model_info):
        cmat_dir = get_best_covmat(model_info, packages_path=resolve_packages_path())
        if np.any(d!=0 for d in cmat_dir['covmat'].shape):
          self.proposal_function = partial(scipy.stats.multivariate_normal.rvs,mean=mean,cov=cmat_dir['covmat'])
        else:
          self.proposal_function = model.prior.sample
    def get(self,random_state=None):
        return self.proposal_function(random_state=random_state)  

class PartialProposer(Proposer):

    # Either sample from true_proposer, or give a random prior sample
    def __init__(self,random_proposal_fraction = 0., bounds, n_d,true_proposer):

        if random_proposal_fraction > 1. or random_proposal_fraction < 0.:
          raise ValueError("Cannot pass a fraction outside of [0,1]. You passed {} for 'random_proposal_fraction'.".format(random_proposal_fraction))
        if not isinstance(true_proposer,Proposer):
          raise ValueError("The true proposer needs to be a valid proposer.")

        self.rpf = random_proposal_fraction
        self.random_proposer = UniformProposer(bounds, n_d)
        self.true_proposer = true_proposer

    def get(self,random_state=None):
        if random_draw() > self.rpf:
            return self.true_proposer.get(random_state=random_state)
        else:
            return self.random_proposer.get(random_state=random_state)
              
