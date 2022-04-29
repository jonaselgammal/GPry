from abc import ABCMeta, abstractmethod
from cobaya.cosmo_input.autoselect_covmat import get_best_covmat
from cobaya.tools import resolve_packages_path
from functools import partial
import scipy.stats

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

