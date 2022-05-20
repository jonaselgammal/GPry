"""
Provides different methods for getting random samples from the allowed sampling
region for either drawing an initial set of samples to start the bayesian
optimization from or get locations from where to start the acquisition
function's optimizer.
"""

from abc import ABCMeta, abstractmethod
from cobaya.cosmo_input.autoselect_covmat import get_best_covmat
from cobaya.tools import resolve_packages_path
from cobaya.model import get_model
from functools import partial
import scipy.stats
import numpy as np
from random import choice
from gpry.mc import generate_sampler_for_gp


class Proposer(metaclass=ABCMeta):

    @abstractmethod
    def get(random_state=None):
        """
        Returns a random sample (given a certain random state) in the parameter space for
        the acquisition function to be optimized from.

        Parameters
        ----------

        random_state : int or numpy.RandomState, optional
            The generator used to initialize the centers. If an integer is
            given, it fixes the seed. Defaults to the global numpy random
            number generator.
        """

    def update(self, gpr):
        """
        Updates the internal GP instance if it has been updated with new data.

        Parameters
        ----------

        gpr : GaussianProcessRegressor
        The gpr instance that has been updated.
        """
        pass


class UniformProposer(Proposer):
    """
    Generates proposals uniformly in a hypercube determined by the bounds

    Parameters
    ----------
    bounds : array-like, shape=(n_dims,2)
        Array of bounds of the prior [lower, upper] along each dimension.

    """

    def __init__(self, bounds):
        n_d = len(bounds)
        proposal_pdf = scipy.stats.uniform(
            loc=bounds[:, 0], scale=bounds[:, 1]-bounds[:, 0])
        self.proposal_function = partial(proposal_pdf.rvs, size=n_d)

    def get(self, random_state=None):
        return self.proposal_function(random_state=random_state)


class MeanCovProposer(Proposer):
    """
    Generates proposals from a multivariate normal distribution given a mean
    vector and covariance matrix.

    Parameters
    ----------
    mean : array-like, shape=(n_dims,)
        Mean-vector of the multivariate normal distribution.

    cov : array-like, shape=(n_dims, n_dims)
        Covariance matrix of the multivariate normal distribution.

        .. note::

            We conduct no explicit checks on whether the covariance matrix you
            provide is singular. Make sure that your matrix is a valid
            covariance matrix!
    """

    def __init__(self, mean, cov):
        self.proposal_function = scipy.stats.multivariate_normal(
            mean=mean, cov=cov, allow_singular=True).rvs

    def get(self, random_state=None):
        return self.proposal_function(random_state=random_state)


class MeanAutoCovProposer(Proposer):
    """
    Does the same as :class:`MeanCovProposer` but tries to get an automatically
    generated covariance matrix from Cobaya's `Cosmo Input Generator`.

    Parameters
    ----------
    mean : array-like, shape=(n_dims,)
        Mean-vector of the multivariate normal distribution.

    model_info : dict
        The info dictionary used to generate the model.
    """

    def __init__(self, mean, model_info):
        cmat_dir = get_best_covmat(
            model_info, packages_path=resolve_packages_path())
        if np.any(d != 0 for d in cmat_dir['covmat'].shape):
            self.proposal_function = scipy.stats.multivariate_normal(
                mean=mean, cov=cmat_dir['covmat'], allow_singular=True).rvs
        else:
            # TODO :: how to gracefully fall back if autocovmat not found
            raise Exception("Autocovmat is not valid")
            # UNDEFINED: model
            self.proposal_function = model.prior.sample

    def get(self, random_state=None):
        return self.proposal_function(random_state=random_state)


class SmallChainProposer(Proposer):
    """
    Uses a short MCMC chain starting from a random training point of the GP to
    generate proposals. Runs a chain of `npoints` length and takes the `nsteps`
    last points as proposals. Once the proposals have been exhausted or the GP
    has been fit with new data a new chain is run from a different starting
    location.

    Parameters
    ----------
    bounds : array-like, shape=(n_dims,2)
        Array of bounds of the prior [lower, upper] along each dimension.

    n_points : int, optional (default=100)
        The max number of samples that is generated from the chain

    nsteps : int, optional (default=10)
        The number of MC steps that is taken from the end of the chain.
    """

    def __init__(self, bounds, npoints=100, nsteps=10):
        self.samples = []
        self.bounds = bounds
        self.npoints = npoints
        self.nsteps = nsteps

    def get(self, random_state=None):
        if len(self.samples) > 0:
            return self.samples.pop()
        else:
            self.resample().pop()

    def resample(self):
        this_i = choice(range(len(self.gpr.X_train)))
        this_X = np.copy(self.gpr.X_train[this_i])
        logpost = self.gpr.y_train[this_i]
        self.sampler.current_point.add(this_X, logpost)
        # reset the number of samples and run
        self.sampler.collection.reset()
        self.sampler.run()
        points = self.sampler.products()["sample"][self.parnames].values[::-self.n_steps]
        self.samples = points
        return self.samples

    def update(self, gpr):
        self.samples = []
        surr_info, sampler = generate_sampler_for_gp(
            gpr, self.bounds, sampler="mcmc", add_options={'max_tries': self.npoints})
        self.sampler = sampler
        self.parnames = list(surr_info['params'])
        self.gpr = gpr


class PartialProposer(Proposer):
    """
    Combines any of the other proposers with a :class:`UniformProposer` with
    a fraction drawn from the uniform proposer to encourage exploration.

    Parameters
    ----------
    bounds : array-like, shape=(n_dims,2)
        Array of bounds of the prior [lower, upper] along each dimension.

    true_proposer: Proposer
        The initialized Proposer instance to use instead of uniform for a
        fraction of samples.

    random_proposal_fraction : float, between 0 and 1, optional (default=0.25)
        The fraction of proposals that is drawn from the UniformProposer.
    """

    # Either sample from true_proposer, or give a random prior sample
    def __init__(self, bounds, true_proposer, random_proposal_fraction=0.25):

        n_d = len(bounds)
        if random_proposal_fraction > 1. or random_proposal_fraction < 0.:
            raise ValueError(
                "Cannot pass a fraction outside of [0,1]. "
                f"You passed 'random_proposal_fraction={random_proposal_fraction}'")
        if not isinstance(true_proposer, Proposer):
            raise ValueError("The true proposer needs to be a valid proposer.")

        self.rpf = random_proposal_fraction
        # ToDo: Make this a sample of the prior instead of uniform hypercube.
        self.random_proposer = UniformProposer(bounds, n_d)
        self.true_proposer = true_proposer

    def get(self, random_state=None):
        if np.random.random() > self.rpf:
            return self.true_proposer.get(random_state=random_state)
        else:
            return self.random_proposer.get(random_state=random_state)

    def update(self, gpr):
        self.true_proposer.update(gpr)



class Centroids(Proposer):
    """
    Proposes points at the centroids of subsets of dim-1 training points. It
    perturbs some of the proposals away from the centroids to encourage
    exploration.

    bounds : array-like, shape=(n_dims,2)
        Array of bounds of the prior [lower, upper] along each dimension.

    lambda: float, optional (default=1)
        Controls the scale of the perturbation of samples. Lower values
        correspond to more exploration.
    """

    def __init__(self, bounds, lambd=1.):
        self.bounds = bounds
        # Set bounds to None as we want to be able to initialize the proposer
        # before we have trained our model and it's updated in the propose
        # method.
        self.training = None
        # TODO: adapt lambd to dimensionality!!!
        # e.g. 1 seems to work well for d=2, and ~0.5 for d=30
        self.kicking_pdf = scipy.stats.expon(scale=1 / lambd)

    @property
    def d(self):
        return len(self.bounds)

    def get(self, random_state=None):
        # TODO: actually use the random_state!
        m = self.d + 1
        subset = self.training[
            np.random.choice(len(self.training), size=m, replace=False)]
        centroid = np.average(subset, axis=0)
        # perturb the point: per dimension, add a random multiple of the difference
        # between the centroid and one of the points.
        kick = -centroid + np.array(
            [subset[j][i] for i, j in enumerate(
                np.random.choice(m, size=self.d, replace=False))])
        kick *= self.kicking_pdf.rvs(self.d)
        # This might have to be modified if the optimizer can't deal with
        # points which are exactly on the edges.
        return np.clip(centroid + kick, self.bounds[:, 0], self.bounds[:, 1])

    def update(self, gpr):
        # Get training locations from gpr and save them
        self.training = gpr.X_train
