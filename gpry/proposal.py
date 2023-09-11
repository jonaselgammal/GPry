# Random proposals from which to optimize the acquisition function

from abc import ABCMeta, abstractmethod
from cobaya.cosmo_input.autoselect_covmat import get_best_covmat
from cobaya.tools import resolve_packages_path
from functools import partial
import scipy.stats
import numpy as np
from gpry.mc import mc_sample_from_gp
from cobaya.model import LogPosterior
from gpry.tools import check_random_state
from math import inf


class Proposer(metaclass=ABCMeta):
    """
    Base proposer class. All other proposers inherit from it. If you want to define your
    own custom proposer it should also inherit from it.
    """

    @abstractmethod
    def get(random_state=None):
        """
        Returns a random sample (given a certain random state) in the parameter space for
        getting initial training samples or the acquisition function to be optimized from.

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

class InitialPointProposer(metaclass=ABCMeta):
    """
    Base proposer class for all proposers which work for initial point generation.
    """

class ReferenceProposer(Proposer, InitialPointProposer):
    """
    Generates proposals from the "reference" distribution defined in the model. If no
    reference distribution is defined it defaults to the prior.

    Parameters:
    -----------
    model : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_
        The model from which to draw the samples.
    """

    def __init__(self, model, max_tries=inf, warn_if_tries='10d', ignore_fixed=True):
        self.prior = model.prior
        self.warn=True
        self.max_tries = max_tries
        self.warn_if_tries = warn_if_tries
        self.ignore_fixed = ignore_fixed

    def get(self, random_state=None):
        ref = self.prior.reference(
            max_tries=self.max_tries, warn_if_tries=self.warn_if_tries,
            ignore_fixed=self.ignore_fixed, warn_if_no_ref=self.warn,
            random_state=random_state)
        self.warn = False
        return ref

class PriorProposer(Proposer, InitialPointProposer):
    """
    Generates proposals from the prior of the model.

    Parameters:
    -----------
    model : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_
        The model from which to draw the samples.
        max_tries=inf, warn_if_tries='10d', ignore_fixed=False
    """

    def __init__(self, model):
        self.model = model

    def get(self, random_state=None):
        return self.model.prior.sample(random_state=random_state)[0]

class UniformProposer(Proposer, InitialPointProposer):
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
            loc=bounds[:, 0], scale=bounds[:, 1] - bounds[:, 0])
        self.proposal_function = partial(proposal_pdf.rvs, size=n_d)

    def get(self, random_state=None):
        return self.proposal_function(random_state=random_state)


class PartialProposer(Proposer, InitialPointProposer):
    """
    Combines any of the other proposers with a :class:`UniformProposer` with
    a fraction drawn from the uniform proposer to encourage exploration.

    .. warning::

        If you want to use this proposer for initial point generation make sure that your
        true_proposer is compatible.

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

        if random_proposal_fraction > 1. or random_proposal_fraction < 0.:
            raise ValueError(
                "Cannot pass a fraction outside of [0,1]. "
                f"You passed 'random_proposal_fraction={random_proposal_fraction}'")
        if not isinstance(true_proposer, Proposer):
            raise ValueError("The true proposer needs to be a valid proposer.")

        self.rpf = random_proposal_fraction
        # ToDo: Make this a sample of the prior instead of uniform hypercube.
        self.random_proposer = UniformProposer(bounds)
        self.true_proposer = true_proposer

    def get(self, random_state=None):
        rng = check_random_state(random_state)
        if rng.random() > self.rpf:
            return self.true_proposer.get(random_state=rng)
        else:
            return self.random_proposer.get(random_state=rng)

    def update(self, gpr):
        self.true_proposer.update(gpr)


class MeanCovProposer(Proposer, InitialPointProposer):
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


# UNUSED (not really, but should not be here, being specific to one problem)
class MeanAutoCovProposer(Proposer, InitialPointProposer):
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


# UNUSED
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

    nretries : int, optional (default=3)
        The number of times that the MC chain is restarted if it fails.
        If the chain fails `nretries` times a warning is printed and samples
        from a uniform distribution are returned.
    """

    def __init__(self, bounds, npoints=100, nsteps=20, nretries=3):
        self.samples = []
        self.bounds = bounds
        self.nretries = nretries
        self.npoints = npoints
        self.nsteps = nsteps
        self.random_proposer = UniformProposer(bounds)

    def get(self, random_state=None):
        if len(self.samples) > 0:
            last, self.samples = self.samples[-1], self.samples[:-1]
            return last
        else:
            self.resample(random_state)
            last, self.samples = self.samples[-1], self.samples[:-1]
            return last

    def resample(self, random_state=None):
        rng = check_random_state(random_state)
        for i in range(self.nretries):
            this_i = rng.choice(range(len(self.gpr.X_train)))
            this_X = np.copy(self.gpr.X_train[this_i])
            logpost = self.gpr.y_train[this_i]
            self.sampler.current_point.add(this_X, LogPosterior(logpost=logpost))
            # reset random state and number of samples
            self.sampler._rng = rng
            self.sampler.collection.reset()
            try:
                self.sampler.run()
                points = self.sampler.products()["sample"][self.parnames].values
                self.samples = points[::-self.nsteps]
                return
            except Exception:
                pass
        # if resample_tries failed raise Warning and pass uniform points
        print("[proposer] WARNING: MC chain got stuck. Taking random uniform points")
        self.samples = np.empty((self.nsteps, len(self.bounds)))
        for i in range(self.nsteps):
            self.samples[i] = self.random_proposer.get()

    def update(self, gpr):
        self.samples = []
        surr_info, sampler = mc_sample_from_gp(
            gpr, self.bounds, sampler="mcmc", run=False, add_options={
                'max_samples': self.npoints, 'max_tries': 10 * self.npoints})
        self.sampler = sampler
        self.parnames = list(surr_info['params'])
        self.gpr = gpr


class CentroidsProposer(Proposer):
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
        # TODO: adapt lambda to dimensionality!
        # e.g. 1 seems to work well for d=2, and ~0.5 for d=30
        self.kicking_pdf = scipy.stats.expon(scale=1 / lambd)

    @property
    def d(self):
        """Dimensionality of the prior."""
        return len(self.bounds)

    def get(self, random_state=None):
        rng = check_random_state(random_state)
        m = self.d + 1
        subset = self.training[rng.choice(len(self.training), size=m, replace=False)]
        centroid = np.average(subset, axis=0)
        # perturb the point: per dimension, add a random multiple of the difference
        # between the centroid and one of the points.
        kick = -centroid + np.array(
            [subset[j][i] for i, j in enumerate(
                rng.choice(m, size=self.d, replace=False))])
        kick *= self.kicking_pdf.rvs(self.d, random_state=rng)
        # This might have to be modified if the optimizer can't deal with
        # points which are exactly on the edges.
        return np.clip(centroid + kick, self.bounds[:, 0], self.bounds[:, 1])

    def update(self, gpr):
        # Get training locations from gpr and save them
        self.training = np.copy(gpr.X_train)
