"""
This module provides different classes to generate random proposals within the prior or
trust bounds, to be used as initial samples for the active learning cycle, or as starting
points from which to optimize the acquisition function.

The way these points are proposed becomes increasingly important with higher dimensions as
the volume of the parameter space grows exponentially.

Every proposer has a ``get`` method which returns a random sample from the proposer.
"""

from abc import ABCMeta, abstractmethod
from functools import partial
from math import inf
from warnings import warn

import scipy.stats
import numpy as np

from gpry.tools import check_random_state, is_in_bounds
from gpry import mpi


def check_in_bounds(get_method):
    """
    Decorator for ``get`` methods of ``Proposer`` sub-classes, that call the method until
    the returned proposal falls within the ``bounds`` defined as an attribute.

    Print a warning every 1000 failed attempts.

    It does not need to be used if the returned proposals are guaranteed to fulfil it.
    """

    def wrapper(self, *args, **kwargs):
        i = 0
        x = [np.nan]
        while not is_in_bounds([x], self.bounds, validate=False)[0]:
            i += 1
            if not i % 1000:
                warn(
                    f"[{self.__class__.__name__}] Could produce a proposal"
                    f" within the given bounds after {i} tries."
                )
            x = get_method(self, *args, **kwargs)
        return x

    return wrapper


class Proposer(metaclass=ABCMeta):
    """
    Base proposer class. All other proposers inherit from it. If you want to define your
    own custom proposer it should also inherit from it.
    """

    @abstractmethod
    def get(self, rng=None):
        """
        Returns a random sample (given a certain random state) in the parameter space for
        getting initial training samples or the acquisition function to be optimized from.

        If the output is not guaranteed by construction to be within the bounds defined in
        the ``bounds`` attribute, decorated this method with ``check_in_bounds``.

        Parameters
        ----------
        rng : int or numpy.random.Generator, optional
            The generator used to propose points. If an integer is given, it is used as a
            seed for the default global numpy random number generator.
        """

    def update_bounds(self, bounds):
        """
        Updates the bounds for the proposal.

        Parameters
        ----------
        bounds : array
            Bounds in which to optimize the acquisition function, assumed to be of shape
            (d,2) for d dimensional prior
        """
        self.bounds = np.atleast_2d(bounds)

    def update(self, surrogate):
        """
        Updates the internal GP instance if it has been updated with new data.

        Parameters
        ----------
        surrogate : SurrogateModel
            The surrogate instance that has been updated.
        """


class InitialPointProposer(metaclass=ABCMeta):
    """
    Base proposer class for all proposers which work for initial point generation.
    """


class ReferenceProposer(Proposer, InitialPointProposer):
    """
    Generates proposals from the "reference" distribution defined in the Truth. If no
    reference distribution is defined it defaults to the prior.

    Parameters
    ----------
    truth : Truth
        The true model from which to draw the samples.
    """

    def __init__(self, truth, bounds=None):
        self.truth = truth
        self.update_bounds(bounds if bounds is not None else truth.prior_bounds)

    @check_in_bounds
    def get(self, rng=None):
        return self.truth.ref_sample(rng)


class PriorProposer(Proposer, InitialPointProposer):
    """
    Generates proposals from the prior of the problem.

    Parameters
    ----------
    truth : Truth
        The true model from which to draw the samples.
    """

    def __init__(self, truth, bounds=None):
        self.truth = truth
        self.update_bounds(bounds if bounds is not None else truth.prior_bounds)

    @check_in_bounds
    def get(self, rng=None):
        return self.truth.prior_sample(rng)


class UniformProposer(Proposer, InitialPointProposer):
    """
    Generates proposals uniformly in a hypercube determined by the bounds

    Parameters
    ----------
    bounds : array-like, shape=(n_dims,2)
        Array of bounds of the prior [lower, upper] along each dimension.

    """

    def __init__(self, bounds):
        self.update_bounds(bounds)

    def update_bounds(self, bounds):
        super().update_bounds(bounds)
        n_d = len(bounds)
        proposal_pdf = scipy.stats.uniform(
            loc=bounds[:, 0], scale=bounds[:, 1] - bounds[:, 0]
        )
        self.proposal_function = partial(proposal_pdf.rvs, size=n_d)

    # Within updated bounds by construction: no need to decorate it.
    def get(self, rng=None):
        return self.proposal_function(random_state=rng)


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
        if random_proposal_fraction > 1.0 or random_proposal_fraction < 0.0:
            raise ValueError(
                "Cannot pass a fraction outside of [0,1]. "
                f"You passed 'random_proposal_fraction={random_proposal_fraction}'"
            )
        if not isinstance(true_proposer, Proposer):
            raise ValueError("The true proposer needs to be a valid proposer.")

        self.rpf = random_proposal_fraction
        # TODO: Make this a sample of the prior instead of uniform hypercube.
        self.random_proposer = UniformProposer(bounds)
        self.true_proposer = true_proposer

    # Not decorating it, assuming the individual ones fulfil the in-bounds criterion,
    # either by construction or because of their own ``check_in_bounds`` decorator.
    def get(self, rng=None):
        rng = check_random_state(rng)
        if rng.random() > self.rpf:
            return self.true_proposer.get(rng=rng)
        return self.random_proposer.get(rng=rng)

    def update(self, surrogate):
        self.true_proposer.update(surrogate)

    def update_bounds(self, bounds):
        self.random_proposer.update_bounds(bounds)
        self.true_proposer.update_bounds(bounds)


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

    include_mean : bool (defaulf: False)
        If True, returns the mean in the first call to ``get`` (only for the 1st MPI rank)
    """

    def __init__(self, bounds, mean, cov, include_mean=False):
        self.update_bounds(bounds)
        self._mean_used = not include_mean
        self._mean = np.array(mean)
        self.proposal_function = scipy.stats.multivariate_normal(
            mean=mean, cov=cov, allow_singular=True
        ).rvs

    @check_in_bounds
    def get(self, rng=None):
        if not self._mean_used:
            self._mean_used = True
            if mpi.is_main_process:
                return self._mean
        return self.proposal_function(random_state=rng)


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

    def __init__(self, bounds, lambd=1.0):
        self.training = None
        self.training_ = None  # in-bounds subset
        self.update_bounds(bounds)
        # TODO: adapt lambda to dimensionality!
        # e.g. 1 seems to work well for d=2, and ~0.5 for d=30
        self.kicking_pdf = scipy.stats.expon(scale=1 / lambd)

    @property
    def d(self):
        """Dimensionality of the prior."""
        return len(self.bounds)

    # No need for check_in_bounds, by construction (uses np.clip)
    def get(self, rng=None):
        rng = check_random_state(rng)
        m = self.d + 1
        # If possible, get points inside bounds, otherwise outside (1st iteration(s))
        try:
            subset = self.training_[
                rng.choice(len(self.training_), size=m, replace=False)
            ]
        except ValueError:  # m > len(training_)
            subset = self.training[
                rng.choice(len(self.training), size=m, replace=False)
            ]
        centroid = np.average(subset, axis=0)
        # perturb the point: per dimension, add a random multiple of the difference
        # between the centroid and one of the points.
        kick = -centroid + np.array(
            [
                subset[j][i]
                for i, j in enumerate(rng.choice(m, size=self.d, replace=False))
            ]
        )
        kick *= self.kicking_pdf.rvs(self.d, random_state=rng)
        # This might have to be modified if the optimizer can't deal with
        # points which are exactly on the edges.
        return np.clip(centroid + kick, self.bounds[:, 0], self.bounds[:, 1])

    def update(self, surrogate):
        # Get training locations from surrogate and save them
        self.training = np.copy(surrogate.X_regress)

    def update_bounds(self, bounds):
        super().update_bounds(bounds)
        # Save training samples inside new bounds
        if self.training is None:
            return
        self.training_ = self.training[is_in_bounds(self.training, bounds)]


# FROM HERE ON, UNUSED, POSSIBLY OUTDATED (and Cobaya-dependent) #########################


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
        from cobaya.cosmo_input.autoselect_covmat import get_best_covmat
        from cobaya.tools import resolve_packages_path

        cmat_dir = get_best_covmat(model_info, packages_path=resolve_packages_path())
        if np.any(d != 0 for d in cmat_dir["covmat"].shape):
            self.proposal_function = scipy.stats.multivariate_normal(
                mean=mean, cov=cmat_dir["covmat"], allow_singular=True
            ).rvs
        else:
            # TODO :: how to gracefully fall back if autocovmat not found
            raise Exception("Autocovmat is not valid")
            # UNDEFINED: model
            self.proposal_function = model.prior.sample

    @check_in_bounds
    def get(self, rng=None):
        return self.proposal_function(random_state=rng)


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
        self.update_bounds(bounds)
        self.nretries = nretries
        self.npoints = npoints
        self.nsteps = nsteps

    def update_bounds(self, bounds):
        super().update_bounds(bounds)
        self.random_proposer = UniformProposer(bounds)

    @check_in_bounds
    def get(self, rng=None):
        if len(self.samples) > 0:
            last, self.samples = self.samples[-1], self.samples[:-1]
            return last
        else:
            self.resample(rng)
            last, self.samples = self.samples[-1], self.samples[:-1]
            return last

    def resample(self, rng=None):
        rng = check_random_state(rng)
        for i in range(self.nretries):
            this_i = rng.choice(range(len(self.surrogate.X_regress)))
            this_X = np.copy(self.surrogate.X_regress[this_i])
            logpost = self.surrogate.y_regress[this_i]
            from cobaya.model import LogPosterior

            self.sampler.current_point.add(this_X, LogPosterior(logpost=logpost))
            # reset random state and number of samples
            self.sampler._rng = rng
            self.sampler.collection.reset()
            try:
                self.sampler.run()
                points = self.sampler.products()["sample"][self.parnames].values
                self.samples = points[:: -self.nsteps]
                return
            except Exception:
                pass
        # if resample_tries failed raise Warning and pass uniform points
        print("[proposer] WARNING: MC chain got stuck. Taking random uniform points")
        self.samples = np.empty((self.nsteps, len(self.bounds)))
        for i in range(self.nsteps):
            self.samples[i] = self.random_proposer.get()

    def update(self, surrogate):
        self.samples = []
        from gpry.mc import mc_sample_from_gp_cobaya

        surr_info, sampler = mc_sample_from_gp_cobaya(
            surrogate,
            self.bounds,
            sampler="mcmc",
            run=False,
            sampler_options={
                "max_samples": self.npoints,
                "max_tries": 10 * self.npoints,
            },
        )
        self.sampler = sampler
        self.parnames = list(surr_info["params"])
        self.surrogate = surrogate
