"""
This module contains several classes and methods for calculating different
convergence criterions which can be used to determine if the BO loop has
converged.

It implements a base :class:`convergence.ConvergenceCriterion` class
of which several inbuilt convergence criteria inherit. Using this base class
it is also possible to construct custom convergence criteria. All convergence criteria
are passed a prior object which is part of the model instance and an options dict.

The fastest one, and reasonably accurate, is :class:`convergence.CorrectCounter`, a local
criterion that checks that the difference between evaluated and predicted values is small
enough, taking into account dimensional regularization.

If mean and covariance matrix of the surrogate posterior are available (e.g. when using
the :class:`NORA` acquisition engine), it is a good idea to combine it with
:class:`GaussianKL`, a global criterion checking stability of KL divergences between
iterations in a Gaussian approximation.

There is also a self-explanatory :class:`DontConverge` criterion, that can be used in
combination with a set training sample budget to ensure that this budget is always
exhausted.

Finally, :class:`TrainAlignment` is a criterion run at the end of the learning loop, that
checks that the result of an MC run on the surrogate posterior represents the mode
described by the training set. This avoids converging on overshoots of the GP Regressor.

Convergence policy
^^^^^^^^^^^^^^^^^^

You can define a ``policy`` for each convergence criterion:

- ``'n'``: necessary (default if not specified)
- ``'s'``: sufficient
- ``'ns'``: necessary and sufficient
- ``'m'``: monitoring (tracked, but will not affect convergence)

If there are no criteria specified as *necessary* or *sufficient*, i.e. all criteria are
set to *monitor*, the run will never converge (but it will stop at evaluation budget
exhaustion).
"""

import sys
import inspect
from copy import deepcopy
from warnings import warn
from abc import ABCMeta, abstractmethod

import numpy as np

from gpry.mc import cobaya_generate_surr_model_input, mcmc_info_from_run
from gpry.tools import (
    kl_norm,
    is_valid_covmat,
    nstd_of_1d_nstd,
    mean_covmat_from_evals,
    credibility_of_nstd,
)
from gpry import mpi

# Policies and default ("necessary") for convergence criteria
_all_convergence_policies_dict = {
    "n": "necessary",
    "s": "sufficient",
    "ns": "necessary and sufficient",
    "m": "monitor",
}
_default_convergence_policy = "n"


class ConvergenceCheckError(Exception):
    """
    Exception to be raised when the computation of the convergence criterion failed.
    """


def builtin_names():
    """
    Lists all names of all built-in convergence criteria.
    """
    list_conv_names = [
        name
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if (
            issubclass(obj.__class__, ConvergenceCriterion.__class__)
            and obj is not ConvergenceCriterion
        )
    ]
    return list_conv_names


class ConvergenceCriterion(metaclass=ABCMeta):
    """Base class for all convergence criteria (CCs). A CC quantifies the
    convergence of the surrogate model. If this value goes below a certain,
    user-set value we consider the surrogate to have converged to the true posterior
    distribution.

    Currently several CCs are supported which should be versatile enough for
    most tasks. If however one wants to specify a custom CC
    it should be a class which inherits from this abstract class.
    This class needs to be of the format::

        from gpry.convergence import ConvergenceCriterion
        class Custom_convergence_criterion(ConvergenceCriterion):
            def __init__(self, prior_bounds, params):
                # prior_bounds should be a list of prior bounds for all parameters.
                # As a minimal requirement this method should set a number
                # at which the algorithm is considered to have converged.
                # Furthermore this method should initialize empty lists in
                # which we can write the values of the convergence criterion
                # as well as the total number of posterior evaluations and the
                # number of accepted posterior evaluations. This allows for
                # easy tracking/plotting of the convergence.
                self.values = []
                self.n_posterior_evals = []
                self.n_accepted_evals = []
                self.limit = ... # stores the limit for convergence

            def is_converged(self, surr, surr_2=None, new_X=None, new_y=None, pred_y=None):
                # Basically a wrapper for the 'criterion_value' method which
                # returns True if the convergence criterion is met and False
                # otherwise.

            def criterion_value():
                # Returns the value of the convergence criterion. Should also
                # append the current value and the number of posterior
                # evaluations to the corresponding variables.
    """

    @abstractmethod
    def __init__(self, prior_bounds, params):
        """Sets all relevant initial parameters from the 'params' dict."""
        self.values = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []
        self._set_convergence_policy(params)

    def get_history(self):
        """Returns the two lists containing the values of the convergence
        criterion at each step as well as the total number of evaluations and
        the number of accepted evaluations.
        """
        try:
            values = self.values
            n_posterior_evals = self.n_posterior_evals
            n_accepted_evals = self.n_accepted_evals
        except Exception as excpt:
            raise AttributeError(
                "The convergence criterion does not save its convergence history."
            ) from excpt
        if len(values) == 0 or len(n_posterior_evals) == 0:
            raise ValueError(
                "Make sure to call the convergence criterion "
                "before getting it's history."
            )
        return values, n_posterior_evals, n_accepted_evals

    @abstractmethod
    def is_converged(
        self, surr, surr_2=None, new_X=None, new_y=None, pred_y=None, acquisition=None
    ):
        """
        Returns False if the algorithm hasn't converged and True if it has.

        If surr_2 is None the last surrogate is taken from the model instance.
        """

    @abstractmethod
    def criterion_value(self, surr, surr_2=None):
        """
        Returns the value of the convergence criterion for the current
        surrogate. If surr_2 is None the last surrogate is taken from the model instance.
        """

    @property
    def last_value(self):
        """Last value of the convergence criterion."""
        return deepcopy(self.values[-1])

    @property
    def is_MPI_aware(self):
        """
        Should return True if the convergence criterion should run in multiple processes
        using MPI communication.
        """
        return False

    def _set_convergence_policy(self, params):
        self._convergence_policy = (params or {}).get(
            "policy", _default_convergence_policy
        )
        try:
            self._convergence_policy = self._convergence_policy.lower()
            if self._convergence_policy not in _all_convergence_policies_dict:
                raise ValueError()
        except (AttributeError, ValueError) as excpt:
            raise ValueError(
                "Convergence 'policy' must be one of the following strings: "
                f"{_all_convergence_policies_dict}. Got {self._convergence_policy}."
            ) from excpt

    @property
    def convergence_policy(self):
        """
        Returns a string describing the convergence policy.
        """
        return self._convergence_policy

    @property
    def convergence_policy_MPI(self):
        """
        Returns a string describing the convergence policy (MPI-wrapped!)
        """
        if self.is_MPI_aware or mpi.is_main_process:
            convergence_policy = self._convergence_policy
        else:
            convergence_policy = None
        return mpi.bcast(convergence_policy)

    def is_converged_MPIwrapped(self, *args, **kwargs):
        """
        MPI-aware wrapper for calling is_converged inside the runner.

        Returns convergence criterion value for process 0.

        If fails, raises ConvergenceCheckError in all processes.
        """
        mpi.sync_processes()  # for timing
        failed = False
        has_converged = None
        err_msg = None
        if self.is_MPI_aware or mpi.is_main_process:
            try:
                has_converged = self.is_converged(*args, **kwargs)
            except ConvergenceCheckError as excpt:
                failed = True
                err_msg = str(excpt)
        if any(mpi.allgather(failed)):
            # Take lowest-rank non-null error message
            err_msg = [msg for msg in mpi.allgather(err_msg) if msg is not None][0]
            raise ConvergenceCheckError(err_msg)
        return mpi.bcast(has_converged)


class DummyMPIConvergeCriterion(ConvergenceCriterion):
    """
    Class to be held by non-0 rank MPI processes if the corresponding criterion is not
    MPI-aware, in order to simplify the code.
    """

    def __init__(self):
        pass

    def criterion_value(self, *args, **kwargs):
        raise TypeError("This method should not be called for this class.")

    def is_converged(self, *args, **kwargs):
        raise TypeError("This method should not be called for this class.")

    @property
    def last_value(self):
        """Last value of the convergence criterion."""
        return np.nan


class DontConverge(ConvergenceCriterion):
    """
    This convergence criterion is mainly for testing purposes and always
    returns False when ``is_converged`` is called. Use this method together
    with the ``max_points`` and ``max_accepted`` keys in the options dict to stop
    the BO loop at a set number of iterations.
    """

    def __init__(self, prior_bounds=None, params=None):
        self.values = []
        self.limit = np.nan
        self.thres = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []
        self.prior_bounds = prior_bounds
        # Explicitly set "necessary" as policy
        self._set_convergence_policy({"policy": "n"})

    def criterion_value(self, surr, surr_2=None):
        self.values.append(np.nan)
        self.thres.append(np.nan)
        self.n_posterior_evals.append(surr.n_total)
        self.n_accepted_evals.append(surr.n_regress)
        return np.nan

    def is_converged(
        self, surr, surr_2=None, new_X=None, new_y=None, pred_y=None, acquisition=None
    ):
        self.criterion_value(surr, surr_2)
        return False


class GaussianKL(ConvergenceCriterion):
    """
    This criterion estimates convergence as stability of the Gaussian-approximated,
    single-mode KL divergence of surrogate posterior samples between runs.

    If a valid GPAcquisition instance is passed to ``is_converged``, mean and covariance
    will be extracted from it. Otherwise, it estimates the mean and covariance by running
    an MCMC sampler on the surrogate (slow).

    In the second case, this convergence criterion is MPI-aware, such that it will run as
    many parallel MCMC chains as running processes to improve the estimation of the mean
    and covariance.

    Parameters
    ----------
    prior_bounds : list
        List of prior bounds.

    params : dict
        Dict with the following keys:

        * ``"limit"``: Value of the KL divergence for which we consider the algorithm
                       converged (default ``2e-2``).
        * ``"limit_times"``: Number of consecutive times that the KL divergence must be
                             lower than the ``limit`` parameter (default ``2``).
        * ``"n_draws"``: Number of steps of the MCMC chain (default: ignored in favour of
                         ``"n_draws_per_dimsquared"``).
        * ``"n_draws_per_dimsquared"``: idem, as a factor of the dimensionality squared
                                        (default 10).
        * ``"max_reused"``: number of times a sample can be reweighted and reused (may
                            miss new high-value regions) (default 4).
    """

    @property
    def is_MPI_aware(self):
        return True

    def __init__(self, prior_bounds, params):
        self.prior_bounds = prior_bounds
        self.mean = None
        self.cov = None
        # The limit cannot be too strict, since the NS sample is not so stable itself
        self.limit = params.get("limit", 2e-2)
        d = len(self.prior_bounds)
        # Needs to at least encompass 2 full MC samples -- TODO: fix in run.py at init
        self.limit_times = int(np.round(params.get("limit_times", d)))
        self._set_convergence_policy(params)
        self.values = []
        self.thres = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []
        # Number of MCMC chains to generate samples
        if params.get("n_draws") and params.get("n_draws_per_dimsquared"):
            raise ValueError(
                "Pass either 'n_draws' or 'n_draws_per_dimsquared', not both"
            )
        if params.get("n_draws"):
            self._n_draws = int(params.get("n_draws"))
        else:
            self.n_draws_per_dimsquared = params.get("n_draws_per_dimsquared", 10)
        # Max times a sample can be reweighted and reused (we may miss new high regions)
        self.max_reused = params.get("max_reused", 4)
        self.n_reused = 0
        # We'll some hight MCMC temperature, to get the tails right
        self.temperature = 2
        # Prepare Cobaya's input
        self.paramnames = [f"x_{i + 1}" for i in range(d)]
        self.cobaya_input = None
        # Save last sample
        self._last_info = {}
        self._last_collection = None

    def _get_new_mean_and_cov(self, surr, acquisition=None):
        try:
            return self._get_new_mean_and_cov_from_acquisition(acquisition)
        except AttributeError:
            warn(
                "Could not get sample from acquisition object. "
                "Running MC process to get mean and covmat."
            )
            return self._get_new_mean_and_cov_from_mc(surr)

    def _get_new_mean_and_cov_from_acquisition(self, acquisition):
        """
        Tries to extract the mean and covmat from the acquisition object.

        Raises AttributeError for null acquisition object or it not having samples.
        """
        mean, cov = None, None
        attr_error, num_error = None, None
        if mpi.is_main_process:
            try:
                X, _, _, w = acquisition.last_MC_sample(warn_reweight=False)
            except AttributeError as excpt:
                attr_error = excpt
            else:
                try:
                    mean = np.average(X, weights=w, axis=0)
                    cov = np.atleast_2d(np.cov(X.T, aweights=w, ddof=0))
                except (ValueError, TypeError) as excpt:
                    num_error = excpt
        attr_error = mpi.bcast(attr_error)
        if attr_error:
            raise AttributeError from attr_error  # all processes!
        num_error = mpi.bcast(num_error)
        if num_error:
            raise ConvergenceCheckError(
                f"Numerical error when computing new mean and cov: {num_error}"
            ) from num_error
        return mpi.bcast((mean, cov) if mpi.is_main_process else None)

    def _get_new_mean_and_cov_from_mc(self, surr):
        self.thres.append(self.limit)
        cov_mcmc = None
        if mpi.is_main_process:
            reused = False
            if self._last_collection is not None:
                points = self._last_collection[self.paramnames].to_numpy(float)
                old_surr_values = -0.5 * self._last_collection["chi2"].to_numpy(float)
                new_surr_values = surr.predict(points)
                weights = self._last_collection["weight"].to_numpy(float)
                logratio = new_surr_values - old_surr_values
                logratio -= max(logratio)
                reweights = weights * np.exp(logratio)
                # Remove points with very small weight: more numerically stable
                i_nonzero = np.argwhere(reweights > 1e-8).T[0]
                reweights = reweights[i_nonzero]
                points = points[i_nonzero]
                mean_reweighted = np.average(points, weights=reweights, axis=0)
                cov_reweighted = np.cov(points.T, aweights=reweights)
                cov_mcmc = cov_reweighted
                # Use max of them
                try:
                    kl_reweight = max(
                        kl_norm(mean_reweighted, cov_reweighted, self.mean, self.cov),
                        kl_norm(self.mean, self.cov, mean_reweighted, cov_reweighted),
                    )
                except np.linalg.LinAlgError as excpt:
                    raise ConvergenceCheckError(
                        f"Could not compute KL norm: {excpt}."
                    ) from excpt
                # If very small, we've probably found nothing yet, so nothing new
                # But assume that if we have hit 10 * limit, we are right on track
                min_kl = self.limit * 1e-2 if max(self.values) < 10 * self.limit else 0
                # If larger than the difference with the last one, bad
                max_kl = self.values[-1]
                if min_kl < kl_reweight < max_kl and self.n_reused < self.max_reused:
                    self.n_reused += 1
                    reused = True
        if mpi.multiple_processes:
            reused = mpi.bcast(reused if mpi.is_main_process else None)
        if reused:
            if mpi.multiple_processes:
                mean_reweighted, cov_reweighted = mpi.bcast(
                    (mean_reweighted, cov_reweighted) if mpi.is_main_process else None
                )
            return mean_reweighted, cov_reweighted
        # No previous mcmc sample, or reweighted mean+cov too different
        self._last_info, samples = self._sample_mcmc(surr, covmat=cov_mcmc)
        # Compute mean and cov, and broadcast
        if mpi.is_main_process:
            mean_new, cov_new = samples.mean(), samples.cov()
            # Only main process caches this one, to save memory
            self._last_collection = samples
        # Broadcast results
        if mpi.multiple_processes:
            mean_new, cov_new = mpi.bcast(
                (mean_new, cov_new) if mpi.is_main_process else None
            )
        return mean_new, cov_new

    def _sample_mcmc(self, surr, covmat=None):
        from cobaya.model import get_model  # type: ignore
        from cobaya.sampler import get_sampler  # type: ignore
        from cobaya.log import LoggedError  # type: ignore

        # Update Cobaya's input: mcmc's proposal covmat and log-likelihood
        self.cobaya_input = cobaya_generate_surr_model_input(
            surr, self.prior_bounds, self.paramnames
        )
        # Supress Cobaya's output
        # (set to True for debug output, or comment out for normal output)
        self.cobaya_input["debug"] = 50
        # Create model and sampler
        model = get_model(self.cobaya_input)
        if covmat is not None and is_valid_covmat(covmat):
            cov = covmat
        else:
            cov = self.cov
        sampler_info = mcmc_info_from_run(model, surr, cov=cov)
        sampler_info["mcmc"]["temperature"] = self.temperature
        high_prec_threshold = (self.values[-1] < 1) if len(self.values) > 0 else False
        # Relax stopping criterion if not yet well converged
        sampler_info["mcmc"].update(
            {
                "Rminus1_stop": (0.01 if high_prec_threshold else 0.2),
                "Rminus1_cl_stop": (0.2 if high_prec_threshold else 0.5),
            }
        )
        mcmc_sampler = get_sampler(sampler_info, model)
        try:
            mcmc_sampler.run()
            success = True
        except LoggedError:
            success = False
        if mpi.multiple_processes:
            success = all(mpi.allgather(success))
        if not success:
            raise ConvergenceCheckError
        updated_info = model.info()
        updated_info["sampler"] = {"mcmc": mcmc_sampler.info()}
        samples = mcmc_sampler.samples(combined=True, skip_samples=0.33)
        samples.reset_temperature()
        return updated_info, samples

    def criterion_value(self, surr, surr_2=None, acquisition=None):
        try:
            mean_new, cov_new = self._get_new_mean_and_cov(
                surr, acquisition=acquisition
            )
        except ConvergenceCheckError as excpt:
            self.values.append(np.nan)
            self.n_posterior_evals.append(surr.n_total)
            self.n_accepted_evals.append(surr.n_regress)
            raise ConvergenceCheckError(
                f"Error when computing mean and covmat: {excpt}"
            ) from excpt
        if surr_2 is not None:
            # TODO: Nothing yet to do with surr_2
            pass
        if self.mean is None or self.cov is None:
            # Nothing to compare to! But save mean, cov for next call
            self.mean, self.cov = mean_new, cov_new
            self.values.append(np.nan)
            self.n_posterior_evals.append(surr.n_total)
            self.n_accepted_evals.append(surr.n_regress)
            raise ConvergenceCheckError(
                "No previous call: cannot compute criterion yet."
            )
        else:
            mean_old, cov_old = np.copy(self.mean), np.copy(self.cov)
        # Compute the KL divergence (gaussian approx) with the previous iteration
        try:
            kl = kl_norm(mean_new, cov_new, mean_old, cov_old)
            if kl < 0:
                raise ValueError("Negative KL -> undefined")
            self.mean = mean_new
            self.cov = cov_new
            self.values.append(kl)
            self.n_posterior_evals.append(surr.n_total)
            self.n_accepted_evals.append(surr.n_regress)
        except Exception as excpt:
            kl = np.nan
            self.mean = mean_new
            self.cov = cov_new
            self.values.append(kl)
            self.n_posterior_evals.append(surr.n_total)
            self.n_accepted_evals.append(surr.n_regress)
            raise ConvergenceCheckError(f"Computation error in KL: {excpt}") from excpt
        return kl

    def is_converged(
        self, surr, surr_2=None, new_X=None, new_y=None, pred_y=None, acquisition=None
    ):
        self.criterion_value(surr, surr_2, acquisition)
        try:
            if np.all(np.abs(np.array(self.values[-self.limit_times :])) < self.limit):
                return True
        except IndexError:
            pass
        return False

    # Safe copying and pickling
    def __getstate__(self):
        return deepcopy(self).__dict__

    def __deepcopy__(self, memo=None):
        # Remove non-picklable surrogate model likelihood
        if self.cobaya_input and "likelihood" in self.cobaya_input:
            like = list(self.cobaya_input["likelihood"])[0]
            self.cobaya_input["likelihood"][like]["external"] = True
            if self._last_info and "likelihood" in self._last_info:
                self._last_info["likelihood"][like]["external"] = True
        new = (lambda cls: cls.__new__(cls))(self.__class__)
        new.__dict__ = {k: deepcopy(v) for k, v in self.__dict__.items() if k != "log"}
        return new


class GaussianKLTrain(GaussianKL):
    """
    This criterion is not aimed at estimating convergence, but at discarding cases in
    which a MC sample from the surrogate model (the last one obtained by the acquisition step, if it
    exists, otherwise computed on the fly) would not sample the mode mapped by the
    training set, but instead some overshooting or large baseline plateau. It compares the
    Gaussian approximation of the last MC sample by the acquisition step with the mean and
    covariance matrix computed from the training set using probabilities as weights.

    Since its a check in the current iteration, by default it is enough for this criterion
    to be satisfied in the last step, and with a high tolerance, since it affects extreme
    cases only.

    At the moment, it assumes that there is a single mode.

    If a valid GPAcquisition instance is passed to ``is_converged``, mean and covariance
    will be extracted from it. Otherwise, it estimates the mean and covariance by running
    an MCMC sampler on the surrogate (slow).

    In the second case, this convergence criterion is MPI-aware, such that it will run as
    many parallel MCMC chains as running processes to improve the estimation of the mean
    and covariance.

    Parameters
    ----------
    prior_bounds : list
        List of prior bounds.

    params : dict
        Dict with the following keys:

        * ``"limit"``: Value of the KL divergence for which we consider the algorithm
                       converged (default ``2e-2``).
        * ``"limit_times"``: Number of consecutive times that the KL divergence must be
                             lower than the ``limit`` parameter (default ``2``).
        * ``"n_draws"``: Number of steps of the MCMC chain (default: ignored in favour of
                         ``"n_draws_per_dimsquared"``).
        * ``"n_draws_per_dimsquared"``: idem, as a factor of the dimensionality squared
                                        (default 10).
        * ``"max_reused"``: number of times a sample can be reweighted and reused (may
                            miss new high-value regions) (default 4).
    """

    def __init__(self, prior_bounds, params):
        params = params or {}
        if params.get("limit") is None:
            params["limit"] = len(prior_bounds)
        if params.get("limit_times") is None:
            params["limit_times"] = 2
        super().__init__(prior_bounds, params)

    def _get_mean_and_cov_from_training(self, surr):
        return mean_covmat_from_evals(surr.X_regress, surr.y_regress)

    def criterion_value(self, surr, surr_2=None, acquisition=None):
        try:
            mean_new, cov_new = self._get_new_mean_and_cov(
                surr, acquisition=acquisition
            )
        except ConvergenceCheckError as excpt:
            self.values.append(np.nan)
            self.n_posterior_evals.append(surr.n_total)
            self.n_accepted_evals.append(surr.n_regress)
            raise ConvergenceCheckError(
                f"Error when computing mean and covmat: {excpt}"
            ) from excpt
        try:
            mean_training, cov_training = self._get_mean_and_cov_from_training(surr)
        except Exception as excpt:
            self.values.append(np.nan)
            self.n_posterior_evals.append(surr.n_total)
            self.n_accepted_evals.append(surr.n_regress)
            raise ConvergenceCheckError(
                f"Error when computing mean and covmat from training: {excpt}"
            ) from excpt
        if surr_2 is not None:
            # TODO: Nothing yet to do with surr_2
            pass
        # Compute the KL divergence (gaussian approx) between with the previous iteration
        try:
            kl = kl_norm(mean_new, cov_new, mean_training, cov_training)
            if kl < 0:
                raise ValueError("Negative KL -> undefined")
            self.mean = mean_new
            self.cov = cov_new
            self.values.append(kl)
            self.n_posterior_evals.append(surr.n_total)
            self.n_accepted_evals.append(surr.n_regress)
        except Exception as excpt:
            kl = np.nan
            self.mean = mean_new
            self.cov = cov_new
            self.values.append(kl)
            self.n_posterior_evals.append(surr.n_total)
            self.n_accepted_evals.append(surr.n_regress)
            raise ConvergenceCheckError(f"Computation error in KL: {excpt}") from excpt
        return kl


class TrainAlignment(GaussianKL):
    """
    This criterion is not aimed at estimating convergence, but at discarding cases in
    which a MC sample from the surrogate model (the last one obtained by the acquisition step, if it
    exists, otherwise computed on the fly) would not sample the mode mapped by the
    training set, but instead some overshooting or large baseline plateau.

    It computes the minimum central confidence level of the mean of the training set with
    respect to a Gaussian approximation of the surrogate posterior.

    Its maximum value is obviously 1, and for the kind of test that this criterion
    addresses a value below 0.5 should be enough. It's minimum value is clipped at 0.001,
    to avoid spoiling the convergence plots with numerical noise.

    Since its a check in the current iteration, by default it is enough for this criterion
    to be satisfied in the last step, and with a high tolerance, since it affects extreme
    cases only.

    At the moment, it assumes that there is a single mode.

    If a valid GPAcquisition instance is passed to ``is_converged``, mean and covariance
    will be extracted from it. Otherwise, it estimates the mean and covariance by running
    an MCMC sampler on the surrogate (slow).

    In the second case, this convergence criterion is MPI-aware, such that it will run as
    many parallel MCMC chains as running processes to improve the estimation of the mean
    and covariance.

    Parameters
    ----------
    prior_bounds : list
        List of prior bounds.

    params : dict
        Dict with the following keys:

        * ``"frac_training"``: fraction, starting from the latest, of the training set to
                               be used (default: 1)
        * ``"limit"``: Probability mass within the minimum CL enclosing the training mean
                       (default ``0.5``).
        * ``"limit_times"``: Number of consecutive times that the criterion must be
                             fulfilled (default ``1``).
        * ``"n_draws"``: Number of steps of the MCMC chain (default: ignored in favour of
                         ``"n_draws_per_dimsquared"``).
        * ``"n_draws_per_dimsquared"``: idem, as a factor of the dimensionality squared
                                        (default 10).
        * ``"max_reused"``: number of times a sample can be reweighted and reused (may
                            miss new high-value regions) (default 4).
    """

    def __init__(self, prior_bounds, params):
        params = params or {}
        self.frac_training = params.get("frac_training", 1)
        if params.get("limit") is None:
            params["limit"] = 0.5
        if params.get("limit_times") is None:
            params["limit_times"] = 1
        super().__init__(prior_bounds, params)

    def _get_mean_from_training(self, surr):
        Nfrac = int(surr.n_regress * self.frac_training)
        return mean_covmat_from_evals(surr.X_regress[-Nfrac:], surr.y_regress[-Nfrac:])[
            0
        ]

    @staticmethod
    def criterion_value_from_means_cov(mean1, mean2, cov):
        mean_diff = mean1 - mean2
        chi2 = mean_diff @ np.linalg.inv(cov) @ mean_diff
        return credibility_of_nstd(np.sqrt(chi2), len(mean1))

    def criterion_value(self, surr, surr_2=None, acquisition=None):
        try:
            mean_new, cov_new = self._get_new_mean_and_cov(
                surr, acquisition=acquisition
            )
        except ConvergenceCheckError as excpt:
            self.values.append(np.nan)
            self.n_posterior_evals.append(surr.n_total)
            self.n_accepted_evals.append(surr.n_regress)
            raise ConvergenceCheckError(
                f"Error when computing mean and covmat: {excpt}"
            ) from excpt
        try:
            mean_training = self._get_mean_from_training(surr)
        except Exception as excpt:
            self.values.append(np.nan)
            self.n_posterior_evals.append(surr.n_total)
            self.n_accepted_evals.append(surr.n_regress)
            raise ConvergenceCheckError(
                f"Error when computing mean and covmat from training: {excpt}"
            ) from excpt
        if surr_2 is not None:
            # TODO: Nothing yet to do with surr_2
            pass
        # Compute the CL corresponding to the Chi-squared of the mean of the training set
        try:
            eps = self.criterion_value_from_means_cov(mean_new, mean_training, cov_new)
            if eps < 0:
                raise ValueError("Negative credibility -> undefined")
            eps = max(eps, 1e-3)  # cap to avoid plotting numerical noise
            self.mean = mean_new
            self.cov = cov_new
            self.values.append(eps)
            self.n_posterior_evals.append(surr.n_total)
            self.n_accepted_evals.append(surr.n_regress)
        except Exception as excpt:
            eps = np.nan
            self.mean = mean_new
            self.cov = cov_new
            self.values.append(eps)
            self.n_posterior_evals.append(surr.n_total)
            self.n_accepted_evals.append(surr.n_regress)
            raise ConvergenceCheckError(
                f"Computation error in train mean alignment: {excpt}"
            ) from excpt
        return eps


class CorrectCounter(ConvergenceCriterion):
    r"""
    This convergence criterion determines convergence by requiring that the surrogate's
    predictions of the posterior values in the last :math:`n` steps are correct up to a
    certain threshold. This condition is fulfilled if

    .. math::

        |f(x)-\overline{f}_{\mathrm{GP}}(x)| < (f_{\mathrm{max}}(x) - f(x)) \cdot r + a

    where the parameters :math:`r` and :math:`a` are the relative and absolute
    tolerances controlled by the `reltol` and `abstol` parameters.
    We set the "value" of the criterion to be the maximum difference of the surrogate's
    prediction and the true posterior in the last batch of accepted evaluations.
    Furthermore this class contains an internal list `thres` which contains the
    threshold values corresponding to this difference.

    Parameters
    ----------
    prior_bounds : list
        List of prior bounds.

    params : dict
        Dict with the following keys:

        * ``"n_correct"``: Number of consecutive samples which need to be under the
                           threshold (default ``max(4, 0.5*N_d)``)
        * ``"reltol"``: Relative tolerance parameter (default ``0.01``)
        * ``"abstol"``: Absolute tolerance parameter (default ``"0.01s"``)
        * ``"verbose"``: Verbosity

        .. note::

            The ``"reltol"`` and ``"abstol"`` parameters can be passed as a string ending
            with either ``"l"`` or ``"s"``. In this case the value of this parameter is
            scaled with the number of dimensions as either linear (``"l"``) or square
            (``"s"``) of the depth of the :math:`\chi^2` of the :math:`1-\sigma`-contour
            assuming a gaussian distribution.

    """

    def __init__(self, prior_bounds, params):
        d = len(prior_bounds)
        self.n_correct = params.get("n_correct", max(4, np.ceil(0.5 * d)))
        reltol = params.get("reltol", 0.01)
        if isinstance(reltol, str):
            try:
                assert reltol[-1] == "l" or reltol[-1] == "s" or reltol[-1] == "r"
                if reltol[-1] == "l":
                    reltol = float(reltol[:-1]) * nstd_of_1d_nstd(1, d)
                elif reltol[-1] == "s":
                    reltol = float(reltol[:-1]) * nstd_of_1d_nstd(1, d) ** 2.0
                elif reltol[-1] == "r":
                    reltol = float(reltol[:-1]) * np.sqrt(nstd_of_1d_nstd(1, d))
            except Exception as excpt:
                raise ValueError(
                    "The 'reltol' parameter can either be a number "
                    + f"or a string with a number followed by 'l' or 's'. Got {reltol}"
                ) from excpt
        self.reltol = reltol
        abstol = params.get("abstol", "0.01s")
        if isinstance(abstol, str):
            try:
                assert abstol[-1] == "l" or abstol[-1] == "s" or abstol[-1] == "r"
                if abstol[-1] == "l":
                    abstol = float(abstol[:-1]) * nstd_of_1d_nstd(1, d)
                elif abstol[-1] == "s":
                    abstol = float(abstol[:-1]) * nstd_of_1d_nstd(1, d) ** 2.0
                elif abstol[-1] == "r":
                    abstol = float(abstol[:-1]) * np.sqrt(nstd_of_1d_nstd(1, d))
            except Exception as excpt:
                raise ValueError(
                    "The 'abstol' parameter can either be a number "
                    + f"or a string with a number followed by 'l' or 's'. Got {abstol}"
                ) from excpt
        self.abstol = abstol
        self.verbose = params.get("verbose", 0)
        self._set_convergence_policy(params)
        self.values = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []
        self.thres = []
        self.n_pred = 0

    def is_converged(
        self, surr, surr_2=None, new_X=None, new_y=None, pred_y=None, acquisition=None
    ):
        self.criterion_value(surr, new_X=new_X, new_y=new_y, pred_y=pred_y)
        return self.n_pred > self.n_correct

    def criterion_value(self, surr, surr_2=None, new_X=None, new_y=None, pred_y=None):
        n_new = len(new_y)
        assert n_new == len(pred_y)
        max_val = 0
        max_diff = 0
        max_thres = 0
        for yn, yl in zip(new_y, pred_y):
            # Remove warning case that does not trigger any condition below
            if yn == -np.inf:
                continue
            diff = np.abs(yl - yn)
            # rel_difference = np.abs((yl - surr.y_max) / (yn - surr.y_max) - 1.)
            thres = np.abs(yn - surr.y_max) * self.reltol + self.abstol
            if diff / thres > max_val:
                max_val = diff / thres
                max_diff = diff
                max_thres = thres
            if diff < thres:
                self.n_pred += 1
                if self.verbose > 0:
                    print(f"Already {self.n_pred} correctly predicted \n")
            else:
                self.n_pred = 0
                if self.verbose > 0:
                    print("Mispredict...")
        self.values.append(max_diff if n_new > 0 else self.values[-1])
        self.thres.append(max_thres if n_new > 0 else self.thres[-1])
        self.n_posterior_evals.append(surr.n_total)
        self.n_accepted_evals.append(surr.n_regress)
        return max_val if n_new > 0 else self.values[-1]

    @property
    def limit(self):
        """Limit for the criterion value (changes along iterations for this criterion)."""
        return self.thres[-1]
