"""
This module contains several classes and methods for calculating different
convergence criterions which can be used to determine if the BO loop has
converged.
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import sys
import inspect
from copy import deepcopy
from gpry.mc import cobaya_generate_gp_model_input, mcmc_info_from_run
from gpry.tools import kl_norm, is_valid_covmat, nstd_of_1d_nstd
from gpry.mpi import mpi_comm, is_main_process, multiple_processes


class ConvergenceCheckError(Exception):
    """
    Exception to be raised when the computation of the convergence criterion failed.
    """

    pass


def builtin_names():
    """
    Lists all names of all built-in convergence criteria.
    """
    list_conv_names = [name for name, obj in inspect.getmembers(sys.modules[__name__])
                       if (issubclass(obj.__class__, ConvergenceCriterion.__class__) and
                           obj is not ConvergenceCriterion)]
    return list_conv_names


class ConvergenceCriterion(metaclass=ABCMeta):
    """Base class for all convergence criteria (CCs). A CC quantifies the
    convergence of the GP surrogate model. If this value goes below a certain,
    user-set value we consider the GP to have converged to the true posterior
    distribution.

    Currently several CCs are supported which should be versatile enough for
    most tasks. If however one wants to specify a custom CC
    it should be a class which inherits from this abstract class.
    This class needs to be of the format::

        from gpry.convergence import ConvergenceCriterion
        class Custom_convergence_criterion(ConvergenceCriterion):
            def __init__(self, prior, params):
                # prior should be a prior object and contain the prior for all
                # parameters
                # params is to be passed as a dictionary. The init should
                # then set the parameters which are needed later accordingly.
                # as a minimal requirement this method should set a number
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

            def is_converged(self, gp, gp_old=None, new_X=None, new_y=None, pred_y=None):
                # Basically a wrapper for the 'criterion_value' method which
                # returns True if the convergence criterion is met and False
                # otherwise.

            def criterion_value():
                # Returns the value of the convergence criterion. Should also
                # append the current value and the number of posterior
                # evaluations to the corresponding variables.
    """

    def get_history(self):
        """Returns the two lists containing the values of the convergence
        criterion at each step as well as the total number of evaluations and
        the number of accepted evaluations.
        """
        try:
            values = self.values
            n_posterior_evals = self.n_posterior_evals
            n_accepted_evals = self.n_accepted_evals
        except Exception:
            raise AttributeError("The convergence criterion does not save it's "
                                 "convergence history.")
        if len(values) == 0 or len(n_posterior_evals) == 0:
            raise ValueError("Make sure to call the convergence criterion "
                             "before getting it's history.")
        return values, n_posterior_evals, n_accepted_evals

    @abstractmethod
    def __init__(self, prior, params):
        """Sets all relevant initial parameters from the 'params' dict."""

    @abstractmethod
    def is_converged(self, gp, gp_old=None, new_X=None, new_y=None, pred_y=None):
        """
        Returns False if the algorithm hasn't converged and True if it has.

        If gp_2 is None the last GP is taken from the model instance.
        """

    @abstractmethod
    def criterion_value(self, gp, gp_2=None):
        """
        Returns the value of the convergence criterion for the current
        gp. If gp_2 is None the last GP is taken from the model instance.
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


class DontConverge(ConvergenceCriterion):
    """
    This convergence criterion is mainly for testing purposes and always
    returns False when ``is_converged`` is called. Use this method together
    with the ``max_points`` and ``max_accepted`` keys in the options dict to stop
    the BO loop at a set number of iterations.
    """

    def __init__(self, prior, params):
        self.values = []
        self.thres = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []
        self.prior = prior

    def criterion_value(self, gp, gp_2=None):
        self.values.append(np.nan)
        self.thres.append(np.nan)
        self.n_posterior_evals.append(gp.n_total)
        self.n_accepted_evals.append(gp.n)
        return np.nan

    def is_converged(self, gp, gp_2=None, new_X=None, new_y=None, pred_y=None):
        self.criterion_value(gp, gp_2)
        return False


class GaussianKL(ConvergenceCriterion):
    """
    This convergence criterion estimates the mean and covariance of a mode of the GP
    (assumed uni-modal) by running an MCMC sampler on the GP, and compares this to the
    mean and covariance of a previous step (or a given reference GP) via the KL divergence
    between Gaussian distributions, which defines the criterion value.

    It is robust, but slow.

    This convergence criterion is MPI-aware, such that it will run as many parallel MCMC
    chains as running processes to improve the estimation of the mean and covariance.

    Parameters
    ----------
    prior : model prior instance
        The prior of the model used. This is not needed in this specific
        convergence criterion so you may pass anything here.

    params : dict
        Dict with the following keys:

        * ``"limit"``: Value of the KL divergence for which we consider the algorithm
                       converged (default ``1e-2``).
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

    def __init__(self, prior, params):
        self.prior = prior
        self.mean = None
        self.cov = None
        self.limit = params.get("limit", 1e-2)
        self.limit_times = params.get("limit_times", 2)
        self.values = []
        self.thres = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []
        # Number of MCMC chains to generate samples
        if params.get("n_draws") and params.get("n_draws_per_dimsquared"):
            raise ValueError(
                "Pass either 'n_draws' or 'n_draws_per_dimsquared', not both")
        if params.get("n_draws"):
            self._n_draws = int(params.get("n_draws"))
        else:
            self.n_draws_per_dimsquared = params.get("n_draws_per_dimsquared", 10)
        # Max times a sample can be reweighted and reused (we may miss new high regions)
        self.max_reused = params.get("max_reused", 4)
        # TODO: restore temperature
        # MCMC temperature
        # self.temperature = params.get("temperature", 2)
        # Prepare Cobaya's input
        self.bounds = self.prior.bounds(confidence_for_unbounded=0.99995)
        self.paramnames = self.prior.params
        self.cobaya_input = None

        # Save last sample
        self._last_info = {}
        self._last_collection = None

    @property
    def cobaya_param_names(self):
        return self.paramnames

    def _get_new_mean_and_cov(self, gp):
        self.thres.append(self.limit)
        cov_mcmc = None
        if is_main_process:
            reused = False
            if self._last_collection is not None:
                points = \
                    self._last_collection[self.cobaya_param_names].to_numpy(np.float64)
                old_gp_values = -0.5 * \
                    self._last_collection["chi2"].to_numpy(np.float64)
                new_gp_values = gp.predict(points)
                weights = self._last_collection["weight"].to_numpy(np.float64)
                logratio = new_gp_values - old_gp_values
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
                        kl_norm(self.mean, self.cov, mean_reweighted, cov_reweighted))
                except np.linalg.LinAlgError as excpt:
                    raise ConvergenceCheckError(f"Could not compute KL norm: {excpt}.")
                # If very small, we've probably found nothing yet, so nothing new
                # But assume that if we have hit 10 * limit, we are right on track
                min_kl = self.limit * 1e-2 if max(self.values) < 10 * self.limit else 0
                # If larger than the difference with the last one, bad
                max_kl = self.values[-1]
                if kl_reweight > min_kl and kl_reweight < max_kl and \
                   self.n_reused < self.max_reused:
                    self.n_reused += 1
                    reused = True
        if multiple_processes:
            reused = mpi_comm.bcast(reused if is_main_process else None)
        if reused:
            if multiple_processes:
                mean_reweighted, cov_reweighted = mpi_comm.bcast(
                    (mean_reweighted, cov_reweighted) if is_main_process else None)
            return mean_reweighted, cov_reweighted
        # No previous mcmc sample, or reweighted mean+cov too different
        self.n_reused = 0
        self._last_info, collection = self._sample_mcmc(gp, covmat=cov_mcmc)
        if multiple_processes:
            all_collections = mpi_comm.gather(collection)
        else:
            all_collections = [collection]
        # Chains in process of rank 0 now!
        # Compute mean and cov, and broadcast
        if is_main_process:
            for i, colect in enumerate(all_collections):
                colect.detemper()
                # Skip 1/3 of the chain
                new_collection = colect[int(np.floor(len(colect) / 3)):]
                if i == 0:
                    single_collection = new_collection
                else:
                    single_collection.append(new_collection)
            mean_new, cov_new = single_collection.mean(), single_collection.cov()
            # Only main process caches this one, to save memory
            self._last_collection = single_collection
        # Broadcast results
        if multiple_processes:
            mean_new, cov_new = mpi_comm.bcast(
                (mean_new, cov_new) if is_main_process else None)
        return mean_new, cov_new

    def _sample_mcmc(self, gpr, covmat=None):
        from cobaya.model import get_model
        from cobaya.sampler import get_sampler
        from cobaya.log import LoggedError
        # Update Cobaya's input: mcmc's proposal covmat and log-likelihood
        self.cobaya_input = cobaya_generate_gp_model_input(gpr, self.bounds, self.paramnames)
        # Supress Cobaya's output
        # (set to True for debug output, or comment out for normal output)
        self.cobaya_input["debug"] = 50
        # Create model and sampler
        model = get_model(self.cobaya_input)
        if covmat is not None and is_valid_covmat(covmat):
            cov = covmat
        else:
            cov = self.cov
        sampler_info = mcmc_info_from_run(model, gpr, cov=cov)
        # TODO: restore temperature
        # sampler_info["mcmc"]["temperature"] = self.temperature
        high_prec_threshold = (self.values[-1] < 1) if len(self.values) else False
        # Relax stopping criterion if not yet well converged
        sampler_info["mcmc"].update({
            "Rminus1_stop": (0.01 if high_prec_threshold else 0.2),
            "Rminus1_cl_stop": (0.2 if high_prec_threshold else 0.5)})
        mcmc_sampler = get_sampler(sampler_info, model)
        try:
            mcmc_sampler.run()
            success = True
        except LoggedError:
            success = False
        if multiple_processes:
            success = all(mpi_comm.allgather(success))
        if not success:
            raise ConvergenceCheckError
        updated_info = model.info()
        updated_info["sampler"] = {"mcmc": mcmc_sampler.info()}
        return updated_info, mcmc_sampler.products()["sample"]

    def criterion_value(self, gp, gp_2=None):
        try:
            mean_new, cov_new = self._get_new_mean_and_cov(gp)
        except ConvergenceCheckError as excpt:
            self.values.append(np.nan)
            self.n_posterior_evals.append(gp.n_total)
            self.n_accepted_evals.append(gp.n)
            raise ConvergenceCheckError(f"Computation error in KL: {excpt}")
        if gp_2 is not None:
            # TODO: Nothing yet to do with gp2
            pass
        if self.mean is None or self.cov is None:
            # Nothing to compare to! But save mean, cov for next call
            self.mean, self.cov = mean_new, cov_new
            self.values.append(np.nan)
            self.n_posterior_evals.append(gp.n_total)
            self.n_accepted_evals.append(gp.n)
            raise ConvergenceCheckError("No previous call: cannot compute criterion.")
        else:
            mean_old, cov_old = np.copy(self.mean), np.copy(self.cov)
        # Compute the KL divergence (gaussian approx) with the previous iteration
        try:
            kl = kl_norm(mean_new, cov_new, mean_old, cov_old)
        except Exception as excpt:
            kl = np.nan
            raise ConvergenceCheckError(f"Computation error in KL: {excpt}")
        finally:  # whether failed or not
            self.mean = mean_new
            self.cov = cov_new
            self.values.append(kl)
            self.n_posterior_evals.append(gp.n_total)
            self.n_accepted_evals.append(gp.n)
        return kl

    def is_converged(self, gp, gp_2=None, new_X=None, new_y=None, pred_y=None):
        self.criterion_value(gp, gp_2)
        try:
            if np.all(np.array(self.values[-self.limit_times:]) < self.limit):
                return True
        except IndexError:
            pass
        return False

    # Safe copying and pickling
    def __getstate__(self):
        return deepcopy(self).__dict__

    def __deepcopy__(self, memo=None):
        # Remove non-picklable gp model likelihood
        if self.cobaya_input and "likelihood" in self.cobaya_input:
            like = list(self.cobaya_input["likelihood"])[0]
            self.cobaya_input["likelihood"][like]["external"] = True
            if self._last_info and "likelihood" in self._last_info:
                self._last_info["likelihood"][like]["external"] = True
        new = (lambda cls: cls.__new__(cls))(self.__class__)
        new.__dict__ = {k: deepcopy(v) for k, v in self.__dict__.items() if k != "log"}
        return new


class CorrectCounter(ConvergenceCriterion):
    r"""
    This convergence criterion determines convergence by requiring that the GP's
    predictions of the posterior values in the last :math:`n` steps are correct up to a
    certain threshold. This condition is fulfilled if

    .. math::

        |f(x)-\overline{f}_{\mathrm{GP}}(x)| < (f_{\mathrm{max}}(x) - f(x)) \cdot r + a

    where the parameters :math:`r` and :math:`a` are the relative and absolute
    tolerances controlled by the `reltol` and `abstol` parameters.
    We set the "value" of the criterion to be the maximum difference of the GP
    prediction and the true posterior in the last batch of accepted evaluations.
    Furthermore this class contains an internal list `thres` which contains the
    threshold values corresponding to this difference.

    Parameters
    ----------
    prior : model prior instance
        The prior of the model used. This is not needed in this specific
        convergence criterion so you may pass anything here.

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

    def __init__(self, prior, params):
        d = prior.d()
        self.ncorrect = params.get("n_correct", max(4, np.ceil(0.5*d)))
        reltol = params.get("reltol", 0.01)
        if isinstance(reltol, str):
            try:
                assert (reltol[-1] == "l" or reltol[-1] == "s" or reltol[-1] == "r")
                if reltol[-1] == "l":
                    reltol = float(reltol[:-1]) * nstd_of_1d_nstd(1, d)
                elif reltol[-1] == "s":
                    reltol = float(reltol[:-1]) * nstd_of_1d_nstd(1, d)**2.
                elif reltol[-1] == "r":
                    reltol = float(reltol[:-1]) * np.sqrt(nstd_of_1d_nstd(1, d))
            except:
                raise ValueError("The 'reltol' parameter can either be a number " + \
                    f"or a string with a number followed by 'l' or 's'. Got {reltol}")
        self.reltol = reltol
        abstol = params.get("abstol", "0.01s")
        if isinstance(abstol, str):
            try:
                assert (abstol[-1] == "l" or abstol[-1] == "s" or reltol[-1] == "r")
                if abstol[-1] == "l":
                    abstol = float(abstol[:-1]) * nstd_of_1d_nstd(1, d)
                elif abstol[-1] == "s":
                    abstol = float(abstol[:-1]) * nstd_of_1d_nstd(1, d)**2.
                elif abstol[-1] == "r":
                    abstol = float(abstol[:-1]) * np.sqrt(nstd_of_1d_nstd(1, d))
            except:
                raise ValueError("The 'abstol' parameter can either be a number " + \
                    f"or a string with a number followed by 'l' or 's'. Got {abstol}")
        self.abstol = abstol
        self.verbose = params.get("verbose", 0)
        self.values = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []
        self.thres = []
        self.n_pred = 0

    def is_converged(self, gp, gp_2=None, new_X=None, new_y=None, pred_y=None):
        self.criterion_value(gp, new_X=new_X, new_y=new_y, pred_y=pred_y)
        return self.n_pred > self.ncorrect

    def criterion_value(self, gp, gp_2=None, new_X=None, new_y=None, pred_y=None):
        n_new = len(new_y)
        assert(n_new == len(pred_y))
        max_val = 0
        max_diff = 0
        max_thres = 0
        for yn, yl in zip(new_y, pred_y):
            # Remove warning case that does not trigger any condition below
            if yn == -np.inf:
                continue
            # rel_difference = np.abs((yl - gp.y_max) / (yn - gp.y_max) - 1.)
            diff = np.abs(yl - yn)
            thres = np.abs(yn - gp.y_max) * self.reltol + self.abstol
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
        self.n_posterior_evals.append(gp.n_total)
        self.n_accepted_evals.append(gp.n)
        return max_val if n_new > 0 else self.values[-1]
