"""
This module contains several classes and methods for calculating different
convergence criterions which can be used to determine if the BQ algorithm has
converged.
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import multivariate_normal, entropy
from scipy.special import logsumexp
import warnings
from random import choice
import sys
import inspect
from copy import deepcopy
from gpry.tools import kl_norm, cobaya_input_prior, cobaya_input_likelihood, \
    mcmc_info_from_run, is_valid_covmat
from gpry.mpi import mpi_rank, mpi_comm, is_main_process, multiple_processes


class ConvergenceCheckError(Exception):
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
    """ Base class for all convergence criteria (CCs). A CC quantifies the
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
        except:
            raise AttributeError("The convergence criterion does not save it's "
                                 "convergence history.")
        if len(values) == 0 or len(n_posterior_evals) == 0:
            raise ValueError("Make sure to call the convergence criterion "
                             "before getting it's history.")
        return values, n_posterior_evals, n_accepted_evals

    @abstractmethod
    def __init__(self, prior, params):
        """sets all relevant initial parameters from the 'params' dict"""

    @abstractmethod
    def is_converged(self, gp, gp_old=None, new_X=None, new_y=None, pred_y=None):
        """Returns False if the algorithm hasn't converged and
        True if it has. If gp_2 is None the last GP is taken from the
        model instance."""

    @abstractmethod
    def criterion_value(self, gp, gp_2=None):
        """Returns the value of the convergence criterion for the current
        gp. If gp_2 is None the last GP is taken from the model instance."""

    @property
    def is_MPI_aware(self):
        return False


class KL_from_draw(ConvergenceCriterion):
    """
    Class to calculate the KL divergence between two steps of the algorithm
    by drawing n points from the prior and evaluating the KL divergence between
    the last surrogate model and the current one at these points.

    Parameters
    ----------

    params : dict
        Dict with the following keys:

        * ``"prior"``: prior object. Needs to be supplied.
        * ``"limit"``: Number, optional (default=1e-2)
        * ``"n_draws"``: int, optional (default=5000)

    """

    def __init__(self, prior, params):
        # get prior
        self.prior = prior

        self.limit = params.get("limit", 1e-2)
        self.n_draws = params.get("n_draws", 5000)
        self.gp_2 = None

        self.values = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []

    def is_converged(self, gp, gp_2=None, new_X=None, new_y=None, pred_y=None):
        kl = self.criterion_value(gp, gp_2)
        print(kl)
        if kl < self.limit:
            return True
        else:
            return False

    def criterion_value(self, gp, gp_2=None):
        """Calculate the Kullback-Liebler (KL) divergence between different
        steps of the GP acquisition. In contrast to the version where the
        training data is used to calculate the KL divergence this method does
        not assume any form of underlying distribution of the data.

        This function approximates the KL divergence by drawing n samples and
        calculating the cov from this.

        Parameters
        ----------

        gp : SKLearn Gaussian Process Regressor
            The first surrogate model from which the training data is
            retrieved.

        gp_2 : SKLearn Gaussian Process Regressor, optional (default=None)
            The second surrogate model from which the training data is
            retrieved.

        Returns
        -------

        KL_divergence : The value of the KL divergence
        """

        if gp_2 is None:
            gp_2 = self.gp_2
            if not hasattr(gp_2, "X_train"):
                # gp_2 is not a GP so we do not calculate anything...
                self.gp_2 = gp
                self.values.append(np.nan)
                self.n_posterior_evals.append(gp.n_total_evals)
                self.n_accepted_evals.append(gp.n_accepted_evals)
                return np.nan
        else:
            if not hasattr(gp_2, "X_train"):
                raise NameError("gp_2 is either not a GP "
                                "regressor or hasn't been fit to data before.")

        # Check all inputs
        if not hasattr(gp, "X_train"):
            raise NameError("gp is either not a GP regressor "
                            "or hasn't been fit to data before.")

        else:
            X_test = self.prior.sample(self.n_draws)

        # Actual calculation of the KL divergence as sum(p * log(p/q))
        # For this p and q need to be normalized such that they add up to 1.
        # Raise exception for all warnings to catch them.
        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            try:
                # First get p and q by predicting them from the respective
                # models
                logp = gp.predict(X_test)
                logq = gp_2.predict(X_test)

                mask = np.isfinite(logp) & np.isfinite(logq)
                norm = np.max(logp[mask])
                p = np.exp(logp[mask]-norm)
                q = np.exp(logq[mask]-norm)

                kl = entropy(p, qk=q)

                """
                # Need to make sure that stuff adds up to 1 which is a
                # bit tricky... Luckily there's the logsumexp function
                logp = logp - logsumexp(logp)
                logq = logq - logsumexp(logq)

                # Now that stuff is normalized we can calculate the KL
                # divergence. We also save exp(p) for later
                mask = np.isfinite(logp)  & np.isfinite(logq)
                p = np.exp(logp[mask])
                kl = np.sum(p * (logp[mask] - logq[mask]), axis=0)
                """
            except Exception:
                print("KL divergence couldn't be calculated.")
                kl = np.nan

            self.values.append(kl)
            self.n_posterior_evals.append(gp.n_total_evals)
            self.n_accepted_evals.append(gp.n_accepted_evals)

            return kl


class KL_from_draw_approx(ConvergenceCriterion):
    """
    Class to calculate the KL divergence between two steps of the algorithm
    by drawing n points from the prior and evaluating the KL divergence between
    the last surrogate model and the current one at these points.

    Parameters
    ----------

    prior : model prior instance
        The prior of the model used.

    params : dict
        Dict with the following keys:

        * ``"limit"``: Value of KL required for convergence (default=1e-2)
        * ``"limit_times"``: The number of times the limit has to be hit in
          consecutive steps (default=2)
        * ``"n_draws"``: Number of samples for calculating the mean and cov
          (default=5000)
        * ``"method"``: The sampling method (``"simple"``=sample from prior,
          ``"lhs"``=latin hypercube sample)

    """

    def __init__(self, prior, params):
        # get prior
        self.prior = prior

        self.limit = params.get("limit", 1e-2)
        self.n_draws = params.get("n_draws", 5000)
        self.method = params.get("method", "simple").lower()
        self.limit_times = params.get("limit_times", 2)
        self.gp_2 = None

        self.mean = None
        self.cov = None

        self.values = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []

    def is_converged(self, gp, gp_2=None, new_X=None, new_y=None, pred_y=None):
        kl = self.criterion_value(gp, gp_2)
        try:
            if np.all(np.array(self.values[-self.limit_times:]) < self.limit):
                return True
        except IndexError:
            pass
        return False

    def criterion_value(self, gp, gp_2=None):
        """Calculate the Kullback-Liebler (KL) divergence between different
        steps of the GP acquisition. In contrast to the version where the
        training data is used to calculate the KL divergence this method does
        not assume any form of underlying distribution of the data.

        This function approximates the KL divergence by drawing n samples and
        calculating the cov from this.

        Parameters
        ----------

        gp : SKLearn Gaussian Process Regressor
            The first surrogate model from which the training data is
            retrieved.

        gp_2 : SKLearn Gaussian Process Regressor, optional (default=None)
            The second surrogate model from which the training data is
            retrieved.

        Returns
        -------

        KL_divergence : The value of the KL divergence
        """

        if gp_2 is None:
            gp_2 = self.gp_2
            if not hasattr(gp_2, "X_train"):
                # gp_2 is not a GP so we do not calculate anything...
                self.gp_2 = gp
                self.values.append(np.nan)
                self.n_posterior_evals.append(gp.n_total_evals)
                self.n_accepted_evals.append(gp.n_accepted_evals)
                return np.nan
        else:
            if not hasattr(gp_2, "X_train"):
                raise NameError("gp_2 is either not a GP "
                                "regressor or hasn't been fit to data before.")

        # Check all inputs
        if not hasattr(gp, "X_train"):
            raise NameError("gp is either not a GP regressor "
                            "or hasn't been fit to data before.")

        else:
            if self.method == "simple":
                X_test = self.prior.sample(self.n_draws)
            elif self.method == "lhs":
                try:
                    import lhsmdu
                except ImportError:
                    raise ImportError(
                        "You need to install 'lhsmdu' with pip to use this criterion")
                # We need to chunk LHS (worse evenly-spacing guarantee) due to memory use
                nchunk = min(100, self.n_draws)
                X_test = None
                for i in range(int(np.ceil(self.n_draws / nchunk))):
                    lhsmdu.setRandomSeed(None)
                    this_X_test = lhsmdu.sample(self.prior.d(), nchunk)
                    if X_test is None:
                        X_test = this_X_test
                    else:
                        X_test = np.concatenate([X_test, this_X_test])
                X_test = X_test[:self.n_draws]
        # Actual calculation of the KL divergence as sum(p * log(p/q))
        # For this p and q need to be normalized such that they add up to 1.
        # Raise exception for all warnings to catch them.
        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            try:

                y_values_1 = gp.predict(X_test)
                y_values_2 = gp_2.predict(X_test)
                exp_y_old = np.exp(y_values_2 - np.max(y_values_2))
                exp_y_new = np.exp(y_values_1 - np.max(y_values_1))
                mean_old = np.average(X_test, axis=0, weights=exp_y_old)
                cov_old = np.cov(X_test.T, aweights=exp_y_old)
                mean_new = np.average(X_test, axis=0, weights=exp_y_new)
                cov_new = np.cov(X_test.T, aweights=exp_y_new)

                self.mean = mean_new
                self.cov = cov_new

            except:
                print("Mean or cov couldn't be calculated...")
                self.mean = None
                self.cov = None
                self.values.append(np.nan)
                self.n_posterior_evals.append(gp.n_total_evals)
                self.n_accepted_evals.append(gp.n_accepted_evals)
                return np.nan
            # Compute the KL divergence (gaussian approx) with the previous iteration
            try:
                kl = kl_norm(mean_new, cov_new, mean_old, cov_old)
                self.values.append(kl)
                self.n_posterior_evals.append(gp.n_total_evals)
                self.n_accepted_evals.append(gp.n_accepted_evals)
                return kl
            except Exception as e:
                print("KL divergence couldn't be calculated.")
                print(e)
                self.values.append(np.nan)
                self.n_posterior_evals.append(gp.n_total_evals)
                self.n_accepted_evals.append(gp.n_accepted_evals)
                return np.nan


class KL_from_training(ConvergenceCriterion):
    """
    Class to calculate the KL divergence between two different GPs assuming
    that the GP follows some (unnormalized) multivariate gaussian distribution.

    params : dict
        ...

    Attributes
    ----------

    mean : The weighted mean of the training data (weighted with the value of
        the posterior distribution).

    cov : The covariance matrix of the training data along its last axis.

    """

    def __init__(self, prior, params):
        self.prior = prior
        self.cov = None
        self.mean = None
        self.limit = params.get("limit", 1e-2)
        self.n_draws = params.get("n_draws", 5000)
        self.gp_2 = None

        self.values = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []

    def is_converged(self, gp, gp_2=None, new_X=None, new_y=None, pred_y=None):
        kl = self.criterion_value(gp, gp_2)
        print(kl)
        if kl < self.limit:
            return True
        else:
            return False

    def criterion_value(self, gp, gp_2=None):
        """Calculate the Kullback-Liebler (KL) divergence between different
        steps of the GP acquisition. Here the KL divergence is used as a
        convergence criterion for the GP. The KL-Divergence assumes a
        multivariate normal distribution as underlying likelihood. Thus it may
        perform strangely when applied to some sort of weird likelihood.

        This function approximates the KL divergence by using the training
        samples weighted by their Likelihood values to get an estimate for the
        mean and covariance matrix along each dimension. The training data is
        taken internally from the surrogate models

        Parameters
        ----------

        gp : SKLearn Gaussian Process Regressor
            The first surrogate model from which the training data is
            retrieved.

        gp_2 : SKLearn Gaussian Process Regressor, optional (default=None)
            The second surrogate model from which the training data is
            retrieved.

        Both models need to have been fit to data before.

        Returns
        -------

        KL_divergence : The value of the KL divergence between the two models.
        If the KL divergence cannot be determined ``None`` is returned.
        """
        # Check all inputs
        if not hasattr(gp, "X_train"):
            raise NameError("GP is either not a GP regressor "
                            "or hasn't been fit to data before.")

        if gp_2 is None:
            # raise warning if mean and cov do not exist/were not calculated
            # successfully before
            if self.mean is None or self.cov is None:
                warnings.warn("The mean and cov have not been fit "
                              "successfully before. Therefore the KL "
                              "divergence cannot be calculated properly...")
            mean_2 = np.copy(self.mean)
            cov_2 = np.copy(self.cov)

        else:
            if not hasattr(gp_2, "X_train"):
                raise NameError("surrogate_model_2 is either not a GP "
                                "regressor or hasn't been fit to data before.")

        # Raise warnings as exceptions from here to control their behaviour
        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            # Start by calculating the mean and cov for surrogate model 1
            try:
                # Calculate mean and cov for PDF from model training
                # data. For weighting the mean and cov we take the model
                # prediction as the data might have some numerical noise.
                # Furthermore we need to exponentiate the target values
                # such that they follow an (unnormalized) PDF. We choose the
                # normalization such that max(exp(y_train))=1.
                X_train = np.copy(gp.X_train)
                y_train = gp.predict(X_train)
                y_train = np.exp(y_train - np.max(y_train))

                # Actually calculate the mean and cov
                mean_1 = np.average(X_train, axis=0, weights=y_train)
                cov_1 = np.cov(X_train.T, aweights=y_train)

                # Save the mean and covariance in the model instance to be able
                # to access it later
                self.mean = mean_1
                self.cov = cov_1

            except Exception as e:
                print("The mean or cov of surrogate_model_1 couldn't be "
                      "calculated.")
                print(e)
                self.mean = None
                self.cov = None
                self.values.append(np.nan)
                self.n_posterior_evals.append(gp.n_total_evals)
                self.n_accepted_evals.append(gp.n_accepted_evals)
                return np.nan

            # Calculate the mean and cov of surrogate model 2
            try:
                if gp_2 is not None:
                    # Same stuff as above
                    X_train = np.copy(gp_2.X_train)
                    y_train = gp_2.predict(X_train)
                    y_train = np.exp(y_train - np.max(y_train))
                    mean_2 = np.average(X_train, axis=0, weights=y_train)
                    cov_2 = np.cov(X_train.T, aweights=y_train)

            except Exception as e:
                print("The mean or cov of surrogate_model_2 couldn't be "
                      "calculated.")
                print(e)
                self.values.append(np.nan)
                self.n_posterior_evals.append(gp.n_total_evals)
                self.n_accepted_evals.append(gp.n_accepted_evals)
                return np.nan

            # Actually calculate the KL divergence
            try:
                kl = kl_norm(mean_1, cov_1, mean_2, cov_2)
                self.values.append(kl)
                self.n_posterior_evals.append(gp.n_total_evals)
                self.n_accepted_evals.append(gp.n_accepted_evals)
                return kl

            except Exception as e:
                print("KL divergence couldn't be calculated.")
                print(e)
                self.values.append(np.nan)
                self.n_posterior_evals.append(gp.n_total_evals)
                self.n_accepted_evals.append(gp.n_accepted_evals)
                return np.nan


class ConvergenceCriterionGaussianApprox(ConvergenceCriterion):

    def _mean_and_cov_from_prior(self, gp):
        """
        Get mean and covmat from a prior sample.
        """
        X_values = self.prior.sample(self.n_initial)
        try:
            y_values = gp.predict(X_values)
        except:
            raise ConvergenceCheckError("Argument is either not a GP "
                                        "regressor or hasn't been fit to data before.")
        # Right now this doesn't check whether the value of the covariance
        # matrix is numerically stable i.e. whether it's converged. This
        # shouldn't matter too much though since it's only run once at the
        # beginning and then the MCMC should take care of the rest.
        # TODO: Maybe need something better here so it can't get stuck
        while True:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    y_values = np.exp(y_values - np.max(y_values))
                    mean = np.average(X_values, axis=0, weights=y_values)
                    cov = np.cov(X_values.T, aweights=y_values)
                    return mean, cov
                except:
                    # Draw more points and try again
                    X_values = np.append(X_values,
                                         self.prior.sample(self.n_initial),
                                         axis=0)

    def _mean_and_cov(self, X, gp):
        """
        Mean and covariance from sample given GP.

        Raises ConvergenceCheckError if singular covariance matrix.
        """
        y_values = gp.predict(X)
        exp_y = np.exp(y_values - np.max(y_values))
        i_nonzeros = np.where(np.isclose(exp_y, 0))[0]
        mean = np.average(X, axis=0, weights=exp_y)
        cov = np.cov(X.T, aweights=exp_y)
        try:
            if np.linalg.matrix_rank(cov, 1e-6) < len(cov):
                raise np.linalg.LinAlgError
        # np.linalg.matrix_rank can raise this one too!
        except np.linalg.LinAlgError:
            raise ConvergenceCheckError("Got singular covariance matrix")
        return mean, cov


class KL_from_MC_training(ConvergenceCriterionGaussianApprox):
    """
    This class is supposed to use a short MCMC chain to get independent samples
    from the posterior distribution. These can then be used to get the KL
    divergence between samples.
    """

    def __init__(self, prior, params):
        from cobaya.model import get_model
        from cobaya.sampler import get_sampler
        from cobaya.collection import SampleCollection
        from cobaya.log import LoggedError
        global get_model, get_sampler, SampleCollection, LoggedError
        self.prior = prior
        self.mean = None
        self.cov = None
        self.values = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []
        self.limit = params.get("limit", 1e-2)
        # Number of MCMC chains to generate samples
        if params.get("n_draws") and params.get("n_draws_per_dimsquared"):
            raise ValueError(
                "Pass either 'n_draws' or 'n_draws_per_dimsquared', not both")
        if params.get("n_draws"):
            self._n_draws = int(params.get("n_draws"))
        else:
            self.n_draws_per_dimsquared = params.get("n_draws_per_dimsquared", 10)
        # Number of jumps necessary to assume decorrelation
        self.n_steps_per_dim = params.get("n_steps_per_dim", 5)
        # Number of prior draws for the initial sample
        self.n_initial = params.get("n_initial", 500)
        # MCMC temperature
        self.temperature = params.get("temperature", 4)
        # Prepare Cobaya's input
        bounds = self.prior.bounds(confidence_for_unbounded=0.99995)
        params_info = {"x_%d" % i: {"prior": {"min": bounds[i, 0], "max": bounds[i, 1]}}
                       for i in range(self.prior.d())}
        self.cobaya_input = {"params": params_info}

    @property
    def n_draws(self):
        if hasattr(self, "_n_draws"):
            return self._n_draws
        else:
            return self.n_draws_per_dimsquared * self.prior.d()**2

    @property
    def n_steps(self):
        return self.n_steps_per_dim * self.prior.d()

    def is_converged(self, gp, gp_2=None, new_X=None, new_y=None, pred_y=None):
        try:
            if np.all(np.array(self.values[-2:]) < self.limit):
                return True
            else:
                return False
        except Exception as e:
            raise(e)
            return False

    def criterion_value(self, gp, gp_2=None):
        """
        Compute the value of the convergence criterion.

        Raises :class:`convergence.ConvergenceCheckError` if it could not be computed.
        """
        mean_new, cov_new = None, None
        kl = np.nan
        try:
            # Get OLD mean and cov
            if gp_2 is None:
                # Use last ones. Notice that `np.copy(None) != None`!)
                mean_old = np.copy(self.mean) if self.mean is not None else None
                cov_old = np.copy(self.cov) if self.cov is not None else None
            else:
                try:
                    mean_old, cov_old = self._mean_and_cov_from_mcmc(gp)
                except ConvergenceCheckError as excpt:
                    raise ConvergenceCheckError(
                        f"Error when computing mean and cov *for gp_2*: {excpt}")
            # Get NEW mean and cov
            mean_new, cov_new = self._mean_and_cov_from_mcmc(gp)
            # Check AFTER computing new ones: we want to save them for next iter anyway
            if mean_old is None or cov_old is None:
                raise ConvergenceCheckError("No mean or cov available from last step.")
            # Compute the KL divergence (gaussian approx) with the previous iteration
            try:
                kl = kl_norm(mean_new, cov_new, mean_old, cov_old)
            except Exception as excpt:
                raise ConvergenceCheckError(f"Computation error in KL: {excpt}")
        except ConvergenceCheckError as excpt:
            raise ConvergenceCheckError(
                f"Could not compute criterion value for this iteration: {excpt}")
        finally:  # whether failed or not
            self.mean = mean_new
            self.cov = cov_new
            self.values.append(kl)
            self.n_posterior_evals.append(gp.n_total_evals)
            self.n_accepted_evals.append(gp.n_accepted_evals)
        return kl

    def _mean_and_cov_from_mcmc(self, gp):
        X_values, y_values = self._draw_from_mcmc(gp)
        try:
            # Compute the mean and covmat using appropriate weighting (if possible)
            exp_y_new = np.exp(y_values - np.max(y_values))
            mean = np.average(X_values, axis=0, weights=exp_y_new)
            cov = np.cov(X_values.T, aweights=exp_y_new)
            return (mean, cov)
        except Exception as excpt:
            raise ConvergenceCheckError(
                f"Could not compute mean and cov from MCMC: {excpt}")

    def _draw_from_mcmc(self, gp):
        # Update Cobaya's input: mcmc's proposal covmat and log-likelihood
        self.cobaya_input["likelihood"] = {
            "gp": {"external":
                   (lambda **kwargs: gp.predict(np.atleast_2d(list(kwargs.values())), do_check_array=False)[0]),
                   "input_params": list(self.cobaya_input["params"])}}
        # Supress Cobaya's output
        # (set to True for debug output, or comment out for normal output)
        self.cobaya_input["debug"] = 50
        model = get_model(self.cobaya_input)
        # If no covariance computed in last step, try to get one from a prior sample
        # TODO: expensive if intermediate step fails. Maybe save last working one?
        covmat = self.cov
        if covmat is None:
            _, covmat = self._mean_and_cov_from_prior(gp)
        sampler_input = {"mcmc": {
            "covmat": covmat, "covmat_params": list(self.cobaya_input["params"]),
            "temperature": self.temperature,
            # Faster: no need to measure speeds, check convergence or learn proposal
            "measure_speeds": False, "learn_every": "1000000d"}}
        # TODO: try to make initialisation even a bit faster
        # (e.g. set an initial point manually)
        mcmc_sampler = get_sampler(sampler_input, model)
        # Draw >1 point per chain to avoid Cobaya initialisation overhead
        # At most, draw #draws/#training per chain, unless there are loads of training
        # At least, in case there are too many training points, #draws=dim
        draws_per_chain = max(int(self.n_draws / len(gp.X_train)), self.prior.d())
        n = 0
        X_values = np.full(fill_value=np.nan, shape=(self.n_draws, self.prior.d()))
        y_values = np.full(fill_value=np.nan, shape=(self.n_draws, ))
        failed = []
        while n < self.n_draws:
            # Pick a point from the training set and set it as initial point of the chain
            # (a little hacky, but there is no API at the moment to do this)
            this_i = choice(range(len(gp.X_train)))
            this_X = np.copy(gp.X_train[this_i])
            logpost = model.logposterior(this_X, temperature=self.temperature)
            mcmc_sampler.current_point.add(this_X, logpost)
            # reset the number of samples and run
            mcmc_sampler.collection = SampleCollection(
                model, mcmc_sampler.output, temperature=self.temperature)
            this_n_draws = min(self.n_draws - n, draws_per_chain)
            mcmc_sampler.max_samples = (1 + this_n_draws) * self.n_steps
            try:
                mcmc_sampler.run()
            except LoggedError:
                # max_tries reached: chain stuck!
                # for now, fail (return NaN) if some fraction of the training samples
                # failed (with repetition).
                # It would not scale too well with iterations, but highly unlikely to
                # produce this error after having run for a bit.
                failed += [this_i]
                fraction = 0.5
                if len(failed) >= fraction * len(gp.X_train):
                    raise ConvergenceCheckError("Could not draw enough points from MCMC")
                else:
                    continue
            points = mcmc_sampler.products()["sample"][
                list(self.cobaya_input["params"])].values[::-self.n_steps]
            gp_values = -0.5 * \
                mcmc_sampler.products()["sample"]["chi2"].values[::-self.n_steps]
            # There is sometimes one more point, hence the [:this_n_draws] below
            X_values[n:n + this_n_draws] = points[:this_n_draws]
            y_values[n:n + this_n_draws] = gp_values[:this_n_draws]
            n += this_n_draws
        # Test: plots points (2d only)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(*gp.X_train.T, marker="o", label="train")
        # plt.scatter(*X_values.T, marker="^", label="mcmc")
        # plt.legend()
        # plt.show()
        # plt.savefig("images/MC-generated.png")
        # plt.close()
        return X_values, y_values


class KL_from_draw_approx_alt(ConvergenceCriterionGaussianApprox):
    """
    Class to calculate the KL divergence between two steps of the algorithm
    by computing the KL divergence between estimations of the mean and
    covariance matrix of the previous and current surrogate models. The mean
    and covariance are computed using either a sample from the prior or a
    sample from a set multiple of the last mean and covariance.

    Parameters
    ----------

    prior : model prior instance
        The prior of the model used.

    params : dict
        Dict with the following keys:

        * ``"limit"``: Value of KL required for convergence (default=1e-2)
        * ``"n_draws"``: Number of samples for calculating the mean and cov (default=5000)
        * ``"cov_multiple"``: The number by which the covariance is multiplied when sampling from it (default=1)

    """

    def __init__(self, prior, params):
        # get prior
        self.prior = prior

        self.limit = params.get("limit", 1e-2)
        self.n_draws = params.get("n_draws", 5000)
        self.cov_multiple = params.get("cov_multiple", 1)

        self.mean = None
        self.cov = None

        self.values = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []

    def is_converged(self, gp, gp_2=None, new_X=None, new_y=None, pred_y=None):
        kl = self.criterion_value(gp, gp_2)
        print(kl)
        if kl < self.limit:
            return True
        else:
            return False

    def criterion_value(self, gp, gp_2=None):
        """Calculate the Kullback-Liebler (KL) divergence between different
        steps of the GP acquisition. In contrast to the version where the
        training data is used to calculate the KL divergence this method does
        not assume any form of underlying distribution of the data.

        Parameters
        ----------

        gp : SKLearn Gaussian Process Regressor
            The first surrogate model from which the training data is
            retrieved.

        gp_2 : SKLearn Gaussian Process Regressor, optional (default=None)
            The second surrogate model from which the training data is
            retrieved. If not specified, the previous mean and covmat are used.

        Returns
        -------

        KL_divergence : The value of the KL divergence
        """
        # First, get the X sample (from prior if mean and cov not prev set)
        if self.mean is None or self.cov is None:
            X_test = self.prior.sample(self.n_draws)
        else:
            try:
                X_test = self._sparse_sample_from_gaussian()
            except ConvergenceCheckError as excpt:
                self.values.append(np.nan)
                self.n_posterior_evals.append(gp.n_total_evals)
                self.n_accepted_evals.append(gp.n_accepted_evals)
                raise ConvergenceCheckError(f"Computation error in KL: {excpt}")
        try:
            mean_new, cov_new = self._mean_and_cov(X_test, gp)
        except ConvergenceCheckError as excpt:
            self.values.append(np.nan)
            self.n_posterior_evals.append(gp.n_total_evals)
            self.n_accepted_evals.append(gp.n_accepted_evals)
            raise ConvergenceCheckError(f"Computation error in KL: {excpt}")
        if gp_2 is None:
            if self.mean is None or self.cov is None:
                # Nothing to compare to! But save mean, cov for next call
                self.mean, self.cov = mean_new, cov_new
                self.values.append(np.nan)
                self.n_posterior_evals.append(gp.n_total_evals)
                self.n_accepted_evals.append(gp.n_accepted_evals)
                raise ConvergenceCheckError("No previous call: needs gp_2 to compare.")
            else:
                mean_old, cov_old = np.copy(self.mean), np.copy(self.cov)
        else:
            mean_old, cov_old = self._mean_and_cov(X_test, gp_2)
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
            self.n_posterior_evals.append(gp.n_total_evals)
            self.n_accepted_evals.append(gp.n_accepted_evals)
        return kl

    def _sparse_sample_from_gaussian(self, mean=None, cov=None, n_draws=None):
        if mean is None:
            mean = self.mean
        if cov is None:
            cov = self.cov
        if n_draws is None:
            n_draws = self.n_draws
        X_test = []  # will be an array
        mult = self.cov_multiple
        while len(X_test) < n_draws:
            this_n_draws = n_draws - len(X_test)
            try:
                this_X_test = np.atleast_2d(multivariate_normal(
                    mean=mean, cov=mult * cov).rvs(this_n_draws))
            except np.linalg.LinAlgError:
                raise ConvergenceCheckError("Covariance is singular.")
            logpriors = [self.prior.logp(X) for X in this_X_test]
            this_X_test = this_X_test[np.isfinite(logpriors)]
            # Shrink contour if not too many points
            if len(this_X_test) < 0.1 * this_n_draws:
                if mult > 1:
                    mult -= 1
                else:  # probably very misestimated
                    mult *= 0.5
            if X_test == []:
                X_test = this_X_test
            else:
                X_test = np.concatenate([X_test, this_X_test])
        return np.array(X_test[:self.n_draws])


class ConvergenceCriterionGaussianMCMC(ConvergenceCriterionGaussianApprox):

    @property
    def is_MPI_aware(self):
        return True

    def __init__(self, prior, params):
        self.values = []
        self.n_posterior_evals = []
        self.prior = prior
        self.mean = None
        self.cov = None
        self.limit = params.get("limit", 1e-2)
        self.limit_times = params.get("limit_times", 2)
        self.values = []
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
        # Number of jumps necessary to assume decorrelation
        self.n_steps_per_dim = params.get("n_steps_per_dim", 5)
        # Number of prior draws for the initial sample
        self.n_initial = params.get("n_initial", 500)
        # MCMC temperature
        self.temperature = params.get("temperature", 2)
        # Max times a sample can be reweighted and reused (we may miss new high regions)
        self.max_reused = params.get("max_reused", 4)
        # Prepare Cobaya's input
        self.cobaya_input = cobaya_input_prior(prior)
        # Save last sample
        self._last_info = {}
        self._last_collection = None

    @property
    def cobaya_param_names(self):
        return list(self.cobaya_input["params"])

    def _get_new_mean_and_cov(self, gp):
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

    def criterion_value(self, gp, gp_2=None):
        try:
            mean_new, cov_new = self._get_new_mean_and_cov(gp)
        except ConvergenceCheckError as excpt:
            self.values.append(np.nan)
            self.n_posterior_evals.append(gp.n_total_evals)
            self.n_accepted_evals.append(gp.n_accepted_evals)
            raise ConvergenceCheckError(f"Computation error in KL: {excpt}")
        if gp_2 is not None:
            # TODO: Nothing yet to do with gp2
            pass
        if self.mean is None or self.cov is None:
            # Nothing to compare to! But save mean, cov for next call
            self.mean, self.cov = mean_new, cov_new
            self.values.append(np.nan)
            self.n_posterior_evals.append(gp.n_total_evals)
            self.n_accepted_evals.append(gp.n_accepted_evals)
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
            self.n_posterior_evals.append(gp.n_total_evals)
            self.n_accepted_evals.append(gp.n_accepted_evals)
        return kl

    def _sample_mcmc(self, gp, covmat=None):
        from cobaya.model import get_model
        from cobaya.sampler import get_sampler
        from cobaya.log import LoggedError
        # Update Cobaya's input: mcmc's proposal covmat and log-likelihood
        self.cobaya_input.update(cobaya_input_likelihood(gp, self.cobaya_param_names))
        # Supress Cobaya's output
        # (set to True for debug output, or comment out for normal output)
        self.cobaya_input["debug"] = 50
        # Create model and sampler
        model = get_model(self.cobaya_input)
        sampler_info = mcmc_info_from_run(model, gp, convergence=self)
        if covmat is not None and is_valid_covmat(covmat):
            # Prefer the one explicitly passed
            sampler_info["mcmc"]["covmat"] = covmat
        sampler_info["mcmc"]["temperature"] = self.temperature
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

class DontConverge(ConvergenceCriterion):
    """
    This convergence criterion is mainly for testing purposes and always
    returns False when ``is_converged`` is called. Use this method together
    with the `max_points` and `max_accepted` keys in the options dict to stop
    the BO loop at a set number of iterations.
    """

    @property
    def is_MPI_aware(self):
        return True

    def __init__(self, prior, params):
        self.values = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []
        self.prior = prior

    def criterion_value(self, gp, gp_2=None):
        self.values.append(np.nan)
        self.n_posterior_evals.append(gp.n_total_evals)
        self.n_accepted_evals.append(gp.n_accepted_evals)
        return np.nan

    def is_converged(self, gp, gp_2=None, new_X=None, new_y=None, pred_y=None):
        self.criterion_value(gp, gp_2)
        return False


class CorrectCounter(ConvergenceCriterion):
    """
    This convergence criterion is the standard one used by GPry. It determines
    convergence by requiring that the GP's predictions of the posterior values
    in the last :math:`n` steps are correct up to a certain percentage.
    We set the "value" of the criterion to be the maximum difference of the GP
    prediction and the true posterior in the last batch of accepted evaluations.

    Parameters
    ----------

    prior : model prior instance
        The prior of the model used. This is not needed in this specific
        convergence criterion so you may pass anything here.

    params : dict
        Dict with the following keys:

        * ``"n_correct"``: Value of KL required for convergence (default=1e-2)
        * ``"threshold"``: Number of samples for calculating the mean and cov (default=5000)
        * ``"verbose"``: Verbosity

    """

    def __init__(self, prior, params):
        self.ncorrect = params.get("n_correct", 5)
        self.reltol = params.get("threshold", 0.01)
        self.abstol = params.get("abstol",0.05)
        self.verbose = params.get("verbose", 0)
        self.values = []
        self.n_posterior_evals = []
        self.n_accepted_evals = []
        self.n_pred = 0

    def is_converged(self, gp, gp_2=None, new_X=None, new_y=None, pred_y=None):
        self.criterion_value(gp, new_X=new_X, new_y=new_y, pred_y=pred_y)
        return self.n_pred > self.ncorrect

    def criterion_value(self, gp, gp_2=None, new_X=None, new_y=None, pred_y=None):
        n_new = len(new_y)
        assert(n_new == len(pred_y))
        max_val = 0
        for yn,yl in zip(new_y, pred_y):
            #rel_difference = np.abs((yl-gp.y_max)/(yn-gp.y_max)-1.)
            diff = np.abs(yl-yn)
            max_val = max(np.max(diff), max_val)
            thresh = np.abs(yn-gp.y_max) * self.reltol + self.abstol
            if diff < thresh:
                self.n_pred += 1
                if self.verbose > 0:
                    print(f"Already {self.n_pred} correctly predicted \n")
            else:
                self.n_pred = 0
                if self.verbose > 0:
                    print("Mispredict...")
        self.values.append(max_val if n_new>0 else self.values[-1])
        self.n_accepted_evals.append(gp.n_accepted_evals)
        self.n_posterior_evals.append(gp.n_total_evals)
        return max_val if n_new>0 else self.values[-1]
