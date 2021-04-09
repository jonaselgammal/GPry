"""
This module contains several classes and methods for calculating different
convergence criterions which can be used to determine if the BQ algorithm has
converged.

**This part of the code is still not complete as further studies are needed to
find a suitable convergence criterion for our purposes...**
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.linalg import det
from numpy import trace as tr
from scipy.stats import multivariate_normal, entropy
from scipy.special import logsumexp
import warnings
from random import choice
import sys


class Convergence_criterion(metaclass=ABCMeta):
    """ Base class for all convergence criteria (CCs). A CC quantifies the
    convergence of the GP surrogate model. If this value goes below a certain,
    user-set value we consider the GP to have converged to the true posterior
    distribution.

    Currently several CCs are supported which should be versatile enough for
    most tasks. If however one wants to specify a custom CC
    it should be a class which inherits from this abstract class.
    This class needs to be of the format::

        from Acquisition_functions import Acquisition_function
        Class custom_acq_func(Acquisition_Function):
            def __init__(self, prior, params):
                # prior should be a prior object and contain the prior for all
                # parameters
                # params is to be passed as a dictionary. The init should
                # then set the parameters which are needed later accordingly.
                # as a minimal requirement this method should set a number
                # at which the algorithm is considered to have converged.
                # Furthermore this method should initialize empty lists in
                # which we can write the values of the convergence criterion
                # as well as the number of posterior evaluations. This allows
                # for easy tracking/plotting of the convergence.
                self.values = []
                self.n_posterior_evals = []
                self.limit = ... # stores the limit for convergence

            def is_converged(self, gp):
                # Basically a wrapper for the 'criterion_value' method which
                # returns True if the convergence criterion is met and False
                # otherwise.

            def criterion_value():
                # Returns the value of the convergence criterion. Should also
                # append the current value and the number of posterior
                # evaluations to the corresponding variables.
    """

    def get_n_evals_from_gp(self, gp):
        """Method which returns the number of posterior evaluations from the
        gp."""
        if gp.account_for_inf is None:
            n_evals = len(gp.y_train)
        else:
            n_evals = len(gp.account_for_inf.y_train)
        return n_evals

    def get_history(self):
        """Returns the two lists containing the values of the convergence
        criterion at each step as well as the number of posterior evaluations.
        """
        return self.values, self.n_posterior_evals

    @abstractmethod
    def __init__(self, prior, params):
        """sets all relevant initial parameters from the 'params' dict"""

    @abstractmethod
    def is_converged(self, gp, gp_2=None):
        """Returns False if the algorithm hasn't converged and
        True if it has. If gp_2 is None the last GP is taken from the
        model instance."""

    @abstractmethod
    def criterion_value(self, gp, gp_2=None):
        """Returns the value of the convergence criterion for the current
        gp. If gp_2 is None the last GP is taken from the model instance."""


class KL_from_draw(Convergence_criterion):
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

    def is_converged(self, gp, gp_2=None):
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
                self.n_posterior_evals.append(self.get_n_evals_from_gp(gp))
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
            self.n_posterior_evals.append(self.get_n_evals_from_gp(gp))

            return kl


class KL_from_draw_approx(Convergence_criterion):
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

        self.mean = None
        self.cov = None

        self.values = []
        self.n_posterior_evals = []

    def is_converged(self, gp, gp_2=None):
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
                self.n_posterior_evals.append(self.get_n_evals_from_gp(gp))
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
                self.n_posterior_evals.append(self.get_n_evals_from_gp(gp))
                return np.nan
            # Compute the KL divergence (gaussian approx) with the previous iteration
            try:
                cov_old_inv = np.linalg.inv(cov_old)
                kl = 0.5 * (np.log(det(cov_old)) - np.log(det(cov_new))
                            - self.prior.d() + tr(cov_old_inv @ cov_new)
                            + (mean_old - mean_new).T @ cov_old_inv
                            @ (mean_old - mean_new))
                self.values.append(kl)
                self.n_posterior_evals.append(self.get_n_evals_from_gp(gp))
                return kl
            except Exception as e:
                print("KL divergence couldn't be calculated.")
                print(e)
                self.values.append(np.nan)
                self.n_posterior_evals.append(self.get_n_evals_from_gp(gp))
                return np.nan


class KL_from_training(Convergence_criterion):
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

    def is_converged(self, gp, gp_2=None):
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
                self.n_posterior_evals.append(self.get_n_evals_from_gp(gp))
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
                # Invert cov_2 since it will be needed to calculate the
                # KL-divergence
                cov_2_inv = np.linalg.inv(cov_2)

            except Exception as e:
                print("The mean or cov of surrogate_model_2 couldn't be "
                      "calculated.")
                print(e)
                self.values.append(np.nan)
                self.n_posterior_evals.append(self.get_n_evals_from_gp(gp))
                return np.nan

            # Actually calculate the KL divergence
            try:
                kl = 0.5 * (np.log(det(cov_2)) - np.log(det(cov_1))
                            - X_train.shape[-1] + tr(cov_2_inv@cov_1)
                            + (mean_2-mean_1).T @ cov_2_inv
                            @ (mean_2-mean_1))
                self.values.append(kl)
                self.n_posterior_evals.append(self.get_n_evals_from_gp(gp))
                return kl

            except Exception as e:
                print("KL divergence couldn't be calculated.")
                print(e)
                self.values.append(np.nan)
                self.n_posterior_evals.append(self.get_n_evals_from_gp(gp))
                return np.nan


class KL_from_MC_training(Convergence_criterion):
    """
    This class is supposed to use a short MCMC chain to get independent samples
    from the posterior distribution. These can then be used to get the KL
    divergence between samples.
    """

    def __init__(self, prior, params):
        from cobaya.model import get_model
        from cobaya.sampler import get_sampler
        global get_model, get_sampler
        self.prior = prior
        self.cov = None
        self.mean = None
        self.values = []
        self.limit = params.get("limit", 1e-2)
        self.n_posterior_evals = []
        # Number of MCMC chains to generate samples
        self.n_draws_per_dimsquared = params.get("n_draws_per_dimsquared", 10)
        # Number of jumps necessary to assume decorrelation
        self.n_steps_per_dim = params.get("n_steps_per_dim", 5)
        # Number of prior draws for the initial sample
        self.n_initial = params.get("n_initial", 500)
        # Prepare Cobaya's input
        bounds = self.prior.bounds(confidence_for_unbounded=0.99995)
        params_info = {"x_%d" % i: {"prior": {"min": bounds[i, 0], "max": bounds[i, 1]}}
                       for i in range(self.prior.d())}
        self.cobaya_input = {"params": params_info}

    @property
    def n_draws(self):
        return self.n_draws_per_dimsquared * self.prior.d()**2

    @property
    def n_steps(self):
        return self.n_steps_per_dim * self.prior.d()

    def is_converged(self, gp, gp_2=None):
        try:
            if np.all(np.array(self.values[-2:]) < self.limit):
                return True
            else:
                return False
        except Exception as e:
            raise(e)
            return False

    def from_prior(self, gp):
        """
        Get mean and covmat from a prior sample.
        """
        X_values = self.prior.sample(self.n_initial)
        y_values = gp.predict(X_values)
        # Right now this doesn't check whether the value of the covariance
        # matrix is numerically stable i.e. whether it's converged. This
        # shouldn't matter too much though since it's only run once at the
        # beginning and then the MCMC should take care of the rest.
        while True:  # Maybe need something better here so it can't get stuck
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

    def criterion_value(self, gp, gp_2=None):
        """
        This is the important part of the code. Here the calculation of the
        value of the convergence criterion should be performed.
        """
        # Draw samples from the prior until mean and cov stabilize
        # to get an initial guess for the MCMC
        if self.mean is None or self.cov is None:
            mean_prior, cov_prior = self.from_prior(gp)
            if self.mean is None:
                self.mean = mean_prior
            if self.cov is None:
                self.cov = cov_prior
        # Update Cobaya's input: mcmc's propolsal covmat and log-likelihood
        self.cobaya_input["likelihood"] = {
            "gp": {"external":
                   (lambda **kwargs: gp.predict(np.atleast_2d(list(kwargs.values())))[0]),
                   "input_params": list(self.cobaya_input["params"])}}
        self.cobaya_input["debug"] = 50
        model = get_model(self.cobaya_input)
        # TODO: improve this implementation by running longer chains each time,
        # so that max_sampler is n_steps * #points_to_be_acquired
        # (but maybe then points are too decorrelated from training/starting points?)
        n = 0
        X_values = np.empty(shape=(self.n_draws, self.prior.d()))
        y_values = np.empty(shape=(self.n_draws, ))
        while n < self.n_draws:
            this_X = choice(gp.X_train)
            # Starting point of the chain
            # TODO: this is hacky. Update when prior.set_ref created (or sth in mcmc)
            for i, (p, x) in enumerate(zip(self.cobaya_input["params"], this_X)):
                model.prior.ref_pdf[i] = x
            model.prior._ref_is_pointlike = True
            # TODO: hopefully in the future not necessary to re-initialise the sampler
            info_sampler = {
                "mcmc":
                {"covmat": self.cov, "covmat_params": list(self.cobaya_input["params"]),
                 "measure_speeds": False, "max_samples": self.n_steps}}
            mcmc_sampler = get_sampler(info_sampler, model)
            try:
                mcmc_sampler.run()
            except Exception as e:
                # max_tries reached (but may catch and ignore other errors. Not ideal!)
                print(e)
                # sys.exit()
                continue
            last_point = mcmc_sampler.products()["sample"][
                list(self.cobaya_input["params"])].values[-1]
            last_gp_value = -0.5 * mcmc_sampler.products()["sample"]["chi2"].values[-1]
            X_values[n] = last_point
            y_values[n] = last_gp_value
            n += 1
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(*gp.X_train.T, marker="o", label="train")
        # plt.scatter(*X_values.T, marker="^", label="mcmc")
        # plt.legend()
        # plt.savefig("images/MC-generated.png")
        # plt.close()
        # Compute the mean and covmat using appropriate weighting (if possible)
        try:
            mean_old = np.copy(self.mean)
            cov_old = np.copy(self.cov)
            exp_y_new = np.exp(y_values - np.max(y_values))
            self.mean = np.average(X_values, axis=0, weights=exp_y_new)
            self.cov = np.cov(X_values.T, aweights=exp_y_new)
        except:
            print("Mean or cov couldn't be calculated...")
            self.mean = None
            self.cov = None
            self.values.append(np.nan)
            self.n_posterior_evals.append(self.get_n_evals_from_gp(gp))
            return np.nan
        # Compute the KL divergence (gaussian approx) with the previous iteration
        try:
            cov_old_inv = np.linalg.inv(cov_old)
            kl = 0.5 * (np.log(det(cov_old)) - np.log(det(self.cov))
                        - self.prior.d() + tr(cov_old_inv @ self.cov)
                        + (mean_old - self.mean).T @ cov_old_inv
                        @ (mean_old - self.mean))
            self.values.append(kl)
            self.n_posterior_evals.append(self.get_n_evals_from_gp(gp))
            return kl
        except Exception as e:
            print("KL divergence couldn't be calculated.")
            print(e)
            self.values.append(np.nan)
            self.n_posterior_evals.append(self.get_n_evals_from_gp(gp))
            return np.nan
