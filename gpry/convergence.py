"""
This module contains several classes and methods for calculating different
convergence criterions which can be used to determine if the BQ algorithm has
converged.

**This part of the code is still not complete as further studies are needed to
find a suitable convergence criterion for our purposes...**
"""

import numpy as np
from numpy.linalg import det
from numpy import trace as tr
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import warnings

class KL_divergence:
    """
    Class to calculate the KL divergence between two different GPs assuming that
    the GP follows some (unnormalized) multivariate gaussian distribution.

    Attributes
    ----------

    mean : The weighted mean of the training data (weighted with the value of
        the posterior distribution).

    cov : The covariance matrix of the training data along its last axis.

    """
    def __init__(self, bounds):
        self.cov = None
        self.mean = None
        self.bounds = bounds

    def kl_from_training(self, surrogate_model_1, surrogate_model_2=None):
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

        surrogate_model_1 : SKLearn Gaussian Process Regressor
            The first surrogate model from which the training data is retrieved.

        surrogate_model_2 : SKLearn Gaussian Process Regressor, optional
            (default=None)
            The second surrogate model from which the training data is
            retrieved.

        Both models need to have been fit to data before.

        Returns
        -------

        KL_divergence : The value of the KL divergence between the two models.
        If the KL divergence cannot be determined ``None`` is returned.
        """
        # Check all inputs
        if not hasattr(surrogate_model_1, "X_train"):
            raise NameError("surrogate_model_1 is either not a GP regressor "\
                "or hasn't been fit to data before.")

        if surrogate_model_2 is None:
            # raise warning if mean and cov do not exist/were not calculated
            # successfully before
            if self.mean is None or self.cov is None:
                warnings.warn("The mean and cov have not been fit "\
                    "successfully before. Therefore the KL divergence cannot "\
                    "be calculated properly...")
            mean_2 = np.copy(self.mean)
            cov_2 = np.copy(self.cov)

        else:
            if not hasattr(surrogate_model_2, "X_train"):
                raise NameError("surrogate_model_2 is either not a GP "\
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
                X_train = np.copy(surrogate_model_1.X_train)
                y_train = surrogate_model_1.predict(X_train)
                y_train = np.exp(y_train - np.max(y_train))

                # Actually calculate the mean and cov
                mean_1 = np.average(X_train, axis=0, weights=y_train)
                cov_1 = np.cov(X_train.T, aweights=y_train)

                # Save the mean and covariance in the model instance to be able
                # to access it later
                self.mean = mean_1
                self.cov = cov_1

            except Exception as e:
                print("The mean or cov of surrogate_model_1 couldn't be "\
                    "calculated.")
                print(e)
                self.mean = None
                self.cov = None
                return np.nan

            # Calculate the mean and cov of surrogate model 2
            try:
                if surrogate_model_2 is not None:
                    # Same stuff as above
                    X_train = np.copy(surrogate_model_2.X_train)
                    y_train = surrogate_model_2.predict(X_train)
                    y_train = np.exp(y_train - np.max(y_train))
                    mean_2 = np.average(X_train, axis=0, weights=y_train)
                    cov_2 = np.cov(X_train.T, aweights=y_train)
                # Invert cov_2 since it will be needed to calculate the
                # KL-divergence
                cov_2_inv = np.linalg.inv(cov_2)

            except Exception as e:
                print("The mean or cov of surrogate_model_2 couldn't be "\
                    "calculated.")
                print(e)
                return np.nan

            # Actually calculate the KL divergence
            try:
                kl = 0.5 * (np.log(det(cov_2)) - np.log(det(cov_1)) \
                    - X_train.shape[-1] + tr(cov_2_inv@cov_1) \
                    + (mean_2-mean_1).T @ cov_2_inv \
                    @ (mean_2-mean_1))
                return kl

            except Exception as e:
                print("KL divergence couldn't be calculated.")
                print(e)
                return np.nan

    def kl_from_draw(self, surrogate_model_1, surrogate_model_2,
            n_draws = 5000, guess_initial=True):
        """Calculate the Kullback-Liebler (KL) divergence between different
        steps of the GP acquisition. In contrast to the version where the
        training data is used to calculate the KL divergence this method does
        not assume any form of underlying distribution of the data.

        This function approximates the KL divergence by drawing n samples and
        calculating the cov from this.

        Parameters
        ----------

        surrogate_model_1 : SKLearn Gaussian Process Regressor
            The first surrogate model from which the training data is retrieved.

        surrogate_model_2 : SKLearn Gaussian Process Regressor
            The second surrogate model from which the training data is
            retrieved.

        bounds : array-like, shape=(n_dims,2)
            Array of bounds of the prior [lower, upper] along each dimension.

        n_draws : int, optional (default=5000)
            The number of samples that are drawn from the GPs

        guess_initial : bool, optional (default=True)
            Whether to draw the random samples from the multivariate gaussian
            distribution that has been fit to data before. For this the KL
            divergence needs to have been called before.

        Returns
        -------

        KL_divergence : The value of the KL divergence
        """

        # Check all inputs
        if not hasattr(surrogate_model_1, "X_train"):
            raise NameError("surrogate_model_1 is either not a GP regressor "\
                "or hasn't been fit to data before.")

        if not hasattr(surrogate_model_2, "X_train"):
            raise NameError("surrogate_model_2 is either not a GP "\
                "regressor or hasn't been fit to data before.")

        # Draw training data either from uniform distribution or initial guess
        if guess_initial:
            if self.mean is None or self.cov is None:
                warnings.warn("mean or cov is None so samples will be drawn "\
                    "uniformly inside bounds.")
                X_train = np.random.uniform(self.bounds[:,0], self.bounds[:,1],
                    (n_draws,len(self.bounds[:,0])))
                weights = None
            else:
                # Drawing from initial guess means drawing samples from the
                # multivariate prpopsal normal dist. Essentially this is just
                # importance sampling.
                try:
                    rv = multivariate_normal(self.mean, self.cov)
                    X_train = rv.rvs(size=n_draws)
                    # Delete all points which fall outside of the bounds.
                    # Unfortunately we have to do that manually
                    mask = (X_train > self.bounds[:, 0]) \
                        & (X_train < self.bounds[:, 1])
                    mask = mask[:, 0] & mask[:, 1]
                    X_train = X_train[mask]
                    # Raise warning if too many points lie outside of the bounds
                    if X_train.shape[0] < n_draws / 4.:
                        warnings.warn("More than 3/4 of the points drawn from "\
                            "the multivariate normal dist. are outside of the "\
                            "bounds. You may want to consider using uniform "\
                            "sampling instead.")

                    # Get the value of the PDF at X_train as a measure of p(x) dx.
                    # This is done to remove the effect of importance sampling.
                    # We only need to keep track of the log as it's easier to use...
                    weights = np.log(rv.pdf(X_train))
                except:
                    warnings.warn("Could not draw points from the "\
                        "multivariate normal dist. Falling back to uniform.")
                    X_train = np.random.uniform(self.bounds[:,0], self.bounds[:,1],
                        (n_draws,len(self.bounds[:,0])))
                    weights = None
        else:
            X_train = np.random.uniform(self.bounds[:,0], self.bounds[:,1],
                (n_draws,len(self.bounds[:,0])))
            weights = None

        # Actual calculation of the KL divergence as sum(p * log(p/q))
        # For this p and q need to be normalized such that they add up to 1.
        # Raise exception for all warnings to catch them.
        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            try:
                # First get p and q by predicting them from the respective models
                logp = surrogate_model_1.predict(X_train)
                logq = surrogate_model_2.predict(X_train)
                # Now we need to normalize everything
                if weights is not None:
                    # We are in log-space so we need to substract the log of
                    # the weights
                    logp = logp - weights
                    logq = logq - weights
                # Still need to make sure that stuff adds up to 1 which is a bit
                # tricky... Luckily there's the logsumexp function
                logp = logp - logsumexp(logp)
                logq = logq - logsumexp(logq)

                # Now that stuff is normalized we can calculate the KL
                # divergence. We also save exp(p) for later
                mask = np.isfinite(logp)
                p = np.exp(logp[mask])
                kl = np.sum(p * (logp[mask] - logq[mask]), axis=0)

            except Exception as e:
                print("KL divergence couldn't be calculated.")
                self.mean = None
                self.cov = None
                return np.nan

            # Also empirically calculate the mean and cov in case we want to
            # draw from them
            try:
                self.mean = np.average(X_train, axis=0, weights=p)
                self.cov = np.cov(X_train.T, aweights=p)

            except Exception as e:
                print("Couldn't get an estimate for mean and cov.")
            return kl
