
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import eigh
from matplotlib.patches import Ellipse
from scipy.special import logsumexp
import matplotlib.transforms as transforms
from numpy.linalg import det
from numpy import trace as tr

class Whitening:
    """
    **TODO:** Fix whitening transformation and make it somewhat robust or delete it
    altogether.


    A class which can pre-transform the posterior in a way
    that it matches a multivariate normal distribution during the Regression step.
    This is done in the hope that by matching a normal distribution the GP will converge
    faster. The transformation used is the *whitening* transformation which is given by

    .. math::
        X_k^i \\to \\frac{\mathbf{R}^{ij} (X_k^j - m^j)}{\sigma^i}\ .
    
    :math:`\mathbf{R}` is the matrix which solves :math:`\mathbf{C} = 
    \mathbf{R}\mathbf{\Lambda}\mathbf{R}` where :math:`\mathbf{C}` is 
    the empirical covariance matrix (along the dimensions of X) and
    :math:`\mathbf{\Lambda}` a diagonal matrix. :math:`m^j` the 
    empirical mean and :math:`\sigma = \sqrt{\mathbf{C}^{ii}}` the 
    empirical standard deviation.

    This step is neccessary if one assumes that the kernel is isotropic while the
    posterior distribution isn't it is however important to note that this is 
    not very numerically robust since the empirical mean and standard deviation are
    weighted by the posterior values which have a high dynamical range. 
    Therefore I suggest that you use an anisotropic kernel instead.

    This class provides three methods:

        * The ``fit`` method sets the mean, covariance as well as their Eigendecompositions 
        * The ``transform`` method applies the whitening transformation to X
        * The ``inverse_transform`` method reverses the transformation applied to the data
    """

    def __init__(self, whiten=True, bounds_normalized=True):
        self.whiten = whiten
        self.bounds_normalized = bounds_normalized
        
        # Declare all the stuff that needs to be defined internally
        self.can_transform = False
        self.kl_divergence = np.nan

        self.mean_ = None
        self.cov = None
    

    def fit(self, surrogate_model):
        """
        Calculates KL divergence and with that mean/cov matrix + fits the PCA transformation
        """

        with warnings.catch_warnings():
            warnings.filterwarnings('error') # Raise exception for all warnings to catch them.

            # First try to calculate the mean and covariance matrix
            try:
                # Save mean and cov for KL divergence
                last_mean = np.copy(self.mean_)
                last_cov = np.copy(self.cov)

                # Get training data from surrogate and preprocess if neccessary
                X_train = surrogate_model._X_train_
                y_train = surrogate_model._y_train_
                y_train = np.exp(y_train - np.max(y_train)) # Turn into unnormalized probability
                if self.bounds_normalized:
                    X_train = surrogate_model.bto.transform(X_train)

                # Calculate mean and cov for KL div and to fit the transformation
                self.mean_ = np.average(X_train, axis=0, weights=y_train)
                self.cov = np.cov(X_train.T, aweights=y_train)
                # self.cov=np.atleast_2d([self.cov])

                    
            except:
                print("Cannot perform PCA")
                self.can_transform = False # Cannot perform PCA transformation
                self.kl_divergence = np.nan # Cannot calculate KL divergence
                return self

            # Next try to calculate the KL divergence
            try:
                last_cov_inv = np.linalg.inv(last_cov)
                kl = 0.5 * (np.log(det(last_cov)) - np.log(det(self.cov)) - X_train.shape[-1]+\
                            tr(last_cov_inv@self.cov)+(last_mean-self.mean_).T @ last_cov_inv @ (last_mean-self.mean_))
                self.kl_divergence = kl
            except Exception as e:
                print(e)
                self.kl_divergence = np.nan

            # lastly try to calculate the PCA transformation
            try:
                self.evals, self.evecs = eigh(self.cov)
                x = (self.evals)**(-0.5)
                self.can_transform = True
            except:
                print("Cannot perform PCA")
                self.can_transform = False
                return self
            
            print("Performed PCA")
            return self

    def transform(self, X, copy=True):
        if not self.can_transform:
            return X
        if copy:
            X = np.copy(X)
        last_factor =  self.evals**(-0.5)
        return (self.evecs @ (X - self.mean_).T).T * last_factor
        #return (self.evecs.T @ (X - self.mean_).T).T * last_factor

    def inverse_transform(self, X, copy=True):
        if not self.can_transform:
            return X
        if copy:
            X = np.copy(X)
        return (self.evecs.T @ (X * self.evals**(0.5)).T).T + self.mean_


class Normalize_bounds:
    """
    A class which transforms all bounds of the prior such that the prior hypervolume
    occupies the unit hypercube in the interval [0, 1]. This is done because of two reasons:

        #. Confining the bounds while fitting the GP regressor ensures that the hyperparameters
           of the GP (particularly length-scales) are within the same order of magnitude if one assumes
           that the non-zero region of the posterior occupies roughly the same fraction of the prior in
           each direction. This is a reasonable assumption for most realistic cases.
        #. If the with of the posterior distribution is similar along every dimension this makes it
           far easier for the optimizer of the acquisition function to navigate the acquisition function
           space (which has the same number of dimensions as the training data) especially if the optimizer
           uses a fixed jump-length.

    Parameters
    ----------
    bounds : array-like, shape = (n_dims, 2)
        Bounds [lower, upper] along each dimension. 

    Attributes
    ----------
    transformed_bounds : array-like, shape = (n_dims, 2)
        Array with [0, 1] along every dimension.  

    bounds_min : array-like, shape = (n_dims,)
        Lower bounds along every dimension.

    bounds_max : array-like, shape = (n_dims,)
        Upper bounds along every dimension.

    **Methods:**

    .. autosummary::
        :toctree: stubs

        transform
        inverse_transform
    """
    def __init__(self, bounds):
        bounds = np.asarray(bounds)
        self.bounds = bounds
        transformed_bounds = np.ones_like(bounds)
        transformed_bounds[:, 0] = 0
        self.transformed_bounds = transformed_bounds
        self.bounds_min = bounds[:, 0]
        self.bounds_max = bounds[:, 1]
        if np.any(self.min > self.max):
            raise ValueError("The bounds must be in dimension-wise order min->max, got \n" + bounds)
    
    def transform(self, X):
        """Transforms X so that all values lie between 0 and 1. 

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_dims)
            X-values that one wants to transform. Must be between bounds.

        Returns
        -------
        X_transformed : array-like, shape = (n_samples, n_dims)
            Transformed X-values
        """
        if np.any(X < self.bounds_min) or np.any(X > self.bounds_max):
            raise ValueError("all X must be between bounds.")
        return (X - self.bounds_min)/(self.bounds_max - self.bounds_min)

    def inverse_transform(self, X_transformed):
        """Applies the inverse transformation

        Parameters
        ----------
        X_transformed : array-like, shape = (n_samples, n_dims)
            Transformed X-values between 0 and 1.

        Returns
        -------
        X : array-like, shape = (n_samples, n_dims)
            Inverse transformed (original) values.
        """
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError("all X must be between 0 and 1.")
        return (X * (self.bounds_max - self.bounds_min)) + self.bounds_min

class Normalize_y:
    """
    Transforms y-values (target values) such that they are centered around 0
    with a standard deviation of 1. This is done so that the constant pre-factor
    in the kernel (constant kernel) stays within a numerically convenient range.

    Attributes
    ----------
    mean_ : float
        Mean of the y-values  

    std_ : float
        Standard deviation of the y-values

    **Methods:**

    .. autosummary::
        :toctree: stubs

        transform
        inverse_transform
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, y):
        """
        Calculates the mean and standard deviation of y
        and saves them.

        Parameters
        ----------
        y : array-like, shape = (n_samples,)
            y-values (target values) that are used to
            determine the mean and std.
        """
        self.mean_ = np.mean(y)
        self.std_ = np.std(y)

    def transform(self, y):
        """Transforms y.

        Parameters
        ----------
        y : array-like, shape = (n_samples,)
            y-values that one wants to transform.

        Returns
        -------
        y_transformed : array-like, shape = (n_samples,)
            Transformed y-values
        """
        if self.mean_ is None or self.std_ is None:
            raise TypeError("mean_ and std_ have not been fit before")
        return (y - self.mean_) / self.std_

    def inverse_transform(self, y_transformed):
        """Applies inverse transformation to y.

        Parameters
        ----------
        y_transformed : array-like, shape = (n_samples,)
            Transformed y-values.

        Returns
        -------
        y : array-like, shape = (n_samples,)
            Original y-values.
        """
        if self.mean_ is None or self.std_ is None:
            raise TypeError("mean_ and std_ have not been fit before")
        return y_transformed * self.std_ + self.mean_

