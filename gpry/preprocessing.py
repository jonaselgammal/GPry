"""
This module contains several methods of preprocessing the training and target
values for both the GP Regressor and acquisition module. Instances of different
preprocessors can be chained together thereby building a *pipeline*.

The preprocessors are implemented into the GP Acquisition and GP Regressor
module in a way which performs the transformations *behind the scenes* meaning
that the user can work in the non-transformed space and all transformations will
be performed internally.

You can build your own preprocessor if you want. This requires you to build a
custom class. How to do that for X- and y-preprocessors is explained in the
:class:`Pipeline_X` and :class:`Pipeline_y` classes respectively.
"""

import numpy as np
import warnings
from scipy.linalg import eigh


class Pipeline_X:
    """
    Used for building a pipeline for preprocessing X-values. This is provided
    with a list of preprocessors in the order they shall be applied. The
    ``transform_bounds``, ``fit``, ``transform`` and ``inverse_transform``
    methods can then be called as if the pipeline was a single preprocessor.

    Parameters
    ----------
    preprocessors : list of preprocessors for X.
        The preprocessors in the order that the transformations shall be
        transformed. These need to either be inbuilt preprocessors or you can
        build your own preprocessor. For this you need to build a custom class
        with the signature::

            from gpry.preprocessing import Preprocessor
            class My_X_preprocessor:
                def __init__(self, ...):
                    # Add here any objects that the preprocessor might need
                    ...

                def fit(self, X, y):
                    # This method should fit the transformation
                    # (if neccessary).
                    ...
                    return self

                def transform_bounds(self, bounds):
                    # This method should transform the bounds of the prior. If
                    # the bounds remain unchanged after the transformation this
                    # method should just return the (untransformed) bounds.
                    ...
                    return transformed_bounds

                def transform(self, X, copy=True):
                    # This method transforms the X-data. For this the fit
                    # method has to have been called before at least once.
                    # The 'copy' argument controls whether X is copied or the
                    # transformation is performed in place.
                    ...
                    return X_transformed

                def inverse_transform(self, X, copy=True):
                    # Applies the inverse transformation to 'transform'.
                    ...
                    return inverse_transformed_X

        .. note::

            All the preprocessor objects need to be initialized! Furthermore
            `transform` and `inverse_transform` need to preserve the shape of X
    """

    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def transform_bounds(self, bounds):
        transformed_bounds = bounds
        for preprocessor in self.preprocessors:
            transformed_bounds = preprocessor.transform_bounds(
                transformed_bounds)
        return transformed_bounds

    def fit(self, X, y):
        """
        Consecutively fit several preprocessors by passing the transformed data
        through each one and fitting.
        """
        X_transformed = X
        for preprocessor in self.preprocessors:
            preprocessor.fit(X_transformed, y)
            X_transformed = preprocessor.transform(X_transformed)
        return self

    def transform(self, X, copy=True):
        """
        Transform the data through the pipeline
        """
        X_transformed = np.copy(X) if copy else X
        for preprocessor in self.preprocessors:
            X_transformed = preprocessor.transform(X_transformed, copy=copy)
        return X_transformed

    def inverse_transform(self, X, copy=True):
        """
        Inverse transform the data through the pipeline (by applying each
        inverse transformation in reverse order).
        """
        X_transformed = np.copy(X) if copy else X
        for preprocessor in reversed(self.preprocessors):
            X_transformed = preprocessor.inverse_transform(X, copy=copy)
        return X_transformed


# UNUSED
class Whitening:
    r"""
    **TODO:** Fix whitening transformation and make it somewhat robust or
    delete it altogether.


    A class which can pre-transform the posterior in a way
    that it matches a multivariate normal distribution during the Regression
    step. This is done in the hope that by matching a normal distribution the
    GP will converge faster. The transformation used is the *whitening*
    transformation which is given by

    .. math::
        X_k^i \to \frac{\mathbf{R}^{ij} (X_k^j - m^j)}{\sigma^i}\ .

    :math:`\mathbf{R}` is the matrix which solves :math:`\mathbf{C} =
    \mathbf{R}\mathbf{\Lambda}\mathbf{R}` where :math:`\mathbf{C}` is
    the empirical covariance matrix (along the dimensions of X) and
    :math:`\mathbf{\Lambda}` a diagonal matrix. :math:`m^j` the
    empirical mean and :math:`\sigma = \sqrt{\mathbf{C}^{ii}}` the
    empirical standard deviation.

    This step is neccessary if one assumes that the kernel is isotropic while
    the posterior distribution isn't it is however important to note that this
    is not very numerically robust since the empirical mean and standard
    deviation are weighted by the posterior values which have a high dynamical
    range. Therefore I suggest that you use an anisotropic kernel instead.

    This class provides three methods:

        * The ``fit`` method sets the mean, covariance as well as their
          Eigendecompositions
        * The ``transform`` method applies the whitening transformation to X
        * The ``inverse_transform`` method reverses the transformation
          applied to the data
    """

    def __init__(self):
        # Have a variable which tells whether the whitening
        # transformation works
        self.can_transform = False

        self.mean_ = None
        self.cov = None

    def transform_bounds(self, bounds):
        return bounds

    def fit(self, X, y):
        """
        Fits the whitening transformation
        """
        with warnings.catch_warnings():
            # Raise exception for all warnings to catch them.
            warnings.filterwarnings('error')

            # First try to calculate the mean and covariance matrix
            try:
                # Get training data and transform exponentially
                X_train = np.copy(X)
                y_train = np.copy(y)
                y_train = np.exp(y_train - np.max(y_train))

                # Calculate mean and cov for KL div and to fit the
                # transformation
                self.mean_ = np.average(X_train, axis=0, weights=y_train)
                self.cov = np.cov(X_train.T, aweights=y_train)

            except Exception:
                print("Cannot whiten the data")
                self.can_transform = False  # Cannot perform PCA transformation
                return self

            # Try to calculate eigendecomposition of the covariance matrix
            try:
                self.evals, self.evecs = eigh(self.cov)
                self.last_factor = (self.evals)**(-0.5)
                self.can_transform = True
            except Exception:
                print("Cannot whiten the data")
                self.can_transform = False
                return self

            return self

    def transform(self, X, copy=True):
        if not self.can_transform:
            return X
        if copy:
            X = np.copy(X)
        return (self.evecs @ (X - self.mean_).T).T * self.last_factor

    def inverse_transform(self, X, copy=True):
        if not self.can_transform:
            return X
        if copy:
            X = np.copy(X)
        return (self.evecs.T @ (X * self.last_factor**(0.5)).T).T + self.mean_


class Normalize_bounds:
    """
    A class which transforms all bounds of the prior such that the prior
    hypervolume occupies the unit hypercube in the interval [0, 1].
    This is done because of two reasons:

    #. Confining the bounds while fitting the GP regressor ensures that the
       hyperparameters of the GP (particularly length-scales) are within the
       same order of magnitude if one assumes that the non-zero region of the
       posterior occupies roughly the same fraction of the prior in each
       direction. This is a reasonable assumption for most realistic cases.
    #. If the with of the posterior distribution is similar along every
       dimension this makes it far easier for the optimizer of the acquisition
       function to navigate the acquisition function space (which has the same
       number of dimensions as the training data) especially if the optimizer
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
    """

    def __init__(self, bounds):
        _ = self.transform_bounds(bounds)

    def transform_bounds(self, bounds):
        bounds = np.asarray(bounds)
        self.bounds = bounds
        transformed_bounds = np.ones_like(bounds)
        transformed_bounds[:, 0] = 0
        self.transformed_bounds = transformed_bounds
        self.bounds_min = bounds[:, 0]
        self.bounds_max = bounds[:, 1]
        if np.any(self.bounds_min > self.bounds_max):
            raise ValueError("The bounds must be in dimension-wise order "
                             "min->max, got \n" + bounds)
        return transformed_bounds

    def fit(self, X, y):
        """Fits the transformer (which in reality does nothing)
        """
        pass

    def transform(self, X, copy=True):
        """Transforms X so that all values lie between 0 and 1.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_dims)
            X-values that one wants to transform. Must be between bounds.

        copy : bool, default: True
            Return a copy if True, or transform in place if False.

        Returns
        -------
        X_transformed : array-like, shape = (n_samples, n_dims)
            Transformed X-values
        """
        # if np.any(X < self.bounds_min) or np.any(X > self.bounds_max):
        #     raise ValueError("all X must be between bounds.")
        X = np.copy(X) if copy else X
        return (X - self.bounds_min) / (self.bounds_max - self.bounds_min)

    def inverse_transform(self, X, copy=True):
        """Applies the inverse transformation

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_dims)
            Transformed X-values between 0 and 1.

        copy : bool, default: True
            Return a copy if True, or transform in place if False.

        Returns
        -------
        X : array-like, shape = (n_samples, n_dims)
            Inverse transformed (original) values.
        """
        # if np.any(X < 0) or np.any(X > 1):
        #     raise ValueError("all X must be between 0 and 1.")
        X = np.copy(X) if copy else X
        return (X * (self.bounds_max - self.bounds_min)) + self.bounds_min


class Pipeline_y:
    """
    Used for building a pipeline for preprocessing y-values. This is provided
    with a list of preprocessors in the order they shall be applied. The
    ``fit``, ``transform`` and ``inverse_transform`` methods can then be called
    as if the pipeline was a single preprocessor.

    Parameters
    ----------
    preprocessors : list of preprocessors for y.
        The preprocessors in the order that the transformations shall be
        transformed. These need to either be inbuilt preprocessors or you can
        build your own preprocessor. For this you need to build a custom class
        with the signature::

            class My_y_preprocessor:
                def __init__(self, ...):
                    # Add here any objects that the preprocessor might need
                    ...

                def fit(self, X, y):
                    # This method should fit the transformation
                    # (if neccessary).
                    ...
                    return self

                def transform_noise_level(self, noise_level):
                    # This method should transform the noise level of
                    # the training data such that it represents the noise level
                    # of the transformed data.
                    ...
                    return transformed_noise_level

                def inverse_transform_noise_level(self, noise_level):
                    # This method should invert the transformation applied by
                    # 'transform_noise_level'.
                    ...
                    return inverse_transformed_noise_level

                def transform(self, y, copy=True):
                    # This method transforms the y-data. For this the fit
                    # method has to have been called before at least once.
                    # The 'copy' argument controls whether y is copied or the
                    # transformation is performed in place.
                    ...
                    return transformed_y

                def inverse_transform(self, y, copy=True):
                    # Applies the inverse transformation to 'transform'.
                    ...
                    return inverse_transformed_y

        .. note::

            All the preprocessor objects need to be initialized! Furthermore
            the `transform` and `inverse_transform` methods need to preserve
            the shape of y. In contrast to the preprocessors for X this does
            not need to contain a method to transform bounds but instead one to
            transform the noise level (alpha).
    """

    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def fit(self, X, y):
        """
        Consecutively fit several preprocessors by passing the transformed data
        through each one and fitting.
        """
        y_transformed = y
        for preprocessor in self.preprocessors:
            preprocessor.fit(X, y_transformed)
            y_transformed = preprocessor.transform(y_transformed)
        return self

    def transform_noise_level(self, noise_level):
        """
        Transforms the noise level through the pipeline
        """
        noise_level_transformed = np.copy(noise_level)
        for preprocessor in self.preprocessors:
            noise_level_transformed = \
                preprocessor.transform_noise_level(noise_level_transformed)
        return noise_level_transformed

    def inverse_transform_noise_level(self, noise_level):
        """
        Inverse transforms the noise level through the pipeline
        """
        noise_level_transformed = np.copy(noise_level)
        for preprocessor in reversed(self.preprocessors):
            noise_level_transformed = \
                preprocessor.inverse_transform_noise_level(
                    noise_level_transformed)
        return noise_level_transformed

    def transform(self, y, copy=True):
        """
        Transform the data through the pipeline
        """
        y_transformed = np.copy(y) if copy else y
        for preprocessor in self.preprocessors:
            y_transformed = preprocessor.transform(y_transformed, copy=copy)
        return y_transformed

    def inverse_transform(self, y, copy=True):
        """
        Inverse transform the data through the pipeline (by applying each
        inverse transformation in reverse order).
        """
        y_transformed = np.copy(y) if copy else y
        for preprocessor in reversed(self.preprocessors):
            y_transformed = preprocessor.inverse_transform(y, copy=copy)
        return y_transformed


class Normalize_y:
    """
    Transforms y-values (target values) such that they are centered around 0
    with a standard deviation of 1. This is done so that the constant
    pre-factor in the kernel (constant kernel) stays within a numerically
    convenient range.

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

    def fit(self, X, y):
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

    def transform_noise_level(self, noise_level, copy=True):
        if self.mean_ is None or self.std_ is None:
            raise TypeError("mean_ and std_ have not been fit before")
        noise_level = np.copy(noise_level) if copy else noise_level
        return noise_level / self.std_  # Divide by the standard deviation

    def inverse_transform_noise_level(self, noise_level, copy=True):
        if self.mean_ is None or self.std_ is None:
            raise TypeError("mean_ and std_ have not been fit before")
        noise_level = np.copy(noise_level) if copy else noise_level
        return noise_level * self.std_  # Multiply by the standard deviation

    def transform(self, y, copy=True):
        """Transforms y.

        Parameters
        ----------
        y : array-like, shape = (n_samples,)
            y-values that one wants to transform.

        copy : bool, default: True
            Return a copy if True, or transform in place if False.

        Returns
        -------
        y_transformed : array-like, shape = (n_samples,)
            Transformed y-values
        """
        if self.mean_ is None or self.std_ is None:
            raise TypeError("mean_ and std_ have not been fit before")
        y = np.copy(y) if copy else y
        return (y - self.mean_) / self.std_

    def inverse_transform(self, y_transformed, copy=True):
        """Applies inverse transformation to y.

        Parameters
        ----------
        y_transformed : array-like, shape = (n_samples,)
            Transformed y-values.

        copy : bool, default: True
            Return a copy if True, or transform in place if False.

        Returns
        -------
        y : array-like, shape = (n_samples,)
            Original y-values.
        """
        if self.mean_ is None or self.std_ is None:
            raise TypeError("mean_ and std_ have not been fit before")
        y = np.copy(y_transformed) if copy else y_transformed
        return (y * self.std_) + self.mean_
