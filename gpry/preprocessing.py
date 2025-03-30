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

NB: all ``transform``-like methods should return a copy of the input, but avoid
unnecessary ``copy`` statements.
"""

import warnings
from numbers import Number
from itertools import product

import numpy as np
from scipy.linalg import eigh, LinAlgError

from gpry.tools import delta_logp_of_1d_nstd


class DummyPreprocessor:

    is_linear = True

    @classmethod
    def fit(cls, *args, **kwargs):
        pass

    @classmethod
    def transform_bounds(cls, bounds):
        return bounds

    @classmethod
    def transform(cls, _):
        return _

    @classmethod
    def inverse_transform(cls, _):
        return _

    @classmethod
    def transform_scale(cls, _):
        return _

    @classmethod
    def inverse_transform_scale(cls, _):
        return _


class Pipeline_X:
    """
    Used for building a pipeline for preprocessing X-values. This is provided
    with a list of preprocessors in the order they shall be applied. The
    ``transform_scale``, ``fit``, ``transform`` and ``inverse_transform``
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

                def transform(self, X):
                    # This method transforms the X-data. For this the fit
                    # method has to have been called before at least once.
                    ...
                    return X_transformed

                def inverse_transform(self, X):
                    # Applies the inverse transformation to ``transform``.
                    ...
                    return inverse_transformed_X

                def transform_scale(self, scale):
                    # This method should transform a scale in X, e.g. the bounds
                    # of the prior.
                    ...
                    return transformed_scale

                def transform_scale(self, scale):
                    # Applies the inverse transform to ``transform_scale``.
                    ...
                    return transformed_scale
        .. note::

            All the preprocessor objects need to be initialized! Furthermore
            ``transform` and ``inverse_transform`` need to preserve the shape of X.
            All transformations must return a copy (but avoid unnecessary copy
            statements).
    """

    def __init__(self, preprocessors):
        self.preprocessors = preprocessors
        self.fitted = False

    def transform_bounds(self, bounds):
        transformed_bounds = bounds
        for preprocessor in self.preprocessors:
            transformed_bounds = preprocessor.transform_bounds(transformed_bounds)
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
        self.fitted = True
        return self

    def transform(self, X):
        """
        Transform the data through the pipeline
        """
        X_transformed = X
        for preprocessor in self.preprocessors:
            X_transformed = preprocessor.transform(X_transformed)
        return X_transformed

    def inverse_transform(self, X):
        """
        Inverse transform the data through the pipeline (by applying each
        inverse transformation in reverse order).
        """
        X_transformed = X
        for preprocessor in reversed(self.preprocessors):
            X_transformed = preprocessor.inverse_transform(X_transformed)
        return X_transformed

    def transform_scale(self, scale):
        transformed_scale = scale
        for preprocessor in self.preprocessors:
            transformed_scale = preprocessor.transform_scale(transformed_scale)
        return transformed_scale

    def inverse_transform_scale(self, scale):
        """
        Inverse transform the data through the pipeline (by applying each
        inverse transformation in reverse order).
        """
        transformed_scale = scale
        for preprocessor in reversed(self.preprocessors):
            transformed_scale = preprocessor.inverse_transform_scale(transformed_scale)
        return transformed_scale


# TODO: finish and fix
class Whitening:
    r"""
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

    This can help with highly anisotropic distributions, especially if degeneracy
    directions are known a priori.

    When adapting it with `learn=True`, this may not be very numerically robust, since the
    empirical mean and standard deviation are weighted by the posterior values which have
    a high dynamical range. An anisotropic kernel may be preferred.
    """

    def __init__(self, bounds, mean=None, cov=None, learn=False):
        self.transf_matrix, self.inv_transf_matrix = None, None
        if cov is None:
            if not learn:
                raise ValueError("Needs a cov, or to be able to learn it `learn=True`.")
        else:
            try:
                self.transf_matrix, self.inv_transf_matrix = self.prepare_transform(cov)
            except ValueError as excpt:
                raise ValueError(
                    f"Cannot initialize whitening transform: {excpt}"
                ) from excpt
        self.cov = cov
        self.learn = learn
        # If mean is None at the beginning, but cov is defined, assume central point.
        # NB: if cov undefined, mean is ignored.
        if mean is None:
            if self.cov is not None:
                warnings.warn(
                    "Cov passed but not mean. Using the center of the prior hypercube."
                )
                bounds_arr = np.array(bounds)
                mean = (bounds_arr[:, 0] + bounds_arr[:, 1]) / 2
        self.mean = mean

    @staticmethod
    def prepare_transform(cov):
        """
        Compute the relevant elements for the transform from the mean and covmat.

        Raises `ValueError` if it fails at the eigen-decomposition.
        """
        try:
            eigenvals, eigenvecs = eigh(cov)
        except LinAlgError as excpt:
            raise ValueError(
                f"Could not compute the eigen-decomposition of the covmat: {excpt}"
            ) from excpt
        transf_matrix = np.diag(eigenvals**-0.5) @ eigenvecs.T
        inv_transf_matrix = eigenvecs @ np.diag(eigenvals**0.5)
        return transf_matrix, inv_transf_matrix

    @staticmethod
    def compute_mean_cov(X, logp):
        """
        Computes mean and cov using the given points weighted by the given
        log-probabilities.

        Raises ValueError if failed to get a non-singular covmat.
        """
        with warnings.catch_warnings():
            # Raise exception for all warnings to catch them.
            warnings.filterwarnings("error")
            try:
                logp_exp = np.exp(logp - np.max(logp))
                mean = np.average(X, axis=0, weights=logp_exp)
                cov = np.cov(X, aweights=logp_exp, ddof=0)
            except (ZeroDivisionError, TypeError, ValueError, RuntimeWarning) as excpt:
                raise ValueError(
                    f"Could not compute covmat with the given points: {excpt}"
                ) from excpt
        return mean, cov

    def fit(self, X, y):
        """
        Fits the whitening transformation, if initialised with `learn=True`.

        If an error is encountered, keeps the previous transform.
        """
        if not self.learn:
            return self
        warn_msg = "Could not fit a whitening transformation."
        warn_msg_end = "Keeping previous transfrom."
        try:
            self.mean, self.cov = self.compute_mean_cov(X, y)
            self.transf_matrix, self.inv_transf_matrix = self.prepare_transform(self.cov)
        except ValueError as excpt:
            warnings.warn(warn_msg + str(excpt) + warn_msg_end)
        return self

    def transform(self, X):
        if self.cov is None:  # adaptive (or could not initialise), but not fitted yet
            return np.copy(X)
        return (self.transf_matrix @ (X - self.mean).T).T

    def inverse_transform(self, X):
        if self.cov is None:  # adaptive (or could not initialise), but not fitted yet
            return np.copy(X)
        return (self.inv_transf_matrix @ X.T).T + self.mean

    def transform_bounds(self, bounds):
        if self.cov is None:  # adaptive (or could not initialise), but not fitted yet
            return bounds
        vertices = np.array(list(product(*bounds)))
        transf_vertices = self.transform(vertices)
        transf_bounds = np.array(
            [np.min(transf_vertices.T, axis=1), np.max(transf_vertices.T, axis=1)]
        ).T
        print("bounds:", bounds)
        print("vertices:", vertices)
        print("transf_vertices:", transf_vertices)
        print("transf_bounds:", transf_bounds)
        return transf_bounds


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
        self.update_bounds(bounds)
        self.fitted = True  # only needs fitting at init

    def update_bounds(self, bounds):
        bounds = np.asarray(bounds)
        self.bounds = bounds
        self.bounds_min = bounds[:, 0]
        self.bounds_max = bounds[:, 1]
        if np.any(self.bounds_min > self.bounds_max):
            raise ValueError(
                "The bounds must be in dimension-wise order " "min->max, got \n" + bounds
            )

    def transform_bounds(self, bounds):
        transformed_bounds = np.ones_like(bounds)
        transformed_bounds[:, 0] = 0
        return transformed_bounds

    def fit(self, X, y):
        """Fits the transformer (which in reality does nothing)"""

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
        return (X - self.bounds_min) / (self.bounds_max - self.bounds_min)

    def inverse_transform(self, X):
        """Applies the inverse transformation

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_dims)
            Transformed X-values between 0 and 1.

        Returns
        -------
        X : array-like, shape = (n_samples, n_dims)
            Inverse transformed (original) values.
        """
        return (X * (self.bounds_max - self.bounds_min)) + self.bounds_min

    def inverse_transform_scale(self, X):
        """Applies the inverse transformation to an unbounded scale (e.g. the kernel
        length scale).

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_dims)
            Transformed X-values between 0 and 1.

        Returns
        -------
        X : array-like, shape = (n_samples, n_dims)
            Inverse transformed (original) values.
        """
        return X * (self.bounds_max - self.bounds_min)


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

                def transform(self, y):
                    # This method transforms the y-data. For this the fit
                    # method has to have been called before at least once.
                    ...
                    return transformed_y

                def inverse_transform(self, y):
                    # Applies the inverse transformation to ``transform``.
                    ...
                    return inverse_transformed_y

                def transform_scale(self, scale):
                    # This method should transform a scale-like quantity
                    # (e.g. the noise level of the training data) such that
                    # it represents the corresponding scale
                    # of the transformed data.
                    ...
                    return transformed_scale

                def inverse_transform_scale(self, scale):
                    # This method should invert the transformation applied by
                    # ``transform_scale``.
                    ...
                    return inverse_transformed_scale

        .. note::

            All the preprocessor objects need to be initialized! Furthermore
            the `transform` and `inverse_transform` methods need to preserve
            the shape of y. In contrast to the preprocessors for X this does
            not need to contain a method to transform bounds, but it still needs
            one to transform scales such as the the noise level (alpha).
    """

    def __init__(self, preprocessors):
        self.preprocessors = preprocessors
        self.fitted = False

    def fit(self, X, y):
        """
        Consecutively fit several preprocessors by passing the transformed data
        through each one and fitting.
        """
        y_transformed = y
        for preprocessor in self.preprocessors:
            preprocessor.fit(X, y_transformed)
            y_transformed = preprocessor.transform(y_transformed)
        self.fitted = True
        return self

    def transform(self, y):
        """
        Transform the data through the pipeline
        """
        y_transformed = y
        for preprocessor in self.preprocessors:
            y_transformed = preprocessor.transform(y_transformed)
        return y_transformed

    def inverse_transform(self, y):
        """
        Inverse transform the data through the pipeline (by applying each
        inverse transformation in reverse order).
        """
        y_transformed = y
        for preprocessor in reversed(self.preprocessors):
            y_transformed = preprocessor.inverse_transform(y_transformed)
        return y_transformed

    def transform_scale(self, scale):
        """
        Transforms the scale through the pipeline
        """
        scale_transformed = scale
        for preprocessor in self.preprocessors:
            scale_transformed = preprocessor.transform_scale(scale_transformed)
        return scale_transformed

    def inverse_transform_scale(self, scale):
        """
        Inverse transforms the scale through the pipeline
        """
        scale_transformed = scale
        for preprocessor in reversed(self.preprocessors):
            scale_transformed = preprocessor.inverse_transform_scale(scale_transformed)
        return scale_transformed


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

    def __init__(self, use_median=False):
        self.mean_ = None
        self.std_ = None
        self.use_median = bool(use_median)
        if self.use_median:
            self.get_mean_std = lambda y: (lambda y25, y50, y75: (y50, y75 - y25))(
                *np.percentile(y, [25, 50, 75])
            )
        else:
            self.get_mean_std = lambda y: (np.mean(y), np.std(y))

    @property
    def is_linear(self):
        return True

    @property
    def fitted(self):
        return self.mean_ is not None and self.std_ is not None

    def fit(self, X, y):
        """
        Calculates the mean and standard deviation of y
        and saves them.

        Parameters
        ----------
        X : array-like, shape = (n_samples, dimension)
            X-values (target values) that can be used to tune the transformation.
            determine the mean and std.

        y : array-like, shape = (n_samples,)
            y-values (target values) that are used to
            determine the mean and std.
        """
        self.mean_, self.std_ = self.get_mean_std(y[np.isfinite(y)])

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
        if not self.fitted:
            raise TypeError("mean_ and std_ have not been fit before")
        return (y - self.mean_) / self.std_

    def inverse_transform(self, y):
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
        if not self.fitted:
            raise TypeError("mean_ and std_ have not been fit before")
        return (y * self.std_) + self.mean_

    def transform_scale(self, scale):
        if not self.fitted:
            raise TypeError("mean_ and std_ have not been fit before")
        return scale / self.std_  # Divide by the standard deviation

    def inverse_transform_scale(self, scale):
        if not self.fitted:
            raise TypeError("mean_ and std_ have not been fit before")
        return scale * self.std_  # Multiply by the standard deviation


class NormalizeChi2_y(Normalize_y):
    """
    Transforms y-values (target values) such that they are centered around the Gaussian
    1-sigma value with respect to the largest logp, so that the standard deviation is the
    distance between the maximum and the value that defines that 0.

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

    def __init__(self, nsigma=1):
        try:
            assert isinstance(nsigma, Number) and nsigma > 0
        except TypeError as excpt:
            raise TypeError(
                f"nsigma must be a positive number. Got {nsigma} of type {type(nsigma)}."
            ) from excpt
        self.nsigma = nsigma
        self.delta_logp = None
        super().__init__()

    def fit(self, X, y):
        """
        Calculates the mean and standard deviation of y and saves them.

        Parameters
        ----------
        X : array-like, shape = (n_samples, dimension)
            X-values (target values) that can be used to tune the transformation.
            determine the mean and std.

        y : array-like, shape = (n_samples,)
            y-values (target values) that are used to determine the mean and std.

        """
        dim = np.atleast_2d(X).shape[1]
        self.delta_logp = delta_logp_of_1d_nstd(self.nsigma, dim)
        self.mean_ = max(y) - self.delta_logp
        self.std_ = self.delta_logp
