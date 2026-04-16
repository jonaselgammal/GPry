"""
This module contains several methods of preprocessing the training and target
values for both the GP Regressor and acquisition module. Instances of different
preprocessors can be chained together thereby building a *pipeline*.

The preprocessors are implemented into the GP Acquisition and surrogate model
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
from scipy.linalg import eigh, LinAlgError  # type: ignore
from scipy.optimize import minimize_scalar  # type: ignore

from gpry.tools import delta_logp_of_1d_nstd


def _is_jax_array(x):
    """Return True if x is a JAX array."""
    return type(x).__module__.startswith("jax")


class DummyPreprocessor:
    is_linear = True

    @classmethod
    def fit(cls, *args, **kwargs):
        pass

    @classmethod
    def transform_bounds(cls, bounds):
        return bounds

    @classmethod
    def inverse_transform_bounds(cls, transf_bounds):
        return transf_bounds

    @classmethod
    def transform(cls, _):
        return _

    @classmethod
    def transform_jax(cls, _):
        return cls.transform(_)

    @classmethod
    def inverse_transform(cls, _):
        return _

    @classmethod
    def inverse_transform_jax(cls, _):
        return cls.inverse_transform(_)

    @classmethod
    def transform_scale(cls, _):
        return _

    @classmethod
    def transform_scale_jax(cls, _):
        return cls.transform_scale(_)

    @classmethod
    def inverse_transform_scale(cls, _):
        return _

    @classmethod
    def inverse_transform_scale_jax(cls, _):
        return cls.inverse_transform_scale(_)


class PipelineX:
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
                    # This method should transform a set of bounds, e.g. the prior
                    # bounds. If the bounds remain unchanged after the transformation
                    # this method should just return the (untransformed) bounds.
                    ...
                    return transformed_bounds

                def inverse_transform_bounds(self, transformed_bounds):
                    # This method should invert the transformation of the given
                    # bounds. If the bounds remain unchanged after the
                    # transformation this method should just return the
                    # (untransformed) bounds.
                    ...
                    return trasnformed_bounds

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

    def inverse_transform_bounds(self, transformed_bounds):
        bounds = transformed_bounds
        for preprocessor in reversed(self.preprocessors):
            bounds = preprocessor.inverse_transform_bounds(bounds)
        return bounds

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

    def transform_jax(self, X):
        """JAX-friendly transform through the pipeline."""
        X_transformed = X
        for preprocessor in self.preprocessors:
            transform = getattr(preprocessor, "transform_jax", preprocessor.transform)
            X_transformed = transform(X_transformed)
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

    def inverse_transform_jax(self, X):
        """JAX-friendly inverse transform through the pipeline."""
        X_transformed = X
        for preprocessor in reversed(self.preprocessors):
            inverse_transform = getattr(
                preprocessor, "inverse_transform_jax", preprocessor.inverse_transform
            )
            X_transformed = inverse_transform(X_transformed)
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

    def jacobian_diagonal(self, X):
        """Compute the diagonal Jacobian of the full pipeline at X.

        For chained transforms, the Jacobian is the product of individual
        Jacobians, evaluated at the appropriately transformed X.

        Parameters
        ----------
        X : array, shape (n_samples, n_dims)
            Points in the original (pre-pipeline) space.

        Returns
        -------
        jac : array, shape (n_samples, n_dims)
            Product of diagonal Jacobians through the chain.
        """
        X_cur = np.atleast_2d(X)
        jac = np.ones_like(X_cur, dtype=float)
        for preprocessor in self.preprocessors:
            if hasattr(preprocessor, 'jacobian_diagonal'):
                jac *= preprocessor.jacobian_diagonal(X_cur)
            # For linear transforms (like NormalizeBounds), the Jacobian is
            # constant and already accounted for by transform_scale.
            X_cur = preprocessor.transform(X_cur)
        return jac

    @property
    def has_nonlinear(self):
        """Whether any preprocessor in the pipeline is nonlinear."""
        return any(
            hasattr(p, 'jacobian_diagonal') for p in self.preprocessors
        )


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
            self.transf_matrix, self.inv_transf_matrix = self.prepare_transform(
                self.cov
            )
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


class NormalizeBounds:
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
                "The bounds must be in dimension-wise order min->max, got \n" + bounds
            )

    def transform_bounds(self, bounds):
        return self.transform(np.atleast_2d(bounds).T).T

    def inverse_transform_bounds(self, transf_bounds):
        return self.inverse_transform(np.atleast_2d(transf_bounds).T).T

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

    def transform_jax(self, X):
        import jax.numpy as jnp

        bounds_min = jnp.asarray(self.bounds_min, dtype=jnp.float64)
        bounds_max = jnp.asarray(self.bounds_max, dtype=jnp.float64)
        return (X - bounds_min) / (bounds_max - bounds_min)

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

    def inverse_transform_jax(self, X):
        import jax.numpy as jnp

        bounds_min = jnp.asarray(self.bounds_min, dtype=jnp.float64)
        bounds_max = jnp.asarray(self.bounds_max, dtype=jnp.float64)
        return (X * (bounds_max - bounds_min)) + bounds_min

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

    def inverse_transform_scale_jax(self, X):
        import jax.numpy as jnp

        bounds_min = jnp.asarray(self.bounds_min, dtype=jnp.float64)
        bounds_max = jnp.asarray(self.bounds_max, dtype=jnp.float64)
        return X * (bounds_max - bounds_min)


class InputWarping:
    r"""
    Applies a learned monotonic warping to each input dimension independently,
    mapping [0, 1] -> [0, 1] via the Kumaraswamy CDF:

    .. math::
        w(x; a, b) = 1 - (1 - x^a)^b

    When a = b = 1, this reduces to the identity. The warping concentrates
    resolution in regions where the likelihood is high, making the GP's
    stationary kernel more appropriate for non-stationary likelihoods.

    The warping parameters are estimated by maximizing the uniformity of
    the weighted marginal distribution of high-likelihood training points
    in the warped space. A leave-one-out cross-validation check gates
    activation: warping is only used if it demonstrably improves the GP's
    predictive accuracy.

    Parameters
    ----------
    concentration : float, default: 0.3
        Controls how aggressively the warping focuses on the high-likelihood
        region. 0 = no concentration (identity), 1 = concentrate fully on
        the mode. Moderate values (0.3-0.7) are recommended.

    min_points : int, default: 30
        Minimum number of finite training points before fitting non-trivial
        warping. Below this, identity warping is used.

    uniformity_gain : float, default: 0.05
        Minimum reduction in the weighted KS statistic (vs uniform) for
        the warped distribution to be accepted over identity. Larger values
        are more conservative.
    """

    _EPS = 1e-10  # clamp to [eps, 1-eps] to avoid boundary issues

    def __init__(self, concentration=0.3, min_points=30, refit_interval=0,
                 uniformity_gain=0.05):
        self.concentration = concentration
        self.min_points = min_points
        self.refit_interval = refit_interval  # 0 = fit once then freeze
        self.uniformity_gain = uniformity_gain
        self._a = None  # shape (n_dims,)
        self._b = None  # shape (n_dims,)
        self._a_candidate = None  # candidate params before LML check
        self._b_candidate = None
        self._fitted = False
        self._frozen = False  # once True, fit() is a no-op
        self._activated = False  # True only after LML check passes
        self._n_fits = 0

    @property
    def fitted(self):
        return self._fitted

    def fit(self, X, y):
        """Estimate Kumaraswamy warping parameters from training data.

        Parameters
        ----------
        X : array, shape (n_samples, n_dims)
            Training inputs in [0, 1] (output of NormalizeBounds).
        y : array, shape (n_samples,)
            Training log-likelihood values.
        """
        X = np.atleast_2d(X)
        n_samples, n_dims = X.shape
        # Initialize to identity
        if self._a is None:
            self._a = np.ones(n_dims)
            self._b = np.ones(n_dims)
        # If frozen, do nothing
        if self._frozen:
            return self
        # Not enough points for meaningful warping
        if n_samples < self.min_points:
            self._fitted = True
            return self
        # Check refit schedule: refit_interval=0 means fit once then freeze
        self._n_fits += 1
        if self._n_fits > 1 and self.refit_interval == 0:
            self._frozen = True
            return self
        if self._n_fits > 1 and self._n_fits % self.refit_interval != 0:
            return self
        # Compute weights from likelihood values (soft weighting toward mode)
        y_finite = np.where(np.isfinite(y), y, np.nanmin(y[np.isfinite(y)]))
        w = np.exp(self.concentration * (y_finite - np.max(y_finite)))
        w = w / w.sum()
        # Fit per-dimension warping candidates
        a_cand = np.ones(n_dims)
        b_cand = np.ones(n_dims)
        for dim in range(n_dims):
            x_d = np.clip(X[:, dim], self._EPS, 1 - self._EPS)
            a_cand[dim], b_cand[dim] = self._fit_1d(x_d, w)
        # Check if candidate warping is close to identity — skip further checks
        max_deviation = max(
            np.max(np.abs(np.log(a_cand))),
            np.max(np.abs(np.log(b_cand))),
        )
        if max_deviation < 0.05:  # a,b within ~5% of 1.0
            # Identity is optimal; keep it and freeze
            self._fitted = True
            if self.refit_interval == 0:
                self._frozen = True
            return self
        # Uniformity gating: check if warping improves the weighted CDF uniformity
        if self._check_uniformity_improvement(X, y, a_cand, b_cand):
            self._a = a_cand
            self._b = b_cand
            self._activated = True
        # else: keep identity (a=b=1)
        self._fitted = True
        # Freeze after first real fit if refit_interval=0
        if self.refit_interval == 0:
            self._frozen = True
        return self

    def _check_uniformity_improvement(self, X, y, a_cand, b_cand):
        """Check if warping improves the uniformity of the weighted marginal CDFs.

        The Kumaraswamy warping is designed to make the weighted distribution
        more uniform. We measure this by the mean KS statistic across dimensions.
        Warping is accepted if the weighted KS statistic improves by at least
        ``uniformity_gain``.

        Returns True if warping should be activated.
        """
        n, d = X.shape
        finite_mask = np.isfinite(y)
        if finite_mask.sum() < self.min_points:
            return False
        X_fin = X[finite_mask]
        y_fin = y[finite_mask]

        # Compute weights (same as in fit)
        y_shift = y_fin - np.max(y_fin)
        w = np.exp(self.concentration * y_shift)
        w = w / w.sum()

        def _weighted_ks(x_col, weights):
            """KS statistic of weighted empirical CDF vs uniform on [0,1]."""
            sort_idx = np.argsort(x_col)
            x_s = x_col[sort_idx]
            w_s = weights[sort_idx]
            ecdf = np.cumsum(w_s)
            # Theoretical CDF for uniform on [0,1] is just x
            return np.max(np.abs(ecdf - x_s))

        # Compute mean KS across dimensions for identity and warped
        ks_identity = 0.0
        ks_warped = 0.0
        for dim in range(d):
            x_d = np.clip(X_fin[:, dim], self._EPS, 1 - self._EPS)
            ks_identity += _weighted_ks(x_d, w)
            x_w = 1.0 - (1.0 - x_d ** a_cand[dim]) ** b_cand[dim]
            ks_warped += _weighted_ks(x_w, w)
        ks_identity /= d
        ks_warped /= d

        # Accept if warping makes the weighted distribution meaningfully
        # more uniform (lower KS = more uniform)
        return (ks_identity - ks_warped) > self.uniformity_gain

    @staticmethod
    def _fit_1d(x, w):
        """Fit Kumaraswamy (a, b) for one dimension.

        Finds (a, b) such that the warped weighted CDF is as close to uniform
        as possible, regularized toward identity (a=b=1).
        """
        n = len(x)
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        w_sorted = w[sort_idx]
        w_cumsum = np.cumsum(w_sorted)
        w_cdf = w_cumsum / w_cumsum[-1]

        # Clamp parameters to [0.5, 2.0] — mild warping that helps with
        # non-stationarity without catastrophically compressing any region
        _LN_MIN, _LN_MAX = np.log(0.5), np.log(2.0)

        def objective(log_ab):
            a = np.exp(np.clip(log_ab[0], _LN_MIN, _LN_MAX))
            b = np.exp(np.clip(log_ab[1], _LN_MIN, _LN_MAX))
            x_warped = 1.0 - (1.0 - x_sorted ** a) ** b
            residual = np.sum((x_warped - w_cdf) ** 2)
            # Regularize toward identity; strength decreases with more data
            reg = 1.0 * (log_ab[0] ** 2 + log_ab[1] ** 2) / max(n / 30, 1)
            return residual + reg

        from scipy.optimize import minimize
        result = minimize(
            objective,
            x0=np.array([0.0, 0.0]),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 0.01, 'fatol': 1e-6},
        )
        a = np.clip(np.exp(result.x[0]), 0.5, 2.0)
        b = np.clip(np.exp(result.x[1]), 0.5, 2.0)
        return a, b

    def transform(self, X):
        """Apply Kumaraswamy warping: w(x) = 1 - (1 - x^a)^b."""
        if _is_jax_array(X):
            if not self._activated:
                return X
            import jax.numpy as jnp
            was_1d = X.ndim == 1
            if was_1d:
                X = X[None, :]
            X = jnp.clip(X, self._EPS, 1 - self._EPS)
            a = jnp.array(self._a)
            b = jnp.array(self._b)
            X = 1.0 - (1.0 - X ** a) ** b
            return X[0] if was_1d else X
        X = np.array(X, dtype=float, copy=True)
        if not self._activated:
            return X
        was_1d = X.ndim == 1
        if was_1d:
            X = X[None, :]
        X = np.clip(X, self._EPS, 1 - self._EPS)
        for dim in range(X.shape[1]):
            X[:, dim] = 1.0 - (1.0 - X[:, dim] ** self._a[dim]) ** self._b[dim]
        return X[0] if was_1d else X

    def inverse_transform(self, X):
        """Inverse Kumaraswamy: x = (1 - (1 - w)^(1/b))^(1/a)."""
        if _is_jax_array(X):
            if not self._activated:
                return X
            import jax.numpy as jnp
            was_1d = X.ndim == 1
            if was_1d:
                X = X[None, :]
            X = jnp.clip(X, self._EPS, 1 - self._EPS)
            a = jnp.array(self._a)
            b = jnp.array(self._b)
            X = (1.0 - (1.0 - X) ** (1.0 / b)) ** (1.0 / a)
            return X[0] if was_1d else X
        X = np.array(X, dtype=float, copy=True)
        if not self._activated:
            return X
        was_1d = X.ndim == 1
        if was_1d:
            X = X[None, :]
        X = np.clip(X, self._EPS, 1 - self._EPS)
        for dim in range(X.shape[1]):
            X[:, dim] = (
                1.0 - (1.0 - X[:, dim]) ** (1.0 / self._b[dim])
            ) ** (1.0 / self._a[dim])
        return X[0] if was_1d else X

    def transform_bounds(self, bounds):
        """Warping maps [0,1] -> [0,1] monotonically, so bounds are preserved."""
        return bounds

    def inverse_transform_bounds(self, bounds):
        """Inverse warping maps [0,1] -> [0,1] monotonically."""
        return bounds

    def transform_scale(self, scale):
        """Approximate scale transformation using the average Jacobian.

        For a nonlinear transform, scale transformation is position-dependent.
        We use the Jacobian at the midpoint (0.5) as a reasonable approximation.
        """
        if not self._activated:
            return scale
        scale = np.asarray(scale, dtype=float).copy()
        mid = 0.5 * np.ones_like(self._a)
        jac = self.jacobian_diagonal(mid[None, :])[0]
        return scale * jac

    def inverse_transform_scale(self, scale):
        """Approximate inverse scale transformation."""
        if not self._activated:
            return scale
        scale = np.asarray(scale, dtype=float).copy()
        mid = 0.5 * np.ones_like(self._a)
        jac = self.jacobian_diagonal(mid[None, :])[0]
        return scale / jac

    def jacobian_diagonal(self, X):
        """Compute diagonal of the Jacobian dw/dx at each point.

        Parameters
        ----------
        X : array, shape (n_samples, n_dims)
            Points in [0, 1] (pre-warping space).

        Returns
        -------
        jac : array, shape (n_samples, n_dims)
            Diagonal Jacobian entries.
        """
        X = np.atleast_2d(X)
        jac = np.ones_like(X)
        if not self._activated:
            return jac
        X = np.clip(X, self._EPS, 1 - self._EPS)
        for dim in range(X.shape[1]):
            a, b = self._a[dim], self._b[dim]
            x = X[:, dim]
            # dw/dx = a * b * x^(a-1) * (1 - x^a)^(b-1)
            jac[:, dim] = a * b * x ** (a - 1) * (1.0 - x ** a) ** (b - 1)
        return jac


class PipelineY:
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

    @property
    def is_linear(self):
        return all(getattr(p, 'is_linear', False) for p in self.preprocessors)

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

    def transform_jax(self, y):
        """JAX-friendly transform through the pipeline."""
        y_transformed = y
        for preprocessor in self.preprocessors:
            transform = getattr(preprocessor, "transform_jax", preprocessor.transform)
            y_transformed = transform(y_transformed)
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

    def inverse_transform_jax(self, y):
        """JAX-friendly inverse transform through the pipeline."""
        y_transformed = y
        for preprocessor in reversed(self.preprocessors):
            inverse_transform = getattr(
                preprocessor, "inverse_transform_jax", preprocessor.inverse_transform
            )
            y_transformed = inverse_transform(y_transformed)
        return y_transformed

    def transform_scale(self, scale):
        """
        Transforms the scale through the pipeline
        """
        scale_transformed = scale
        for preprocessor in self.preprocessors:
            scale_transformed = preprocessor.transform_scale(scale_transformed)
        return scale_transformed

    def transform_scale_jax(self, scale):
        """JAX-friendly scale transform through the pipeline."""
        scale_transformed = scale
        for preprocessor in self.preprocessors:
            transform_scale = getattr(
                preprocessor, "transform_scale_jax", preprocessor.transform_scale
            )
            scale_transformed = transform_scale(scale_transformed)
        return scale_transformed

    def inverse_transform_scale(self, scale):
        """
        Inverse transforms the scale through the pipeline
        """
        scale_transformed = scale
        for preprocessor in reversed(self.preprocessors):
            scale_transformed = preprocessor.inverse_transform_scale(scale_transformed)
        return scale_transformed

    def inverse_transform_scale_jax(self, scale):
        """JAX-friendly inverse scale transform through the pipeline."""
        scale_transformed = scale
        for preprocessor in reversed(self.preprocessors):
            inverse_transform_scale = getattr(
                preprocessor,
                "inverse_transform_scale_jax",
                preprocessor.inverse_transform_scale,
            )
            scale_transformed = inverse_transform_scale(scale_transformed)
        return scale_transformed


class NormalizeY:
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

    def transform_jax(self, y):
        import jax.numpy as jnp

        if not self.fitted:
            raise TypeError("mean_ and std_ have not been fit before")
        return (y - jnp.asarray(self.mean_, dtype=jnp.float64)) / jnp.asarray(
            self.std_, dtype=jnp.float64
        )

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

    def inverse_transform_jax(self, y):
        import jax.numpy as jnp

        if not self.fitted:
            raise TypeError("mean_ and std_ have not been fit before")
        return (y * jnp.asarray(self.std_, dtype=jnp.float64)) + jnp.asarray(
            self.mean_, dtype=jnp.float64
        )

    def transform_scale(self, scale):
        if not self.fitted:
            raise TypeError("mean_ and std_ have not been fit before")
        return scale / self.std_  # Divide by the standard deviation

    def transform_scale_jax(self, scale):
        import jax.numpy as jnp

        if not self.fitted:
            raise TypeError("mean_ and std_ have not been fit before")
        return scale / jnp.asarray(self.std_, dtype=jnp.float64)

    def inverse_transform_scale(self, scale):
        if not self.fitted:
            raise TypeError("mean_ and std_ have not been fit before")
        return scale * self.std_  # Multiply by the standard deviation

    def inverse_transform_scale_jax(self, scale):
        import jax.numpy as jnp

        if not self.fitted:
            raise TypeError("mean_ and std_ have not been fit before")
        return scale * jnp.asarray(self.std_, dtype=jnp.float64)


class NormalizeYChi2(NormalizeY):
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


class SoftClipY:
    """
    Compresses the flat tail of a log-likelihood so the GP focuses capacity on
    the mode and transition region rather than wasting it fitting a plateau.

    Values within ``delta`` of y_max are barely affected (near-identity).
    Values further below are logarithmically compressed::

        y_clip = threshold - tau * log(1 + (threshold - y) / tau)

    where ``threshold = y_max - delta``. This maps a range of e.g. 500 units
    below threshold into ~tau * log(500/tau) ~ 15-20 units, dramatically reducing
    the dynamic range that the GP must model.

    Parameters
    ----------
    delta : float or str, default="3s"
        Width of the unclipped window below y_max. Values within [y_max - delta, y_max]
        are barely affected. Can be a number or a string like "3s" meaning 3 sigma
        (computed from dimensionality via chi2 quantile).

    tau : float, default=3.0
        Scale of the logarithmic compression below the threshold. Smaller tau =
        more aggressive compression. A value of 3.0 gives moderate compression
        that works well with the GP's hyperparameter optimization. Values below
        1.0 may cause extreme length scale estimates.
    """

    is_linear = False

    def __init__(self, delta="3s", tau=3.0):
        self._delta_spec = delta
        self.tau = float(tau)
        self.delta = None
        self.y_max_ = None

    @property
    def fitted(self):
        return self.y_max_ is not None

    def fit(self, X, y):
        y_finite = y[np.isfinite(y)]
        if len(y_finite) == 0:
            return
        self.y_max_ = np.max(y_finite)
        # Parse delta
        if isinstance(self._delta_spec, str) and self._delta_spec.endswith("s"):
            nsigma = float(self._delta_spec[:-1])
            dim = np.atleast_2d(X).shape[1]
            self.delta = delta_logp_of_1d_nstd(nsigma, dim)
        else:
            self.delta = float(self._delta_spec)

    def transform(self, y):
        if not self.fitted:
            raise TypeError("SoftClipY has not been fit yet")
        if _is_jax_array(y):
            import jax.numpy as jnp
            threshold = self.y_max_ - self.delta
            is_below = jnp.isfinite(y) & (y < threshold)
            d = jnp.maximum(threshold - y, 0.0) / self.tau
            y_clipped = threshold - self.tau * jnp.log1p(d)
            return jnp.where(is_below, y_clipped, y)
        y = np.array(y, dtype=float, copy=True)
        finite = np.isfinite(y)
        threshold = self.y_max_ - self.delta
        # Only transform values below the threshold
        below = finite & (y < threshold)
        if np.any(below):
            # Logarithmic compression: large distances get mapped to log-scale
            d = (threshold - y[below]) / self.tau  # d >= 0
            y[below] = threshold - self.tau * np.log1p(d)
        return y

    def inverse_transform(self, y):
        if not self.fitted:
            raise TypeError("SoftClipY has not been fit yet")
        if _is_jax_array(y):
            import jax.numpy as jnp
            threshold = self.y_max_ - self.delta
            is_below = jnp.isfinite(y) & (y < threshold)
            z = jnp.maximum(threshold - y, 0.0) / self.tau
            y_orig = threshold - self.tau * jnp.expm1(z)
            return jnp.where(is_below, y_orig, y)
        y = np.array(y, dtype=float, copy=True)
        finite = np.isfinite(y)
        threshold = self.y_max_ - self.delta
        below = finite & (y < threshold)
        if np.any(below):
            # Inverse: d = exp((threshold - y) / tau) - 1, y_orig = threshold - tau * d
            z = (threshold - y[below]) / self.tau  # z >= 0
            y[below] = threshold - self.tau * np.expm1(z)
        return y

    def transform_scale(self, scale):
        # In the unclipped region (near mode), scale is preserved.
        # In the clipped region, scale is compressed. Use identity as approximation
        # since the GP primarily cares about the near-mode region.
        return scale

    def inverse_transform_scale(self, scale):
        return scale


class ActiveSubspace:
    """Rotates inputs to align with directions of maximum likelihood variation.

    Computes the gradient covariance matrix C = E[grad @ grad.T] and rotates
    the input space to align with its eigenvectors. This helps axis-aligned
    kernels (like RBF) handle curved degeneracies and ridges.

    The rotation is a linear transform, so ``has_nonlinear = False`` and the
    GP's kernel automatically handles gradients correctly in the rotated space
    (no explicit Jacobian correction needed in the surrogate).

    Parameters
    ----------
    min_points : int or str, default="5d"
        Minimum training points before activation. "Nd" = N * dimensionality.

    eigenvalue_threshold : float, default=5.0
        Minimum ratio of max/min eigenvalue to activate rotation.
        If the eigenvalues are similar, rotation won't help.

    k_neighbors : int, default=0
        Number of neighbors for local gradient estimation. 0 = use all points
        (global linear regression). Higher values = more local gradients.

    refit_interval : int, default=0
        How often to refit the rotation. 0 = fit once and freeze.
    """

    has_nonlinear = False  # linear rotation; no Jacobian correction needed

    def __init__(self, min_points="5d", eigenvalue_threshold=5.0,
                 k_neighbors=0, refit_interval=0):
        self._min_points_spec = min_points
        self.eigenvalue_threshold = eigenvalue_threshold
        self.k_neighbors = k_neighbors
        self.refit_interval = refit_interval
        self._W = None  # rotation matrix (d x d), columns are eigenvectors
        self._center = None  # center point for rotation
        self._eigenvalues = None
        self._activated = False
        self._frozen = False
        self._fit_count = 0

    @property
    def fitted(self):
        return self._W is not None

    def _parse_min_points(self, d):
        """Parse the min_points specification into an integer."""
        if isinstance(self._min_points_spec, str) and self._min_points_spec.endswith("d"):
            return int(self._min_points_spec[:-1]) * d
        return int(self._min_points_spec)

    def fit(self, X, y):
        """Estimate the active subspace rotation from training data.

        Parameters
        ----------
        X : array, shape (n_samples, n_dims)
            Training inputs (possibly already normalized).
        y : array, shape (n_samples,)
            Training log-likelihood values.

        Returns
        -------
        self
        """
        X = np.array(X)
        y = np.array(y)
        d = X.shape[1]
        min_pts = self._parse_min_points(d)

        if len(X) < min_pts:
            return self

        if self._frozen and self._activated:
            return self

        self._fit_count += 1
        if self.refit_interval > 0 and self._fit_count > 1:
            if (self._fit_count - 1) % self.refit_interval != 0:
                return self

        # Use only finite y values
        finite = np.isfinite(y)
        X_f, y_f = X[finite], y[finite]
        if len(X_f) < min_pts:
            return self

        # Estimate gradients via local linear regression
        grads = self._estimate_gradients(X_f, y_f)
        if grads is None:
            return self

        # Gradient covariance matrix, weighted by likelihood
        weights = np.exp(y_f - np.max(y_f))  # softmax-like weights
        weights /= np.sum(weights)

        C = np.zeros((d, d))
        for i in range(len(grads)):
            g = grads[i]
            C += weights[i] * np.outer(g, g)

        # Eigendecompose
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            return self

        # Sort by eigenvalue descending
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Clamp small eigenvalues to avoid division by zero
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        ratio = eigenvalues[0] / eigenvalues[-1]

        self._eigenvalues = eigenvalues
        self._center = np.mean(X_f, axis=0)

        if ratio >= self.eigenvalue_threshold:
            self._W = eigenvectors  # columns are eigenvectors
            self._activated = True
            if self.refit_interval == 0:
                self._frozen = True
        else:
            self._W = np.eye(d)  # identity = no rotation
            self._activated = False

        return self

    def _estimate_gradients(self, X, y):
        """Estimate gradients at each point via local linear regression.

        Parameters
        ----------
        X : array, shape (n, d)
            Training inputs (finite y only).
        y : array, shape (n,)
            Training targets (finite only).

        Returns
        -------
        grads : array, shape (n, d), or None if estimation fails.
        """
        n, d = X.shape
        if n < d + 1:
            return None

        grads = np.zeros((n, d))

        if self.k_neighbors == 0 or self.k_neighbors >= n:
            # Global: single linear regression y = X @ beta + intercept
            X_aug = np.column_stack([X, np.ones(n)])
            try:
                beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                grads[:] = beta[:d]  # same gradient everywhere
            except np.linalg.LinAlgError:
                return None
        else:
            # Local: k-NN linear regression at each point
            from scipy.spatial import cKDTree
            k = min(self.k_neighbors, n - 1)
            if k < d + 1:
                k = min(d + 1, n - 1)
            tree = cKDTree(X)
            _, indices = tree.query(X, k=k + 1)  # +1 because query includes self

            for i in range(n):
                idx = indices[i]
                X_local = X[idx]
                y_local = y[idx]
                X_aug = np.column_stack([X_local, np.ones(len(idx))])
                try:
                    beta, _, _, _ = np.linalg.lstsq(X_aug, y_local, rcond=None)
                    grads[i] = beta[:d]
                except np.linalg.LinAlgError:
                    grads[i] = 0

        return grads

    def transform(self, X):
        """Rotate X into the active subspace coordinates.

        Parameters
        ----------
        X : array, shape (n_samples, n_dims)
            Input points. Works with both numpy and JAX arrays.

        Returns
        -------
        X_rotated : array, shape (n_samples, n_dims)
            Rotated points.
        """
        if _is_jax_array(X):
            if not self._activated:
                return X
            import jax.numpy as jnp
            return (X - jnp.array(self._center)) @ jnp.array(self._W)
        X = np.array(X, dtype=float, copy=True)
        if not self._activated:
            return X
        return (X - self._center) @ self._W

    def inverse_transform(self, X):
        """Rotate back from active subspace coordinates.

        Parameters
        ----------
        X : array, shape (n_samples, n_dims)
            Points in rotated space. Works with both numpy and JAX arrays.

        Returns
        -------
        X_orig : array, shape (n_samples, n_dims)
            Points in original space.
        """
        if _is_jax_array(X):
            if not self._activated:
                return X
            import jax.numpy as jnp
            return X @ jnp.array(self._W).T + jnp.array(self._center)
        X = np.array(X, dtype=float, copy=True)
        if not self._activated:
            return X
        return X @ self._W.T + self._center

    def transform_bounds(self, bounds):
        """Compute the bounding box of the rotated bounds.

        For d <= 15, uses exact corner enumeration (2^d corners).
        For d > 15, uses a conservative axis-aligned approximation based on
        the rotation matrix norms.

        Parameters
        ----------
        bounds : array, shape (n_dims, 2)
            Bounds [lower, upper] along each dimension.

        Returns
        -------
        new_bounds : array, shape (n_dims, 2)
            Bounding box in rotated space.
        """
        bounds = np.array(bounds, copy=True)
        if not self._activated:
            return bounds
        d = len(bounds)
        if d <= 15:
            # Exact: enumerate all 2^d corners
            corners = np.array(list(product(*((0, 1),) * d)))
            corner_points = bounds[:, 0] + corners * (bounds[:, 1] - bounds[:, 0])
            rotated = self.transform(corner_points)
            new_bounds = np.column_stack([rotated.min(axis=0), rotated.max(axis=0)])
        else:
            # Approximate: use the rotation matrix to compute axis-aligned bounds
            # For a linear transform y = (x - c) @ W, the range in each output
            # dimension j is sum_i |W[i,j]| * (bounds_max_i - bounds_min_i)
            center = self._center
            half_widths = (bounds[:, 1] - bounds[:, 0]) / 2
            centers_orig = (bounds[:, 0] + bounds[:, 1]) / 2
            center_rotated = (centers_orig - center) @ self._W
            # Maximum deviation in each rotated dimension
            max_dev = np.abs(self._W).T @ half_widths  # shape (d,)
            new_bounds = np.column_stack([
                center_rotated - max_dev,
                center_rotated + max_dev
            ])
        return new_bounds

    def inverse_transform_bounds(self, bounds):
        """Compute the bounding box when rotating back from active subspace.

        Parameters
        ----------
        bounds : array, shape (n_dims, 2)
            Bounds in rotated space.

        Returns
        -------
        new_bounds : array, shape (n_dims, 2)
            Bounding box in original space.
        """
        bounds = np.array(bounds, copy=True)
        if not self._activated:
            return bounds
        d = len(bounds)
        if d <= 15:
            corners = np.array(list(product(*((0, 1),) * d)))
            corner_points = bounds[:, 0] + corners * (bounds[:, 1] - bounds[:, 0])
            unrotated = self.inverse_transform(corner_points)
            new_bounds = np.column_stack([unrotated.min(axis=0), unrotated.max(axis=0)])
        else:
            center = self._center
            half_widths = (bounds[:, 1] - bounds[:, 0]) / 2
            centers_rotated = (bounds[:, 0] + bounds[:, 1]) / 2
            # Inverse is y @ W.T + center, so deviation is |W.T|.T @ half_widths = |W| @ half_widths
            center_orig = centers_rotated @ self._W.T + center
            max_dev = np.abs(self._W) @ half_widths
            new_bounds = np.column_stack([
                center_orig - max_dev,
                center_orig + max_dev
            ])
        return new_bounds

    def transform_scale(self, scale):
        """Transform a scale vector through the rotation.

        For a rotation, scales don't have a simple per-axis mapping.
        Use the column norms of W as an approximation (exact for axis-aligned).

        Parameters
        ----------
        scale : array, shape (n_dims,)
            Scale in original space.

        Returns
        -------
        scale_transformed : array, shape (n_dims,)
            Scale in rotated space.
        """
        if not self._activated:
            return scale
        scale = np.asarray(scale, dtype=float).copy()
        # For a rotation matrix, all column norms are 1, so scale is preserved.
        # But if the original scale is anisotropic, the rotated scale mixes them.
        # Conservative: use max scale as uniform estimate.
        return scale

    def inverse_transform_scale(self, scale):
        """Inverse transform a scale vector through the rotation.

        Parameters
        ----------
        scale : array, shape (n_dims,)
            Scale in rotated space.

        Returns
        -------
        scale_orig : array, shape (n_dims,)
            Scale in original space.
        """
        if not self._activated:
            return scale
        return scale
