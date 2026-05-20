"""Abstract Gaussian-process regressor base for the sklearn and JAX backends.

Architecture (post-refactor)
============================

``BaseGaussianProcessRegressor`` is a pure ``ABC`` — no sklearn parent, no
numerical body. Two concrete subclasses implement the same abstract surface
with different numerics:

- :class:`gpry.gpr_sklearn.SklearnGaussianProcessRegressor` — numpy/scipy
  backend. Owns ``fit``, ``predict``, ``predict_std``, ``_fit_hyperparameters``,
  ``log_marginal_likelihood``. No sklearn-GPR composition; uses scipy linalg
  directly + sklearn-wrapped kernel objects (see :mod:`gpry.kernels_sklearn`).

- :class:`gpry.gpr_jax.JaxGaussianProcessRegressor` — JAX-jitted backend
  (mixed with :class:`gpry.gpr_jax_linalg.JaxGaussianProcessMixin` for the
  cached predictive-state machinery). Owns its own JAX-native ``predict``,
  ``log_marginal_likelihood``, and native acquisition optimizer.

Shared logic lives in dedicated module(s):

- :mod:`gpry.gpr_optim` — backend-agnostic hyperparameter restart-candidate
  generation, scoring, and pathological-optimum filtering.

The two backends were factored apart in Phases 1–4 of the JAX-split refactor;
see ``AGENTS/jax_split_refactor_plan.md`` for the design history.

Public surface (used outside ``gpr_*.py``)
==========================================

This table pins the contract that any future refactor must preserve.

================================================  =============================
Method / property                                 External callers
================================================  =============================
``fit(X, y, noise_level=, fit_hyperparameters=,   ``surrogate.py``
validate=)``
``predict(X, return_std=, return_mean_grad=,      ``surrogate.py``
return_std_grad=, validate=)``
``predict_std(X, ...)``                           ``surrogate.py``
``predict_native(X, ...)``                        ``surrogate.py`` hot path
``predict_std_native(X, ...)``                    ``surrogate.py`` hot path
``log_marginal_likelihood(theta=,                 ``surrogate.py``
eval_gradient=)``
``noise_level``                                   ``surrogate.py``
``fitted_kernel``                                 ``run.py`` (logging only)
``n_hyperopt_restarts``                           ``run.py`` (logging only)
``native_backend_ready``                          ``surrogate.py``
``use_jax``                                       ``surrogate.py``
``optimize_acquisition_native``                   ``gp_acquisition.py``
``optimize_hyperparameters_native``               internal to ``gpr_jax.py``
``log_marginal_likelihood_with_grad_native``      internal to ``gpr_jax.py``
``prime_hyperparameter_optimization``             ``surrogate.py``
``disable_native_acceleration``                   tests, debugging
================================================  =============================

**State attributes that callers must not reach into.** ``X_train_``,
``y_train_``, ``alpha``, ``alpha_``, ``L_``, ``V_``, ``noise_level_``,
``log_marginal_likelihood_value_``. These remain as **internal** storage on
the backend instances (the names still mirror sklearn's GPR conventions
because the kernel-handling code still wraps sklearn kernels for parameter
bookkeeping); they are not part of the public API. Surrogate-level accessors
(``surrogate.X_regress``, ``surrogate.y_regress``, ``surrogate.n_regress``,
``surrogate.noise_level``, etc.) cover everything the rest of the codebase
needs.

Construction (factory dispatch)
===============================

Calling ``GaussianProcessRegressor(use_jax=True, …)`` (the public name
re-exported from :mod:`gpry.gpr`) returns a :class:`JaxGaussianProcessRegressor`
when JAX is importable, otherwise falls back to
:class:`SklearnGaussianProcessRegressor`. The dispatch lives in ``__new__``
below; subclasses are constructed directly when their class is named.
"""

# Builtin
from abc import ABC, abstractmethod
from typing import Mapping

# External
import numpy as np

# Local
from gpry.array_api import ArrayContract
from gpry.kernels_sklearn import RBF, Matern, WhiteKernel, ConstantKernel as C

GPR_CHOLESKY_LOWER = True
EPS_SQ_NOISE = 1e-6  # diagonal term to be added when WhiteKernel used as noise


class BaseGaussianProcessRegressor(ABC):
    def __new__(cls, *args, **kwargs):
        if cls is BaseGaussianProcessRegressor:
            use_jax = kwargs.get("use_jax", True)
            if use_jax:
                try:
                    from gpry.gpr_jax import JaxGaussianProcessRegressor

                    return super().__new__(JaxGaussianProcessRegressor)
                except ImportError:
                    pass
            from gpry.gpr_sklearn import SklearnGaussianProcessRegressor

            return super().__new__(SklearnGaussianProcessRegressor)
        return super().__new__(cls)

    r"""
    Modified version of the GaussianProcessRegressor of sklearn.

    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.

    This modified interface provides, in addition to the sklearn-GPR:

       * Re-implements the ``fit`` method to allow for more control in the noise (alpha)
         update and the hyperparameter optimization.
       * Implements derivative return values in the ``predict`` method, as well as a
         ``predict_std`` method to return the standard deviation of the target only
         (useful for acquisition).
       * In the relevant methods, exposes flags to disable input data validation, for
         an additional speed boost.

    Parameters
    ----------
    kernel : kernel object, string, dict, optional (default: "RBF")
        The kernel specifying the covariance function of the GP.

    output_scale_prior : tuple as (min, max), optional (default: [1e-2, 1e3])

    length_scale_prior : tuple as (min, max), optional (default: [1e-3, 1e1])

    noise_level : float or array-like, optional (default: 1e-2)

    noise_fixed : bool (default: True)

    optimizer : str or callable, optional (default: "fmin_l_bfgs_b")

    n_hyperopt_restarts : int, optional (default: 0)

    random_state : int or numpy.random.Generator, optional

    Attributes
    ----------
    X_train_ : array-like, shape = (n_samples, n_features)

    y_train_ : array-like, shape = (n_samples, [n_output_dims])

    alpha : array-like, shape = (n_samples, [n_output_dims]) or scalar

    fitted_kernel : :mod:`kernels` object

    alpha_ : array-like, shape = (n_samples, n_samples)

    V_ : array-like, shape = (n_samples, n_samples)

    log_marginal_likelihood_value_ : float

    scales : tuple
    """

    def __init__(
        self,
        kernel="RBF",
        output_scale_prior=[1e-2, 1e3],
        length_scale_prior=[1e-2, 1e2],
        noise_level=1e-2,
        noise_fixed=True,
        optimizer="fmin_l_bfgs_b",
        n_hyperopt_restarts=0,
        random_state=None,
        use_jax=True,
    ):
        self.n_eval = 0
        self.n_eval_loglike = 0
        self._fitted = False
        self.fitted_kernel = None
        self.last_hyperopt_num_starts = None
        self.last_hyperopt_requested_restarts = None
        self.use_jax = bool(use_jax)
        self._runtime_enabled = self.use_jax
        # Auto-construct inbuilt kernels
        if isinstance(kernel, str):
            kernel = {kernel: {}}
        if isinstance(kernel, Mapping):
            if len(kernel) != 1:
                raise ValueError("'kernel' must be a single-key dict.")
            kernel_name = list(kernel)[0]
            kernel_args = kernel[kernel_name] or {}
            try:
                length_corr_kernel = {"rbf": RBF, "matern": Matern}[kernel_name.lower()]
            except KeyError as excpt:
                raise ValueError(
                    "Currently only 'RBF' and 'Matern' are "
                    f"supported as standard kernels. Got '{kernel_name}'."
                ) from excpt
            output_scale_init = np.sqrt(output_scale_prior[0] * output_scale_prior[1])
            length_scale_init = np.sqrt(
                length_scale_prior[:, 0] * length_scale_prior[:, 1]
            )
            self.is_noise_in_kernel = not noise_fixed
            if hasattr(noise_level, "__len__"):
                raise TypeError(
                    "If noise is passed per training point, it needs to be fixed. i.e. "
                    "`noise_fixed=True`."
                )
            kernel_args = dict(kernel_args)
            kernel_args.setdefault("length_scale_bounds", length_scale_prior)
            kernel = C(
                output_scale_init**2,
                [output_scale_prior[0] ** 2, output_scale_prior[1] ** 2],
            ) * length_corr_kernel(
                length_scale_init,
                prior_bounds=length_scale_prior,
                **kernel_args,
            )
            if self.is_noise_in_kernel:
                kernel += WhiteKernel(
                    noise_level=noise_level**2,
                    noise_level_bounds=(EPS_SQ_NOISE, noise_level**2),
                )
        else:
            # Custom kernel object: noise treatment fully determined by the kernel itself.
            self.is_noise_in_kernel = bool(noise_fixed is False)
        # Plain attribute assignments (replaces sklearn's __init__).
        self.kernel = kernel
        self.alpha = noise_level**2 if not self.is_noise_in_kernel else EPS_SQ_NOISE
        self.optimizer = optimizer
        self.n_hyperopt_restarts = n_hyperopt_restarts
        self.normalize_y = False
        self.copy_X_train = True
        self.random_state = random_state

    @property
    def scales(self):
        """
        Kernel scales as ``(output_scale, (length_scale_1, ...))``.
        """
        if self.fitted_kernel is None:  # not fitted yet
            length_kernel = self.kernel
        else:
            length_kernel = self.fitted_kernel
        if hasattr(length_kernel.k1, "k1"):  # there is a noise term
            length_kernel = length_kernel.k1
        return (
            np.sqrt(length_kernel.k1.constant_value),
            np.array(length_kernel.k2.length_scale),
        )

    @property
    def array_contract(self):
        return ArrayContract(
            accepted_inputs=frozenset({"numpy"}),
            preferred_input="numpy",
            output_kind="numpy",
        )

    @property
    def preferred_array_kind(self):
        return self.array_contract.preferred_input

    @property
    def native_backend_ready(self):
        return False

    def disable_native_acceleration(self):
        self._runtime_enabled = False

    def prime_hyperparameter_optimization(self):
        return None

    def fit_precompute_native(self):
        return None

    def refresh_native_state(self):
        return None

    def log_marginal_likelihood_with_grad_native(self, theta):
        raise NotImplementedError

    def optimize_hyperparameters_native(self, hyperparameter_bounds, selected_starts):
        raise NotImplementedError

    @property
    def supports_native_acquisition_optimization(self):
        return False

    def optimize_acquisition_native(self, initial_X, bounds, zeta, noise_var, baseline):
        raise NotImplementedError

    def make_ns_loglikelihood_adapter(
        self, numpy_loglikelihood, preprocessing_y, clip_factor, y_clip_min, y_clip_max
    ):
        return None

    def predict_native(self, X, return_std=False):
        if return_std:
            return self.predict(X, return_std=True, validate=False)
        return self.predict(X, validate=False)

    def predict_std_native(self, X):
        return self.predict_std(X, validate=False)

    @property
    def noise_level(self):
        """
        Kernel noise level (not squared).
        """
        if self.is_noise_in_kernel:
            if self.fitted_kernel is None:  # not fitted yet
                kernel = self.kernel
            else:
                kernel = self.fitted_kernel
            return np.sqrt(kernel.k2.noise_level)
        else:
            return np.sqrt(self.alpha)

    @abstractmethod
    def fit(self, X, y, noise_level=None, fit_hyperparameters=True, validate=True):
        """Fit (or refit) the GP. Implemented by each concrete backend."""

    @abstractmethod
    def predict(
        self,
        X,
        return_std=False,
        return_mean_grad=False,
        return_std_grad=False,
        validate=True,
    ):
        """Predict mean (and optionally std / gradients). Implemented per backend."""

    @abstractmethod
    def predict_std(self, X, validate=True):
        """Predict only the std at X. Implemented per backend."""

    @abstractmethod
    def log_marginal_likelihood(
        self, theta=None, eval_gradient=False, clone_kernel=True
    ):
        """Log-marginal likelihood (and optionally gradient). Implemented per backend."""
