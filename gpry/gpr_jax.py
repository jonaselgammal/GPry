"""Concrete JAX Gaussian-process backend."""

import warnings

import numpy as np
import scipy.optimize  # type: ignore
from scipy.linalg import cholesky, solve_triangular, cho_solve  # type: ignore
from scipy.linalg.blas import dtrmm as tri_mul  # type: ignore
from sklearn.base import clone  # type: ignore
from sklearn.utils.optimize import _check_optimize_result  # type: ignore
from typing import Mapping
try:
    from sklearn.utils.validation import validate_data  # type: ignore
except ImportError:
    def validate_data(estimator, *args, **kwargs):  # type: ignore
        return estimator._validate_data(*args, **kwargs)

from gpry import gpr_optim
from gpry.array_api import ArrayContract, to_jax
from gpry.gpr_base import BaseGaussianProcessRegressor, EPS_SQ_NOISE, GPR_CHOLESKY_LOWER
from gpry.gpr_jax_linalg import JaxGaussianProcessMixin
from gpry.tools import check_random_state
from gpry.ns_interfaces import NativeNestedSamplingLogLikelihood


def _numpy_log_marginal_likelihood(gpr, theta, eval_gradient=False, clone_kernel=True):
    """Numpy LML used as a fallback before the JAX value-and-grad fn is built.

    Mirrors the formula in ``SklearnGaussianProcessRegressor.log_marginal_likelihood``;
    duplicated here to avoid a cross-backend method call.
    """
    if clone_kernel:
        kernel = gpr.fitted_kernel.clone_with_theta(theta)
    else:
        kernel = gpr.fitted_kernel
        kernel.theta = theta
    if eval_gradient:
        K, K_gradient = kernel(gpr.X_train_, eval_gradient=True)
    else:
        K = kernel(gpr.X_train_)
    K[np.diag_indices_from(K)] += gpr.alpha
    try:
        L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
    except np.linalg.LinAlgError:
        return (-np.inf, np.zeros_like(theta)) if eval_gradient else -np.inf
    y_train = gpr.y_train_
    if y_train.ndim == 1:
        y_train = y_train[:, np.newaxis]
    alpha_vec = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)
    lml_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha_vec)
    lml_dims -= np.log(np.diag(L)).sum()
    lml_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
    lml = lml_dims.sum(axis=-1)
    if eval_gradient:
        inner_term = np.einsum("ik,jk->ijk", alpha_vec, alpha_vec)
        K_inv = cho_solve(
            (L, GPR_CHOLESKY_LOWER), np.eye(K.shape[0]), check_finite=False
        )
        inner_term -= K_inv[..., np.newaxis]
        grad_dims = 0.5 * np.einsum("ijl,jik->kl", inner_term, K_gradient)
        return lml, grad_dims.sum(axis=-1)
    return lml


class JaxGaussianProcessRegressor(BaseGaussianProcessRegressor, JaxGaussianProcessMixin):
    """JAX-backed GP implementation with runtime state stored directly on self."""

    def __init__(self, *args, **kwargs):
        BaseGaussianProcessRegressor.__init__(self, *args, **kwargs)
        JaxGaussianProcessMixin.__init__(self)
        self.use_jax = True
        self._native_acceleration_enabled = True

    @property
    def array_contract(self):
        return ArrayContract(
            accepted_inputs=frozenset({"numpy", "jax"}),
            preferred_input="jax",
            output_kind="jax",
        )

    @property
    def native_backend_ready(self):
        return self._native_acceleration_enabled and self.ready

    def disable_native_acceleration(self):
        self._native_acceleration_enabled = False

    def prime_hyperparameter_optimization(self):
        alpha_val = float(np.atleast_1d(self.alpha).item()) \
            if np.ndim(self.alpha) == 0 else self.alpha
        self.prime_fit_inputs(self.X_train_, self.y_train_, alpha_val)

    def fit_precompute_native(self):
        if not self.native_backend_ready:
            return None
        kernel = self.fitted_kernel
        wn = 0.0
        if hasattr(kernel, 'k1') and hasattr(kernel.k1, 'k1'):
            product_kernel = kernel.k1
            if hasattr(kernel.k2, 'noise_level'):
                wn = float(kernel.k2.noise_level)
        else:
            product_kernel = kernel
        osc = product_kernel.k1.constant_value
        ls = np.atleast_1d(product_kernel.k2.length_scale)
        alpha_val = (float(np.atleast_1d(self.alpha).item())
                     if np.ndim(self.alpha) == 0 else self.alpha)
        self.update_params(ls, osc, wn, alpha_val, X_train=self.X_train_, y_train=self.y_train_)
        return self.fit_precompute()

    def refresh_native_state(self):
        if self._native_acceleration_enabled:
            self.update_from_gpr(self)

    def log_marginal_likelihood_with_grad_native(self, theta):
        return self.log_marginal_likelihood_with_grad(theta)

    def optimize_hyperparameters_native(self, hyperparameter_bounds, selected_starts):
        return self.optimize_hyperparameters(
            bounds=hyperparameter_bounds,
            theta_candidates=selected_starts,
            rng=self._rng,
        )

    @property
    def supports_native_acquisition_optimization(self):
        return self.native_backend_ready and self.ready_for_acquisition_optimization

    def optimize_acquisition_native(self, initial_X, bounds, zeta, noise_var, baseline):
        return self.optimize_acq(initial_X, bounds, zeta, noise_var, baseline)

    def make_ns_loglikelihood_adapter(
        self, numpy_loglikelihood, preprocessing_y, clip_factor, y_clip_min, y_clip_max
    ):
        if not self.native_backend_ready:
            return None
        return NativeNestedSamplingLogLikelihood(
            numpy_loglikelihood=numpy_loglikelihood,
            jax_builder=self.build_surrogate_loglikelihood_builder(
                preprocessing_y=preprocessing_y,
                clip_factor=clip_factor,
                y_clip_min=y_clip_min,
                y_clip_max=y_clip_max,
            ),
        )

    def predict_native(self, X, return_std=False):
        # When the JAX acceleration is disabled (e.g. on KB-conditioned
        # surrogate deepcopies in NORA / BatchOptimizer) skip the padded JAX
        # path and use a numpy predict instead. The JAX path pays a
        # per-call dispatch overhead (~5-10ms on CPU) that dominates for
        # the single-point predict_std calls in NORA's ranking loop;
        # numpy on small ``n_valid``-sized matrices is much cheaper.
        # Previously this flag only gated ``fit_precompute_native`` etc.,
        # which meant the conditioned-surrogate ranking loop was *still*
        # paying full JAX dispatch overhead per candidate.
        if not self._native_acceleration_enabled:
            return self._predict_numpy(X, return_std=return_std)
        X_native = to_jax(X)
        if return_std:
            return self.predict_mean_std_jax(X_native)
        return self.predict_mean_jax(X_native)

    def predict_std_native(self, X):
        if not self._native_acceleration_enabled:
            return self._predict_std_numpy(X)
        return self.predict_std_jax(to_jax(X))

    def _predict_numpy(self, X, return_std=False):
        """Pure-numpy predict using the post-fit caches. Same algebra as
        sklearn's GaussianProcessRegressor.predict, but using the kernel
        attributes already on this object."""
        from scipy.linalg.blas import dtrmm as tri_mul
        K_trans = self.fitted_kernel(X, self.X_train_)
        y_mean = K_trans @ np.asarray(self.alpha_)
        if not return_std:
            return y_mean
        V_np = np.asarray(self.V_)
        M = tri_mul(1.0, V_np, K_trans.T, lower=True)
        y_var = self.fitted_kernel.diag(X).copy()
        y_var -= np.einsum("ji,ji->i", M, M, optimize=True)
        y_var = np.maximum(y_var, 0.0)
        return y_mean, np.sqrt(y_var)

    def _predict_std_numpy(self, X):
        return self._predict_numpy(X, return_std=True)[1]

    def fit(self, X, y, noise_level=None, fit_hyperparameters=True, validate=True):
        if validate:
            kernel_for_validation = self.fitted_kernel if self.fitted_kernel is not None else self.kernel
            if kernel_for_validation is None or kernel_for_validation.requires_vector_input:
                dtype, ensure_2d = "numeric", True
            else:
                dtype, ensure_2d = None, False
            X, y = validate_data(
                self,
                X,
                y,
                multi_output=True,
                y_numeric=True,
                ensure_2d=ensure_2d,
                dtype=dtype,
            )
        if (X is None and y is not None) or (X is not None and y is None):
            raise ValueError("Pass neither or both of X, y, but not just one of them.")
        if X is None and noise_level is None and fit_hyperparameters is None:
            return self
        if X is not None:
            self.X_train_ = np.copy(X) if self.copy_X_train else X
            self.y_train_ = np.copy(y) if self.copy_X_train else y
        if noise_level is not None:
            if validate:
                if (
                    np.iterable(noise_level)
                    and len(noise_level.shape[0]) != self.y_train_.shape[0]
                ):
                    if noise_level.shape[0] == 1:
                        noise_level = noise_level[0]
                    else:
                        raise ValueError(
                            "noise_level must be a scalar or an array with same number of"
                            f" entries as y. ({noise_level.shape[0]} != "
                            f"{self.y_train_.shape[0]})"
                        )
            if hasattr(noise_level, "__len__") and self.is_noise_in_kernel:
                raise TypeError(
                    "This GPR was initialized with a white noise kernel term. That is "
                    "incompatible with passing noise level per training point."
                )
            elif self.is_noise_in_kernel:
                self.alpha = EPS_SQ_NOISE
            else:
                self.alpha = np.maximum(np.array(noise_level) ** 2, EPS_SQ_NOISE)
        if fit_hyperparameters is not False:
            if self.is_noise_in_kernel:
                k = self.kernel if self.fitted_kernel is None else self.fitted_kernel
                k.k2.noise_level_bounds = (
                    min(noise_level**2, EPS_SQ_NOISE * 0.99),
                    noise_level**2,
                )
                k.noise_level = min(k.k2.noise_level, noise_level**2)
            if fit_hyperparameters is True:
                fit_hyperparameters = {}
            elif not isinstance(fit_hyperparameters, Mapping):
                raise TypeError(
                    "'fit_hyperparameters' kwarg must be bool|dict, but was "
                    f"{fit_hyperparameters}"
                )
            self.prime_hyperparameter_optimization()
            self.log_marginal_likelihood_value_ = self._fit_hyperparameters(
                **fit_hyperparameters
            )
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.fitted_kernel.theta, clone_kernel=False
            )
        native_fit = self.fit_precompute_native()
        if native_fit is not None:
            try:
                L_native, V_native, alpha_native = native_fit
                self.L_ = np.asarray(L_native)
                self.V_ = np.asarray(V_native)
                self.alpha_ = np.asarray(alpha_native)
                return self
            except Exception:
                pass
        K = self.fitted_kernel(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
            self.V_ = solve_triangular(
                self.L_, np.eye(self.L_.shape[0]), lower=True
            )
        except np.linalg.LinAlgError as exc:
            exc.args = (
                (
                    f"The kernel, {self.fitted_kernel}, is not returning a "
                    "positive definite matrix. Try gradually increasing "
                    "the 'alpha' parameter of your "
                    "GaussianProcessRegressor estimator."
                ),
            ) + exc.args
            raise
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train_,
            check_finite=False,
        )
        self.refresh_native_state()
        return self

    def log_marginal_likelihood(
        self, theta=None, eval_gradient=False, clone_kernel=True
    ):
        """JAX-native log-marginal likelihood.

        Returns the cached value if ``theta is None``. Otherwise uses the
        compiled value-and-grad function when available; falls back to a numpy
        implementation when the JAX value-and-grad function has not been
        built yet (e.g. during ``fit(..., fit_hyperparameters=False)`` before
        ``refresh_native_state`` runs).
        """
        self.n_eval_loglike += 1
        if theta is None:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_
        if not hasattr(self, "X_train_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                "log_marginal_likelihood called before the GPR was fit."
            )
        if self._lml_value_and_grad_fn is None:
            return _numpy_log_marginal_likelihood(
                self, theta, eval_gradient=eval_gradient, clone_kernel=clone_kernel
            )
        # JAX path: compute value and grad via the compiled fn.
        prev_theta = np.array(self.fitted_kernel.theta, copy=True)
        try:
            self.fitted_kernel.theta = theta
            value, grad = self.log_marginal_likelihood_with_grad_native(theta)
        finally:
            if clone_kernel:
                self.fitted_kernel.theta = prev_theta
        if eval_gradient:
            return value, grad
        return value

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        """L-BFGS-B optimizer used during hyperparameter fitting (numpy fallback)."""
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}.")
        return theta_opt, func_min

    def _fit_hyperparameters(
        self,
        start_from_current=True,
        n_restarts=None,
        hyperparameter_bounds=None,
        **kwargs,
    ):
        if not self._fitted:
            start_from_current = False
        if n_restarts is None:
            n_restarts = self.n_hyperopt_restarts
        self.last_hyperopt_requested_restarts = int(n_restarts)
        no_optimizer = self.optimizer is None
        no_hyperparams = self.kernel.n_dims == 0
        no_restarts = n_restarts <= 0
        if no_optimizer or no_hyperparams or no_restarts:
            msg_reasons = []
            if no_optimizer:
                msg_reasons += ["no optimizer has been specified"]
            if no_hyperparams:
                msg_reasons += ["the kernel has no hyperparamenters"]
            if no_restarts:
                msg_reasons += ["the number of optimizer restarts requested is 0."]
            warnings.warn(
                f"Hyper-parameters not (re)fit. Reason(s): {'; '.join(msg_reasons)}."
            )
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.fitted_kernel.theta, clone_kernel=False
            )
            self._update_model()
            return self

        def obj_func(theta, eval_gradient=True):
            if eval_gradient:
                lml, grad = self.log_marginal_likelihood(
                    theta, eval_gradient=True, clone_kernel=False
                )
                return -lml, -grad
            return -self.log_marginal_likelihood(theta, clone_kernel=False)

        if self.fitted_kernel is None:
            self.fitted_kernel = clone(self.kernel)
        if hyperparameter_bounds is None:
            hyperparameter_bounds = self.fitted_kernel.bounds
        if n_restarts - int(start_from_current):
            if not np.isfinite(hyperparameter_bounds).all():
                raise ValueError(
                    "There is at least one optimizer run the requires sampling from the "
                    "hyperparameters' prior, but it has not finite density, because not "
                    "all bounds are finite. You can pass some finite bounds manually "
                    "using ``hyperparameter_bounds``."
                )
        self._rng = check_random_state(self.random_state)

        n_random = max(
            n_restarts - int(start_from_current),
            min(max(4 * max(n_restarts, 1), 8), 24),
        )
        prev_theta = (
            np.array(self.fitted_kernel.theta, copy=True) if start_from_current else None
        )
        theta_candidates = gpr_optim.build_restart_candidates(
            self.fitted_kernel,
            self.X_train_,
            np.asarray(self.y_train_).reshape(-1),
            hyperparameter_bounds=hyperparameter_bounds,
            n_random=n_random,
            rng=self._rng,
            prev_theta=prev_theta,
            start_from_current=start_from_current,
        )
        n_select = max(n_restarts, int(start_from_current)) + 2
        selected_starts = gpr_optim.score_and_filter_candidates(
            theta_candidates,
            lambda theta: obj_func(theta, eval_gradient=False),
            hyperparameter_bounds=hyperparameter_bounds,
            n_select=n_select,
        )
        self.last_hyperopt_num_starts = len(selected_starts)

        if self._lml_value_and_grad_fn is not None:
            try:
                theta_opt, neg_lml = self.optimize_hyperparameters_native(
                    hyperparameter_bounds, selected_starts
                )
                if gpr_optim.is_pathological_optimum(theta_opt, hyperparameter_bounds):
                    raise RuntimeError("Selected JAX hyperopt optimum is pathological.")
                self.fitted_kernel.theta = theta_opt
                self._fitted = True
                self.L_, self.V_, self.alpha_ = None, None, None
                return -neg_lml
            except Exception:
                pass

        optima = []
        for theta_initial in selected_starts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                optima.append(
                    self._constrained_optimization(
                        obj_func, theta_initial, hyperparameter_bounds
                    )
                )
        valid_optima = [
            (theta, value)
            for theta, value in optima
            if not gpr_optim.is_pathological_optimum(theta, hyperparameter_bounds)
        ]
        selected_optima = valid_optima if valid_optima else optima
        selected_idx = min(
            range(len(selected_optima)),
            key=lambda idx: (
                selected_optima[idx][1],
                gpr_optim.boundary_penalty(
                    selected_optima[idx][0], hyperparameter_bounds
                ),
            ),
        )
        self.fitted_kernel.theta = selected_optima[selected_idx][0]
        self._fitted = True
        self.L_, self.V_, self.alpha_ = None, None, None
        return -selected_optima[selected_idx][1]

    def predict(
        self,
        X,
        return_std=False,
        return_mean_grad=False,
        return_std_grad=False,
        validate=True,
    ):
        if return_std_grad and not (return_std and return_mean_grad):
            raise ValueError(
                "Not returning std_gradient without returning "
                "the std and the mean grad."
            )
        if X.shape[0] != 1 and (return_mean_grad or return_std_grad):
            raise ValueError(
                "Mean grad and std grad not implemented \
                for n_samples > 1"
            )
        if hasattr(self, "X_train_"):
            self.n_eval += len(X)
            if return_mean_grad:
                x = X[0]
                mean, std, grad_mean, grad_std = self.predict_with_grads(x)
                return_values = [np.array([mean])]
                if return_std:
                    return_values.append(np.array([std]))
                return_values.append(grad_mean)
                if return_std_grad:
                    return_values.append(grad_std)
                return return_values
            if return_std:
                y_mean_j, y_std_j = self.predict_native(X, return_std=True)
                return [np.asarray(y_mean_j), np.asarray(y_std_j)]
            y_mean_j = self.predict_native(X, return_std=False)
            return [np.asarray(y_mean_j)]
        # Not fit yet: predict based on GP prior.
        if validate:
            if self.kernel is None or self.kernel.requires_vector_input:
                dtype, ensure_2d = "numeric", True
            else:
                dtype, ensure_2d = None, False
            X = validate_data(self, X, ensure_2d=ensure_2d, dtype=dtype, reset=False)
        y_mean = np.zeros(X.shape[0])
        if return_std:
            y_var = self.kernel.diag(X)
            y_std = np.sqrt(y_var)
            if not return_mean_grad and not return_std_grad:
                return y_mean, y_std
        if return_mean_grad:
            mean_grad = np.zeros_like(X)
            if return_std:
                if return_std_grad:
                    std_grad = np.zeros_like(X)
                    return y_mean, y_std, mean_grad, std_grad
                return y_mean, y_std, mean_grad
            return y_mean, mean_grad
        return y_mean

    def predict_std(self, X, validate=True):
        if hasattr(self, "X_train_"):
            self.n_eval += len(X)
            return np.asarray(self.predict_std_native(X))
        # Not fit yet: predict_std based on GP prior.
        if validate:
            if self.kernel is None or self.kernel.requires_vector_input:
                dtype, ensure_2d = "numeric", True
            else:
                dtype, ensure_2d = None, False
            X = validate_data(self, X, ensure_2d=ensure_2d, dtype=dtype, reset=False)
        self.n_eval += len(X)
        return np.sqrt(self.kernel.diag(X))
