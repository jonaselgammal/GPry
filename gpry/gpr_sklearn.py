"""Concrete numpy/sklearn Gaussian-process backend."""

import warnings
from typing import Mapping

import numpy as np
import scipy.optimize  # type: ignore
from scipy.linalg import cholesky, solve_triangular, cho_solve  # type: ignore
from scipy.linalg.blas import dtrmm as tri_mul  # type: ignore
from sklearn.base import clone  # type: ignore
from sklearn.utils.optimize import _check_optimize_result  # type: ignore
try:
    from sklearn.utils.validation import validate_data  # type: ignore
except ImportError:
    def validate_data(estimator, *args, **kwargs):  # type: ignore
        return estimator._validate_data(*args, **kwargs)

from gpry import gpr_optim
from gpry.gpr_base import (
    BaseGaussianProcessRegressor,
    EPS_SQ_NOISE,
    GPR_CHOLESKY_LOWER,
)
from gpry.tools import check_random_state


class SklearnGaussianProcessRegressor(BaseGaussianProcessRegressor):
    """Default sklearn/scipy-backed GP implementation.

    Owns the numpy numerics (``fit``, ``predict``, ``predict_std``,
    ``log_marginal_likelihood``, ``_fit_hyperparameters``,
    ``_constrained_optimization``) that used to live on the shared base.
    """

    def fit(self, X, y, noise_level=None, fit_hyperparameters=True, validate=True):
        r"""
        Re-implementation of the sk GPR fit method, that allows for updating the noise
        level (as alpha), and exposes flags for input validation and hyperparameter
        fitting.

        If hyperparameters are kept constant, fitting here refers to the re-calculation of
        the GPR inverse matrix :math:`(K(X,X)+\sigma_n^2 I)^{-1}` which is needed for
        predictions.

        The highest cost incurred by this method is the refitting of the GPR kernel
        hyperparameters :math:`\theta`. It can be useful to disable it
        (``fit_hyperparameters=False``) in cases where it is worth saving the
        computational expense in exchange for a loss of information, such as when
        performing parallelized active sampling (NB: this is only possible when the GPR
        hyperparameters have been fit at least once).

        If called with ``X=None, y=None``, it re-fits the model without adding new points.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features), or None
            Training data to append to the model.

        y : array-like, shape = (n_samples, [n_output_dims]), or None
            Target values to append to the data

        noise_level : number, array-like, shape = (n_samples, [n_output_dims])
            Uncorrelated standard deviation(s) to add to the diagonal part of the
            covariance matrix.

        fit_hyperparameters : Bool or dict (default: True)
            Whether the GPR :math:`\theta`-parameters should be optimised.

        validate : bool, default: True
            If False, ``X`` and ``y`` are assumed to be correctly formatted, and no
            checks are performed on them. Reduces overhead.

        Returns
        -------
        self : object
            GaussianProcessRegressor class instance.
        """
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
        # If neither new points nor new noise nor hyperams fit, return
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
                # The following line causes sometimes noise larger than the one passed.
                self.alpha = np.maximum(np.array(noise_level) ** 2, EPS_SQ_NOISE)
        if fit_hyperparameters is not False:
            if self.is_noise_in_kernel:
                # Used passed noise as an upper bound
                k = self.kernel if self.fitted_kernel is None else self.fitted_kernel
                k.k2.noise_level_bounds = (
                    min(noise_level**2, EPS_SQ_NOISE * 0.99),
                    noise_level**2,
                )
                # Also lower the current noise if needed, in case it's over the bound
                # after preprocessor is refit
                k.noise_level = min(k.k2.noise_level, noise_level**2)
            if fit_hyperparameters is True:
                fit_hyperparameters = {}
            elif not isinstance(fit_hyperparameters, Mapping):
                raise TypeError(
                    "'fit_hyperparameters' kwarg must be bool|dict, but was "
                    f"{fit_hyperparameters}"
                )
            self.log_marginal_likelihood_value_ = self._fit_hyperparameters(
                **fit_hyperparameters
            )
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.fitted_kernel.theta, clone_kernel=False
            )
        # Precompute quantities required for predictions which are independent
        # of actual query points
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K = self.fitted_kernel(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
            self.V_ = solve_triangular(
                self.L_, np.eye(self.L_.shape[0]), lower=True)
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
        return self

    def log_marginal_likelihood(
        self, theta=None, eval_gradient=False, clone_kernel=True
    ):
        """
        Log-marginal likelihood of the kernel hyperparameters given the training data.

        Numpy implementation. Returns the same value (and gradient, if requested)
        as the sklearn implementation it replaces.
        """
        self.n_eval_loglike += 1
        if theta is None:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        if clone_kernel:
            kernel = self.fitted_kernel.clone_with_theta(theta)
        else:
            kernel = self.fitted_kernel
            kernel.theta = theta

        if eval_gradient:
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)

        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        # alpha = L^T \ (L \ y)
        alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)

        # -0.5 . y^T . alpha - sum(log(diag(L))) - n_samples / 2 log(2*pi)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(axis=-1)

        if eval_gradient:
            inner_term = np.einsum("ik,jk->ijk", alpha, alpha)
            K_inv = cho_solve(
                (L, GPR_CHOLESKY_LOWER), np.eye(K.shape[0]), check_finite=False
            )
            inner_term -= K_inv[..., np.newaxis]
            log_likelihood_gradient_dims = 0.5 * np.einsum(
                "ijl,jik->kl", inner_term, K_gradient
            )
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(axis=-1)
            return log_likelihood, log_likelihood_gradient
        return log_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        """L-BFGS-B optimizer used during hyperparameter fitting.

        Inlined from sklearn's GaussianProcessRegressor (was inherited
        previously).
        """
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
        r"""Optimizes the hyperparameters :math:`\theta` for the current training data.

        NB: This function does *NOT* update the precomputed kernel matrices. Do not call
        outside self.fit
        """
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
        # Choose hyperparameters based on maximizing the log-marginal
        # likelihood (potentially starting from several initial values)
        # We don't need to clone the kernel here, even if overwritten during optimization,
        # because it will be recomputed in the final `log_marginal_likelihood` call.

        def obj_func(theta, eval_gradient=True):
            if eval_gradient:
                lml, grad = self.log_marginal_likelihood(
                    theta, eval_gradient=True, clone_kernel=False
                )
                return -lml, -grad
            else:
                return -self.log_marginal_likelihood(theta, clone_kernel=False)

        if self.fitted_kernel is None:
            self.fitted_kernel = clone(self.kernel)
        if hyperparameter_bounds is None:
            hyperparameter_bounds = self.fitted_kernel.bounds
        else:
            # TODO: validate dimensions!
            pass
        # If at least one run will be sampled from the prior, is has to be finite
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
        # The legacy code selected up to `max(n_restarts, int(start_from_current)) + 2`
        # distinct starts after scoring; preserve that.
        n_select = max(n_restarts, int(start_from_current)) + 2
        selected_starts = gpr_optim.score_and_filter_candidates(
            theta_candidates,
            lambda theta: obj_func(theta, eval_gradient=False),
            hyperparameter_bounds=hyperparameter_bounds,
            n_select=n_select,
        )
        self.last_hyperopt_num_starts = len(selected_starts)

        optima = []
        for theta_initial in selected_starts:
            # Run the optimizer!
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                optima.append(
                    self._constrained_optimization(
                        obj_func, theta_initial, hyperparameter_bounds
                    )
                )
        # Select result from run with minimal (negative) log-marginal likelihood.
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
        # Reset pre-computed matrices
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
        """
        Predict output for X.

        Returns the GP-prior prediction if not yet fit; otherwise the GP-posterior.
        """
        self.n_eval += len(X)
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
        if validate:
            if self.kernel is None or self.kernel.requires_vector_input:
                dtype, ensure_2d = "numeric", True
            else:
                dtype, ensure_2d = None, False
            X = validate_data(self, X, ensure_2d=ensure_2d, dtype=dtype, reset=False)
        # If not fit yet, predict based on GP prior
        if not hasattr(self, "X_train_"):
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
                    else:
                        return y_mean, y_std, mean_grad
                else:
                    return y_mean, mean_grad
            else:
                return y_mean
        # If already fit, use GP posterior to predict
        K_trans = self.fitted_kernel(X, self.X_train_)
        y_mean = K_trans.dot(self.alpha_)
        return_values = [y_mean]
        if return_std:
            M = tri_mul(1.0, self.V_, K_trans.T, lower=True)
            y_var = self.fitted_kernel.diag(X).copy()
            y_var -= np.einsum("ji,ji->i", M, M, optimize=True)
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn(
                    "Predicted variances smaller than 0. Setting those variances to 0."
                )
                y_var[y_var_negative] = 0.0
            y_std = np.sqrt(y_var)
            return_values.append(y_std)
        if return_mean_grad:
            grad = self.fitted_kernel.gradient_x(X[0], self.X_train_)
            grad_mean = np.dot(grad.T, self.alpha_)
            return_values.append(grad_mean)
            if return_std_grad:
                if not np.any(y_std):  # do not compute if all stds null
                    grad_std = np.zeros(X.shape[1])
                else:
                    grad_std = (
                        -np.dot(K_trans, np.dot(self.V_.T.dot(self.V_), grad))[0]
                        / y_std
                    )
                return_values.append(grad_std)
        return return_values

    def predict_std(self, X, validate=True):
        """
        Predict output standart deviation for X.
        """
        self.n_eval += len(X)
        if validate:
            if self.kernel is None or self.kernel.requires_vector_input:
                dtype, ensure_2d = "numeric", True
            else:
                dtype, ensure_2d = None, False
            X = validate_data(self, X, ensure_2d=ensure_2d, dtype=dtype, reset=False)
        # If not fit yet, predict based on GP prior
        if not hasattr(self, "X_train_"):
            return np.sqrt(self.kernel.diag(X))
        # If already fit, use GP posterior to predict
        K_trans = self.fitted_kernel(X, self.X_train_)
        M = tri_mul(1.0, self.V_, K_trans.T, lower=True)
        y_var = self.fitted_kernel.diag(X).copy()
        y_var -= np.einsum("ji,ji->i", M, M, optimize=True)
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn(
                "Predicted variances smaller than 0. Setting those variances to 0."
            )
            y_var[y_var_negative] = 0.0
        return np.sqrt(y_var)
