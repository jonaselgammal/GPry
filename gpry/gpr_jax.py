"""Concrete JAX Gaussian-process backend."""

import numpy as np

from gpry.array_api import ArrayContract, to_jax
from gpry.gpr_base import BaseGaussianProcessRegressor
from gpry.jax_accel import JaxRuntimeBundle


class JaxGaussianProcessRegressor(BaseGaussianProcessRegressor, JaxRuntimeBundle):
    """JAX-backed GP implementation with runtime state stored directly on self."""

    def __init__(self, *args, **kwargs):
        BaseGaussianProcessRegressor.__init__(self, *args, **kwargs)
        JaxRuntimeBundle.__init__(self)
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
        kernel = self.kernel_
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

    def make_ns_loglikelihood_builder(
        self, preprocessing_y, clip_factor, y_clip_min, y_clip_max
    ):
        if not self.native_backend_ready:
            return None
        return self.build_surrogate_loglikelihood_builder(
            preprocessing_y=preprocessing_y,
            clip_factor=clip_factor,
            y_clip_min=y_clip_min,
            y_clip_max=y_clip_max,
        )

    def predict_native(self, X, return_std=False):
        X_native = to_jax(X)
        if return_std:
            return self.predict_mean_std_jax(X_native)
        return self.predict_mean_jax(X_native)

    def predict_std_native(self, X):
        return self.predict_std_jax(to_jax(X))
