"""Concrete JAX Gaussian-process backend."""

from gpry.array_api import ArrayContract, to_jax
from gpry.gpr_base import BaseGaussianProcessRegressor
from gpry.jax_accel import JaxRuntimeBundle


class JaxGaussianProcessRegressor(BaseGaussianProcessRegressor, JaxRuntimeBundle):
    """JAX-backed GP implementation with runtime state stored directly on self."""

    def __init__(self, *args, **kwargs):
        BaseGaussianProcessRegressor.__init__(self, *args, **kwargs)
        JaxRuntimeBundle.__init__(self)
        self.use_jax = True
        self._runtime_enabled = True

    @property
    def array_contract(self):
        return ArrayContract(
            accepted_inputs=frozenset({"numpy", "jax"}),
            preferred_input="jax",
            output_kind="jax",
        )

    @property
    def runtime_bundle(self):
        if not self._runtime_enabled:
            return None
        return self

    @property
    def supports_native_acquisition_optimization(self):
        return self.runtime_bundle is not None and self.ready_for_acquisition_optimization

    def optimize_acquisition_native(self, initial_X, bounds, zeta, noise_var, baseline):
        return self.optimize_acq(initial_X, bounds, zeta, noise_var, baseline)

    def make_ns_loglikelihood_builder(
        self, preprocessing_y, clip_factor, y_clip_min, y_clip_max
    ):
        if self.runtime_bundle is None or not self.ready:
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
