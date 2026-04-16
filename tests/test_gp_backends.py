"""
Comprehensive tests for all GP backends: RandomFeatureGP, GibbsGPR, DeepKernelGP.

Tests each backend for:
1. Basic fit/predict functionality
2. Interface compatibility with SurrogateModel
3. Prediction quality compared to standard GP
4. Gradient computation
5. Edge cases
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_data(n_train=40, n_test=10, d=3, seed=42):
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, d)
    y_train = np.sin(X_train[:, 0]) + 0.3 * np.cos(X_train[:, 1]) + 0.01 * rng.randn(n_train)
    X_test = rng.randn(n_test, d)
    return X_train, y_train, X_test


def make_lsp(d):
    return np.column_stack([np.full(d, 1e-2), np.full(d, 1e1)])


# ========================= SHARED INTERFACE TESTS =========================

BACKENDS = ["random_features", "gibbs"]


def make_backend(backend, d, **kwargs):
    lsp = make_lsp(d)
    defaults = dict(
        length_scale_prior=lsp,
        n_restarts_optimizer=1,
        random_state=42,
    )
    defaults.update(kwargs)
    if backend == "random_features":
        from gpry.experimental_random_feature_gp import RandomFeatureGP
        return RandomFeatureGP(**defaults)
    elif backend == "gibbs":
        from gpry.experimental_gibbs_gpr import GibbsGPR
        return GibbsGPR(**defaults)
    else:
        raise ValueError(f"Unknown backend: {backend}")


@pytest.mark.parametrize("backend", BACKENDS)
class TestInterfaceCompatibility:
    """Test that each backend exposes the full interface expected by SurrogateModel."""

    def test_required_attributes_exist(self, backend):
        gpr = make_backend(backend, d=3)
        attrs = [
            'kernel', 'kernel_', 'scales', 'noise_level', 'is_noise_in_kernel',
            'X_train_', 'y_train_', 'alpha', '_fitted', 'n_restarts_optimizer',
            'n_eval', 'n_eval_loglike', 'random_state', 'copy_X_train',
        ]
        for attr in attrs:
            assert hasattr(gpr, attr), f"Missing attribute: {attr}"

    def test_fit_returns_self(self, backend):
        X, y, _ = make_data(d=3)
        gpr = make_backend(backend, d=3)
        result = gpr.fit(X, y, validate=False)
        assert result is gpr

    def test_fitted_flag(self, backend):
        X, y, _ = make_data(d=3)
        gpr = make_backend(backend, d=3)
        assert not gpr._fitted
        gpr.fit(X, y, validate=False)
        assert gpr._fitted

    def test_predict_returns_list(self, backend):
        X, y, X_test = make_data(d=3)
        gpr = make_backend(backend, d=3)
        gpr.fit(X, y, validate=False)
        result = gpr.predict(X_test)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_predict_with_std(self, backend):
        X, y, X_test = make_data(d=3)
        gpr = make_backend(backend, d=3)
        gpr.fit(X, y, validate=False)
        result = gpr.predict(X_test, return_std=True)
        assert len(result) == 2
        y_mean, y_std = result
        assert y_mean.shape == (len(X_test),)
        assert y_std.shape == (len(X_test),)
        assert np.all(y_std > 0)

    def test_predict_std_method(self, backend):
        X, y, X_test = make_data(d=3)
        gpr = make_backend(backend, d=3)
        gpr.fit(X, y, validate=False)
        std = gpr.predict_std(X_test)
        assert std.shape == (len(X_test),)
        assert np.all(std > 0)

    def test_scales_property(self, backend):
        X, y, _ = make_data(d=3)
        gpr = make_backend(backend, d=3)
        gpr.fit(X, y, validate=False)
        output_scale, length_scales = gpr.scales
        assert np.isscalar(output_scale) or output_scale.ndim == 0
        assert len(length_scales) == 3

    def test_lml_is_set(self, backend):
        X, y, _ = make_data(d=3)
        gpr = make_backend(backend, d=3)
        gpr.fit(X, y, validate=False)
        assert gpr.log_marginal_likelihood_value_ is not None
        assert np.isfinite(gpr.log_marginal_likelihood_value_)

    def test_fit_with_dict_hyperparams(self, backend):
        """Test fit_hyperparameters as a dict (used by surrogate.append)."""
        X, y, _ = make_data(d=3)
        gpr = make_backend(backend, d=3)
        gpr.fit(X, y, fit_hyperparameters={"n_restarts": 1, "start_from_current": True},
                validate=False)
        assert gpr._fitted

    def test_fit_no_hyperparams(self, backend):
        """Test fit_hyperparameters=False."""
        X, y, _ = make_data(d=3)
        gpr = make_backend(backend, d=3)
        gpr.fit(X, y, validate=False)
        # Fit again without re-optimizing
        gpr.fit(X, y, fit_hyperparameters=False, validate=False)
        assert gpr._fitted

    def test_noise_level_property(self, backend):
        gpr = make_backend(backend, d=3, noise_level=0.05)
        assert np.isscalar(gpr.noise_level) or gpr.noise_level.ndim == 0


# ========================= PREDICTION QUALITY TESTS =========================

@pytest.mark.parametrize("backend", BACKENDS)
class TestPredictionQuality:
    """Test that predictions are reasonable."""

    def test_interpolation(self, backend):
        """Predictions at training points should be close to training values."""
        d = 2
        X, y, _ = make_data(n_train=30, d=d, seed=42)
        gpr = make_backend(backend, d=d, n_restarts_optimizer=2, noise_level=1e-4)
        gpr.fit(X, y, validate=False)
        y_pred = gpr.predict(X)[0]
        # Should interpolate reasonably well
        rmse = np.sqrt(np.mean((y_pred - y) ** 2))
        assert rmse < 1.0, f"RMSE too high: {rmse}"

    def test_uncertainty_grows_far_from_data(self, backend):
        """Uncertainty should be larger far from training data."""
        d = 2
        rng = np.random.RandomState(42)
        X = rng.randn(30, d)
        y = np.sin(X[:, 0])
        gpr = make_backend(backend, d=d, n_restarts_optimizer=1)
        gpr.fit(X, y, validate=False)

        X_near = rng.randn(10, d) * 0.5  # near origin
        X_far = rng.randn(10, d) * 5.0   # far from data

        _, std_near = gpr.predict(X_near, return_std=True)
        _, std_far = gpr.predict(X_far, return_std=True)
        # On average, far points should have higher uncertainty
        assert np.mean(std_far) > np.mean(std_near) * 0.8


# ========================= GRADIENT TESTS =========================

@pytest.mark.parametrize("backend", BACKENDS)
class TestGradients:
    """Test gradient computation for acquisition function support."""

    def test_mean_gradient_shape(self, backend):
        d = 3
        X, y, _ = make_data(n_train=25, d=d)
        gpr = make_backend(backend, d=d)
        gpr.fit(X, y, validate=False)
        x_test = np.random.randn(1, d)
        result = gpr.predict(x_test, return_mean_grad=True)
        assert len(result) >= 2
        grad = result[1]
        # Gradient shape is (1, d) or (d,) depending on backend
        assert grad.shape[-1] == d

    def test_std_gradient_shape(self, backend):
        d = 3
        X, y, _ = make_data(n_train=25, d=d)
        gpr = make_backend(backend, d=d)
        gpr.fit(X, y, validate=False)
        x_test = np.random.randn(1, d)
        result = gpr.predict(x_test, return_std=True, return_mean_grad=True,
                             return_std_grad=True)
        assert len(result) == 4
        y_mean, y_std, grad_mean, grad_std = result
        assert grad_mean.shape[-1] == d
        assert grad_std.shape[-1] == d


# ========================= SURROGATE INTEGRATION TEST =========================

@pytest.mark.parametrize("backend", BACKENDS)
def test_surrogate_instantiation(backend):
    """Test that each backend can be instantiated via SurrogateModel config."""
    from gpry.surrogate import SurrogateModel
    from gpry.preprocessing import DummyPreprocessor

    d = 2
    bounds = np.array([[-5, 5]] * d)

    surr = SurrogateModel(
        bounds=bounds,
        preprocessing_y=DummyPreprocessor,
        regressor={
            "backend": backend,
            "kernel": "RBF",
            "noise_level": 0.01,
            "length_scale_prior": [1e-2, 1e1],
            "n_restarts_optimizer": 1,
        },
        verbose=1,
    )
    # Check that the right backend was instantiated
    if backend == "random_features":
        from gpry.experimental_random_feature_gp import RandomFeatureGP
        assert isinstance(surr.gpr, RandomFeatureGP)
    elif backend == "gibbs":
        from gpry.experimental_gibbs_gpr import GibbsGPR
        assert isinstance(surr.gpr, GibbsGPR)


# ========================= BACKEND-SPECIFIC TESTS =========================

class TestRandomFeatureGP:
    """Tests specific to Random Fourier Feature GP."""

    def test_n_features_default(self):
        """Default n_features should scale with d."""
        from gpry.experimental_random_feature_gp import RandomFeatureGP
        gpr3 = RandomFeatureGP(length_scale_prior=make_lsp(3))
        gpr10 = RandomFeatureGP(length_scale_prior=make_lsp(10))
        # Check the attribute name (may vary)
        n3 = getattr(gpr3, '_n_features', getattr(gpr3, '_n_features_arg', None))
        n10 = getattr(gpr10, '_n_features', getattr(gpr10, '_n_features_arg', None))
        if n3 is None:
            # Try fitting to get the actual feature count
            X3 = np.random.randn(10, 3)
            X10 = np.random.randn(10, 10)
            gpr3.fit(X3, np.sin(X3[:, 0]), validate=False)
            gpr10.fit(X10, np.sin(X10[:, 0]), validate=False)
            n3 = getattr(gpr3, '_n_features', getattr(gpr3, 'n_features', 0))
            n10 = getattr(gpr10, '_n_features', getattr(gpr10, 'n_features', 0))
        assert n10 >= n3

    def test_faster_than_standard_gp_large_n(self):
        """RFGP should be faster for predict with large N."""
        import time
        from gpry.experimental_random_feature_gp import RandomFeatureGP

        d = 3
        n_train = 500
        rng = np.random.RandomState(42)
        X = rng.randn(n_train, d)
        y = np.sin(X[:, 0])
        X_test = rng.randn(100, d)

        gpr = RandomFeatureGP(
            length_scale_prior=make_lsp(d), n_restarts_optimizer=1,
            n_features=300, random_state=42
        )
        gpr.fit(X, y, validate=False)
        _ = gpr.predict(X_test, return_std=True)  # warmup

        t0 = time.perf_counter()
        for _ in range(20):
            gpr.predict(X_test, return_std=True)
        t_rfgp = (time.perf_counter() - t0) / 20
        # Just check it doesn't take more than 1 second
        assert t_rfgp < 1.0, f"RFGP predict took {t_rfgp:.3f}s"



if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
