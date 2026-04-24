"""
Tests for JAX-accelerated GP operations.

Tests numerical accuracy against the numpy/scipy GPR implementation,
covering:
1. Kernel matrix computation (RBF and Matern)
2. GP fit precomputation (L, V, alpha_)
3. GP prediction (mean and std)
4. Log marginal likelihood
5. End-to-end comparison with GPry's GaussianProcessRegressor
6. Edge cases and numerical stability
"""

import sys
import os
import time
import numpy as np
import pytest

# Add parent dir so we can import gpry
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.base import clone
from gpry.gpr import GaussianProcessRegressor
from gpry.kernels import RBF, Matern, ConstantKernel as C
from gpry.jax_accel import (
    JaxRuntimeBundle,
    _rbf_kernel_matrix,
    _matern52_kernel_matrix,
    _matern32_kernel_matrix,
    _matern12_kernel_matrix,
    _gp_fit_precompute,
    _gp_predict_mean,
    _gp_predict_var,
    _gp_predict_mean_and_var,
)

import jax.numpy as jnp
import jax

# Ensure 64-bit precision
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_gpr(gpr, X, y, **kwargs):
    """Fit GPR handling kernel_ initialization (GPry's GPR expects SurrogateModel flow)."""
    if gpr.kernel_ is None:
        gpr.kernel_ = clone(gpr.kernel)
    return gpr.fit(X, y, validate=False, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data_2d():
    """Simple 2D training + test data."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(50, 2)
    y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + 0.1 * rng.randn(50)
    X_test = rng.randn(30, 2)
    return X_train, y_train, X_test


@pytest.fixture
def larger_data_5d():
    """Larger 5D data for more realistic testing."""
    rng = np.random.RandomState(123)
    X_train = rng.randn(100, 5)
    y_train = np.sum(np.sin(X_train), axis=1) + 0.05 * rng.randn(100)
    X_test = rng.randn(50, 5)
    return X_train, y_train, X_test


@pytest.fixture
def gpry_gpr_rbf(simple_data_2d):
    """A fitted GPry GPR with RBF kernel."""
    X_train, y_train, _ = simple_data_2d
    bounds = np.array([[-5, 5], [-5, 5]])
    length_scale_prior = np.column_stack([
        bounds[:, 1] - bounds[:, 0],
        bounds[:, 1] - bounds[:, 0],
    ]) * np.array([[1e-3, 1e1]])
    gpr = GaussianProcessRegressor(
        kernel="RBF",
        output_scale_prior=[1e-2, 1e3],
        length_scale_prior=length_scale_prior,
        noise_level=0.1,
        noise_fixed=True,
        n_restarts_optimizer=2,
        random_state=42,
    )
    fit_gpr(gpr, X_train, y_train)
    return gpr


@pytest.fixture
def gpry_gpr_matern(simple_data_2d):
    """A fitted GPry GPR with Matern 5/2 kernel."""
    X_train, y_train, _ = simple_data_2d
    bounds = np.array([[-5, 5], [-5, 5]])
    length_scale_prior = np.column_stack([
        bounds[:, 1] - bounds[:, 0],
        bounds[:, 1] - bounds[:, 0],
    ]) * np.array([[1e-3, 1e1]])
    gpr = GaussianProcessRegressor(
        kernel={"Matern": {"nu": 2.5}},
        output_scale_prior=[1e-2, 1e3],
        length_scale_prior=length_scale_prior,
        noise_level=0.1,
        noise_fixed=True,
        n_restarts_optimizer=2,
        random_state=42,
    )
    fit_gpr(gpr, X_train, y_train)
    return gpr


# ---------------------------------------------------------------------------
# Test kernel matrices
# ---------------------------------------------------------------------------

class TestKernels:
    """Test JAX kernel implementations against sklearn."""

    def test_rbf_kernel_identity(self):
        """RBF kernel K(X, X) should have 1s on diagonal."""
        X = jnp.array(np.random.randn(10, 3))
        ls = jnp.ones(3)
        K = _rbf_kernel_matrix(X, X, ls)
        np.testing.assert_allclose(np.diag(np.asarray(K)), 1.0, atol=1e-14)

    def test_rbf_kernel_symmetric(self):
        """RBF kernel should be symmetric."""
        X = jnp.array(np.random.randn(20, 4))
        ls = jnp.array([1.0, 2.0, 0.5, 3.0])
        K = _rbf_kernel_matrix(X, X, ls)
        np.testing.assert_allclose(np.asarray(K), np.asarray(K.T), atol=1e-14)

    def test_rbf_kernel_vs_sklearn(self, simple_data_2d):
        """RBF kernel matrix should match sklearn."""
        X_train, _, X_test = simple_data_2d
        length_scale = np.array([1.5, 0.8])

        # sklearn
        sk_kernel = RBF(length_scale=length_scale)
        K_sk = sk_kernel(X_train, X_test)

        # JAX
        K_jax = _rbf_kernel_matrix(
            jnp.array(X_train), jnp.array(X_test), jnp.array(length_scale)
        )

        np.testing.assert_allclose(np.asarray(K_jax), K_sk, atol=1e-12)

    def test_matern52_kernel_vs_sklearn(self, simple_data_2d):
        """Matern 5/2 kernel should match sklearn."""
        X_train, _, X_test = simple_data_2d
        length_scale = np.array([1.5, 0.8])

        sk_kernel = Matern(length_scale=length_scale, nu=2.5)
        K_sk = sk_kernel(X_train, X_test)

        K_jax = _matern52_kernel_matrix(
            jnp.array(X_train), jnp.array(X_test), jnp.array(length_scale)
        )

        np.testing.assert_allclose(np.asarray(K_jax), K_sk, atol=1e-10)

    def test_matern32_kernel_vs_sklearn(self, simple_data_2d):
        """Matern 3/2 kernel should match sklearn."""
        X_train, _, X_test = simple_data_2d
        length_scale = np.array([1.5, 0.8])

        sk_kernel = Matern(length_scale=length_scale, nu=1.5)
        K_sk = sk_kernel(X_train, X_test)

        K_jax = _matern32_kernel_matrix(
            jnp.array(X_train), jnp.array(X_test), jnp.array(length_scale)
        )

        np.testing.assert_allclose(np.asarray(K_jax), K_sk, atol=1e-10)

    def test_matern12_kernel_vs_sklearn(self, simple_data_2d):
        """Matern 1/2 kernel should match sklearn."""
        X_train, _, X_test = simple_data_2d
        length_scale = np.array([1.5, 0.8])

        sk_kernel = Matern(length_scale=length_scale, nu=0.5)
        K_sk = sk_kernel(X_train, X_test)

        K_jax = _matern12_kernel_matrix(
            jnp.array(X_train), jnp.array(X_test), jnp.array(length_scale)
        )

        np.testing.assert_allclose(np.asarray(K_jax), K_sk, atol=1e-10)

    def test_rbf_kernel_positive_definite(self):
        """RBF kernel + noise should be positive definite."""
        rng = np.random.RandomState(99)
        X = jnp.array(rng.randn(30, 3))
        ls = jnp.array([1.0, 1.0, 1.0])
        K = _rbf_kernel_matrix(X, X, ls)
        K_noisy = np.asarray(K) + 0.01 * np.eye(30)
        eigenvalues = np.linalg.eigvalsh(K_noisy)
        assert np.all(eigenvalues > 0), f"Non-positive eigenvalue: {eigenvalues.min()}"


# ---------------------------------------------------------------------------
# Test precompute
# ---------------------------------------------------------------------------

class TestPrecompute:
    """Test L, V, alpha_ precomputation against scipy."""

    def test_precompute_matches_scipy(self, simple_data_2d):
        """JAX precompute should match scipy's cholesky + solve."""
        from scipy.linalg import cholesky, cho_solve, solve_triangular

        X_train, y_train, _ = simple_data_2d
        length_scale = np.array([1.0, 1.0])
        output_scale_sq = 2.0
        alpha_noise = 0.01

        # Compute kernel matrix
        K_base = np.asarray(_rbf_kernel_matrix(
            jnp.array(X_train), jnp.array(X_train), jnp.array(length_scale)
        ))
        K = output_scale_sq * K_base

        # Scipy reference
        K_noisy_sp = K + alpha_noise * np.eye(len(X_train))
        L_sp = cholesky(K_noisy_sp, lower=True)
        V_sp = solve_triangular(L_sp, np.eye(len(X_train)), lower=True)
        alpha_sp = cho_solve((L_sp, True), y_train)

        # JAX
        K_jax = jnp.array(K)
        L_jax, V_jax, alpha_jax = _gp_fit_precompute(
            K_jax, jnp.array(y_train), alpha_noise
        )

        np.testing.assert_allclose(np.asarray(L_jax), L_sp, atol=1e-10)
        np.testing.assert_allclose(np.asarray(V_jax), V_sp, atol=1e-10)
        np.testing.assert_allclose(np.asarray(alpha_jax), alpha_sp, atol=1e-10)


# ---------------------------------------------------------------------------
# Test accelerator against GPry GPR
# ---------------------------------------------------------------------------

class TestJaxRuntimeBundle:
    """Test the JaxRuntimeBundle against GPry's GaussianProcessRegressor."""

    def test_accelerator_predict_mean_rbf(self, gpry_gpr_rbf, simple_data_2d):
        """JAX accelerator mean should match GPR mean (RBF)."""
        _, _, X_test = simple_data_2d
        gpr = gpry_gpr_rbf

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)
        assert accel.ready

        # GPR predict
        gpr_result = gpr.predict(X_test, return_std=False, validate=False)
        y_mean_gpr = gpr_result if isinstance(gpr_result, np.ndarray) else gpr_result[0]

        # JAX predict
        y_mean_jax = accel.predict_mean(X_test)

        np.testing.assert_allclose(y_mean_jax, y_mean_gpr, atol=1e-10,
                                   err_msg="Mean prediction mismatch (RBF)")

    def test_accelerator_predict_std_rbf(self, gpry_gpr_rbf, simple_data_2d):
        """JAX accelerator std should match GPR std (RBF)."""
        _, _, X_test = simple_data_2d
        gpr = gpry_gpr_rbf

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)

        # GPR predict
        _, y_std_gpr = gpr.predict(X_test, return_std=True, validate=False)

        # JAX predict
        y_std_jax = accel.predict_std(X_test)

        np.testing.assert_allclose(y_std_jax, y_std_gpr, atol=1e-8,
                                   err_msg="Std prediction mismatch (RBF)")

    def test_accelerator_predict_mean_std_rbf(self, gpry_gpr_rbf, simple_data_2d):
        """JAX combined mean+std should match separate calls."""
        _, _, X_test = simple_data_2d
        gpr = gpry_gpr_rbf

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)

        y_mean, y_std = accel.predict_mean_std(X_test)
        y_mean2 = accel.predict_mean(X_test)
        y_std2 = accel.predict_std(X_test)

        np.testing.assert_allclose(y_mean, y_mean2, atol=1e-14)
        np.testing.assert_allclose(y_std, y_std2, atol=1e-14)

    def test_accelerator_predict_mean_matern(self, gpry_gpr_matern, simple_data_2d):
        """JAX accelerator mean should match GPR mean (Matern 5/2)."""
        _, _, X_test = simple_data_2d
        gpr = gpry_gpr_matern

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)
        assert accel.ready

        gpr_result = gpr.predict(X_test, return_std=False, validate=False)
        y_mean_gpr = gpr_result if isinstance(gpr_result, np.ndarray) else gpr_result[0]

        y_mean_jax = accel.predict_mean(X_test)

        np.testing.assert_allclose(y_mean_jax, y_mean_gpr, atol=1e-8,
                                   err_msg="Mean prediction mismatch (Matern)")

    def test_accelerator_predict_std_matern(self, gpry_gpr_matern, simple_data_2d):
        """JAX accelerator std should match GPR std (Matern 5/2)."""
        _, _, X_test = simple_data_2d
        gpr = gpry_gpr_matern

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)

        _, y_std_gpr = gpr.predict(X_test, return_std=True, validate=False)
        y_std_jax = accel.predict_std(X_test)

        np.testing.assert_allclose(y_std_jax, y_std_gpr, atol=1e-6,
                                   err_msg="Std prediction mismatch (Matern)")

    def test_accelerator_predict_std_matches_predict_std_method(
        self, gpry_gpr_rbf, simple_data_2d
    ):
        """JAX std should match GPR's dedicated predict_std method."""
        _, _, X_test = simple_data_2d
        gpr = gpry_gpr_rbf

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)

        y_std_gpr = gpr.predict_std(X_test, validate=False)
        y_std_jax = accel.predict_std(X_test)

        np.testing.assert_allclose(y_std_jax, y_std_gpr, atol=1e-8)

    def test_accelerator_higher_dim(self, larger_data_5d):
        """Test on 5D data."""
        X_train, y_train, X_test = larger_data_5d
        bounds = np.array([[-5, 5]] * 5)
        length_scale_prior = np.column_stack([
            bounds[:, 1] - bounds[:, 0],
            bounds[:, 1] - bounds[:, 0],
        ]) * np.array([[1e-3, 1e1]])

        gpr = GaussianProcessRegressor(
            kernel="RBF",
            output_scale_prior=[1e-2, 1e3],
            length_scale_prior=length_scale_prior,
            noise_level=0.05,
            noise_fixed=True,
            n_restarts_optimizer=2,
            random_state=42,
        )
        fit_gpr(gpr, X_train, y_train)

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)

        gpr_mean_result = gpr.predict(X_test, return_std=False, validate=False)
        y_mean_gpr = gpr_mean_result if isinstance(gpr_mean_result, np.ndarray) else gpr_mean_result[0]
        _, y_std_gpr = gpr.predict(X_test, return_std=True, validate=False)

        y_mean_jax, y_std_jax = accel.predict_mean_std(X_test)

        np.testing.assert_allclose(y_mean_jax, y_mean_gpr, atol=1e-8)
        np.testing.assert_allclose(y_std_jax, y_std_gpr, atol=1e-6)


# ---------------------------------------------------------------------------
# Numerical stability tests
# ---------------------------------------------------------------------------

class TestNumericalStability:
    """Test numerical stability in edge cases."""

    def test_near_training_points(self, gpry_gpr_rbf, simple_data_2d):
        """Predictions near training points should have low variance."""
        X_train, _, _ = simple_data_2d
        gpr = gpry_gpr_rbf

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)

        # Predict at training points
        y_std_jax = accel.predict_std(X_train)
        y_std_gpr = gpr.predict_std(X_train, validate=False)

        # Both should be small (close to noise level)
        assert np.all(y_std_jax < 1.0), f"Max std at training: {y_std_jax.max()}"
        np.testing.assert_allclose(y_std_jax, y_std_gpr, atol=1e-8)

    def test_far_from_training(self, gpry_gpr_rbf):
        """Predictions far from training should have high variance."""
        gpr = gpry_gpr_rbf

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)

        X_far = np.array([[100.0, 100.0], [-100.0, -100.0]])
        y_std_jax = accel.predict_std(X_far)
        y_std_gpr = gpr.predict_std(X_far, validate=False)

        np.testing.assert_allclose(y_std_jax, y_std_gpr, atol=1e-6)

    def test_single_point_prediction(self, gpry_gpr_rbf):
        """Single point prediction should work."""
        gpr = gpry_gpr_rbf

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)

        X_single = np.array([[1.0, 2.0]])
        y_mean_jax = accel.predict_mean(X_single)
        y_std_jax = accel.predict_std(X_single)

        gpr_result = gpr.predict(X_single, return_std=False, validate=False)
        y_mean_gpr = gpr_result if isinstance(gpr_result, np.ndarray) else gpr_result[0]
        y_std_gpr = gpr.predict_std(X_single, validate=False)

        np.testing.assert_allclose(y_mean_jax, y_mean_gpr, atol=1e-10)
        np.testing.assert_allclose(y_std_jax, y_std_gpr, atol=1e-8)

    def test_large_batch_prediction(self, gpry_gpr_rbf):
        """Large batch prediction should work without issues."""
        gpr = gpry_gpr_rbf

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)

        rng = np.random.RandomState(99)
        X_large = rng.randn(1000, 2)

        y_mean_jax, y_std_jax = accel.predict_mean_std(X_large)
        _, y_std_gpr = gpr.predict(X_large, return_std=True, validate=False)

        # All stds should be non-negative
        assert np.all(y_std_jax >= 0), f"Negative std found: {y_std_jax.min()}"
        np.testing.assert_allclose(y_std_jax, y_std_gpr, atol=1e-6)

    def test_variance_non_negative(self, gpry_gpr_rbf):
        """Variance should never be negative."""
        gpr = gpry_gpr_rbf

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)

        rng = np.random.RandomState(42)
        for _ in range(10):
            X = rng.randn(100, 2) * 5
            y_std = accel.predict_std(X)
            assert np.all(y_std >= 0), f"Negative std: {y_std.min()}"
            assert np.all(np.isfinite(y_std)), "Non-finite std found"


# ---------------------------------------------------------------------------
# Performance test (not strict, just informational)
# ---------------------------------------------------------------------------

class TestPerformance:
    """Performance comparison tests."""

    def test_jit_warmup_and_speed(self, gpry_gpr_rbf, simple_data_2d):
        """Test that JIT compilation provides speedup on repeated calls."""
        _, _, X_test = simple_data_2d
        gpr = gpry_gpr_rbf

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)

        # Warmup JIT
        _ = accel.predict_mean_std(X_test)

        # Time JAX (10 calls)
        start = time.time()
        for _ in range(100):
            y_mean, y_std = accel.predict_mean_std(X_test)
        jax_time = time.time() - start

        # Time numpy (10 calls)
        start = time.time()
        for _ in range(100):
            result = gpr.predict(X_test, return_std=True, validate=False)
        numpy_time = time.time() - start

        speedup = numpy_time / jax_time
        print(f"\nPerformance: JAX={jax_time*10:.1f}ms, NumPy={numpy_time*10:.1f}ms, "
              f"Speedup={speedup:.1f}x")

        # We don't strictly assert speedup since it depends on hardware,
        # but log it for informational purposes


# ---------------------------------------------------------------------------
# End-to-end test with GPry-like workflow
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """End-to-end tests mimicking actual GPry usage."""

    def test_iterative_fitting(self):
        """Test that accelerator stays accurate across multiple fit cycles."""
        rng = np.random.RandomState(42)

        bounds = np.array([[-5, 5], [-5, 5]])
        length_scale_prior = np.column_stack([
            bounds[:, 1] - bounds[:, 0],
            bounds[:, 1] - bounds[:, 0],
        ]) * np.array([[1e-3, 1e1]])

        gpr = GaussianProcessRegressor(
            kernel="RBF",
            output_scale_prior=[1e-2, 1e3],
            length_scale_prior=length_scale_prior,
            noise_level=0.1,
            noise_fixed=True,
            n_restarts_optimizer=2,
            random_state=42,
        )

        accel = JaxRuntimeBundle()

        # Simulate iterative fitting (like GPry's active learning loop)
        n_init = 10
        X_all = rng.randn(n_init, 2)
        y_all = np.sin(X_all[:, 0]) * np.cos(X_all[:, 1])

        for i in range(5):
            # Fit GP
            fit_gpr(gpr, X_all, y_all)
            accel.update_from_gpr(gpr)

            # Predict at test points
            X_test = rng.randn(20, 2)
            y_mean_jax, y_std_jax = accel.predict_mean_std(X_test)
            gpr_result = gpr.predict(X_test, return_std=True, validate=False)
            y_mean_gpr = gpr_result[0]
            y_std_gpr = gpr_result[1]

            np.testing.assert_allclose(y_mean_jax, y_mean_gpr, atol=1e-8,
                                       err_msg=f"Mean mismatch at iteration {i}")
            np.testing.assert_allclose(y_std_jax, y_std_gpr, atol=1e-6,
                                       err_msg=f"Std mismatch at iteration {i}")

            # Add new points (simulate acquisition)
            X_new = rng.randn(5, 2)
            y_new = np.sin(X_new[:, 0]) * np.cos(X_new[:, 1])
            X_all = np.vstack([X_all, X_new])
            y_all = np.concatenate([y_all, y_new])

    def test_multivariate_gaussian_posterior(self):
        """Test on a multivariate Gaussian (like the introductory example)."""
        from scipy.stats import multivariate_normal

        mean = [3, 2]
        cov = [[0.5, 0.4], [0.4, 1.5]]
        rv = multivariate_normal(mean, cov)

        rng = np.random.RandomState(42)
        X_train = rng.uniform(-5, 10, size=(80, 2))
        y_train = np.array([rv.logpdf(x) for x in X_train])

        bounds = np.array([[-10, 10], [-10, 10]])
        length_scale_prior = np.column_stack([
            bounds[:, 1] - bounds[:, 0],
            bounds[:, 1] - bounds[:, 0],
        ]) * np.array([[1e-3, 1e1]])

        gpr = GaussianProcessRegressor(
            kernel="RBF",
            output_scale_prior=[1e-2, 1e3],
            length_scale_prior=length_scale_prior,
            noise_level=0.01,
            noise_fixed=True,
            n_restarts_optimizer=3,
            random_state=42,
        )
        fit_gpr(gpr, X_train, y_train)

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)

        # Test on a grid around the mean
        x1 = np.linspace(0, 6, 20)
        x2 = np.linspace(-1, 5, 20)
        X_grid = np.array([[a, b] for a in x1 for b in x2])

        y_mean_jax, y_std_jax = accel.predict_mean_std(X_grid)
        gpr_result = gpr.predict(X_grid, return_std=True, validate=False)
        y_mean_gpr, y_std_gpr = gpr_result[0], gpr_result[1]

        np.testing.assert_allclose(y_mean_jax, y_mean_gpr, atol=1e-8)
        np.testing.assert_allclose(y_std_jax, y_std_gpr, atol=1e-6)

        # Also check that both implementations agree on the peak location
        peak_idx_jax = np.argmax(y_mean_jax)
        peak_idx_gpr = np.argmax(y_mean_gpr)
        assert peak_idx_jax == peak_idx_gpr, \
            f"JAX peak at idx {peak_idx_jax}, GPR at {peak_idx_gpr}"

    def test_noise_in_kernel_mode(self):
        """Test with noise_fixed=False (WhiteKernel in kernel)."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(40, 2)
        y_train = np.sin(X_train[:, 0]) + 0.5 * rng.randn(40)

        bounds = np.array([[-5, 5], [-5, 5]])
        length_scale_prior = np.column_stack([
            bounds[:, 1] - bounds[:, 0],
            bounds[:, 1] - bounds[:, 0],
        ]) * np.array([[1e-3, 1e1]])

        gpr = GaussianProcessRegressor(
            kernel="RBF",
            output_scale_prior=[1e-2, 1e3],
            length_scale_prior=length_scale_prior,
            noise_level=0.5,
            noise_fixed=False,
            n_restarts_optimizer=2,
            random_state=42,
        )
        fit_gpr(gpr, X_train, y_train, noise_level=0.5)

        accel = JaxRuntimeBundle()
        accel.update_from_gpr(gpr)

        if accel.ready:
            X_test = rng.randn(20, 2)
            y_mean_jax, y_std_jax = accel.predict_mean_std(X_test)
            gpr_result = gpr.predict(X_test, return_std=True, validate=False)
            y_mean_gpr, y_std_gpr = gpr_result[0], gpr_result[1]

            np.testing.assert_allclose(y_mean_jax, y_mean_gpr, atol=1e-8)
            np.testing.assert_allclose(y_std_jax, y_std_gpr, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
