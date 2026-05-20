"""
Tests for JAX automatic differentiation gradients in GPry.

Tests cover:
1. LML gradient w.r.t. theta (hyperparameter optimization)
2. Predict mean gradient w.r.t. X (acquisition functions)
3. Predict std gradient w.r.t. X (acquisition functions)
4. Integration with GPR class (predict with gradients, LML with gradients)
5. Consistency with hand-coded numpy gradients
"""

import sys
import os
import numpy as np
from sklearn.base import clone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry.gpr import GaussianProcessRegressor, JaxGaussianProcessRegressor
from gpry.gpr_jax import _numpy_log_marginal_likelihood


def make_gpr(kernel="RBF", n_dims=2, noise_fixed=True, use_jax=True, **kwargs):
    """Helper to create a GPR with standard priors."""
    bounds = np.array([[-5, 5]] * n_dims)
    length_scale_prior = np.column_stack([
        bounds[:, 1] - bounds[:, 0],
        bounds[:, 1] - bounds[:, 0],
    ]) * np.array([[1e-3, 1e1]])
    return GaussianProcessRegressor(
        kernel=kernel,
        output_scale_prior=[1e-2, 1e3],
        length_scale_prior=length_scale_prior,
        noise_level=0.1,
        noise_fixed=noise_fixed,
        n_hyperopt_restarts=2,
        random_state=42,
        use_jax=use_jax,
        **kwargs,
    )


def fit_gpr(gpr, X, y, noise_level=None):
    """Fit a GPR, handling the kernel_ = None quirk."""
    if gpr.fitted_kernel is None:
        gpr.fitted_kernel = clone(gpr.kernel)
    kwargs = {"validate": False}
    if noise_level is not None:
        kwargs["noise_level"] = noise_level
    return gpr.fit(X, y, **kwargs)


# ---------------------------------------------------------------------------
# LML gradient tests
# ---------------------------------------------------------------------------

class TestLMLGradient:
    """Test JAX auto-diff LML gradient against sklearn analytical gradient."""

    def test_lml_gradient_rbf(self):
        """LML gradient for RBF kernel matches sklearn."""
        rng = np.random.RandomState(42)
        gpr = make_gpr("RBF", n_dims=3)
        X = rng.randn(50, 3)
        y = np.sin(X[:, 0]) + 0.1 * rng.randn(50)
        fit_gpr(gpr, X, y)

        theta = gpr.fitted_kernel.theta
        # JAX gradient
        lml_jax, grad_jax = gpr.log_marginal_likelihood_with_grad(theta)
        # sklearn gradient (via super())
        lml_sk, grad_sk = _numpy_log_marginal_likelihood(
            gpr, theta, eval_gradient=True, clone_kernel=True)

        np.testing.assert_allclose(lml_jax, lml_sk, atol=1e-6,
                                   err_msg="LML value mismatch")
        np.testing.assert_allclose(grad_jax, grad_sk, rtol=1e-4, atol=1e-6,
                                   err_msg="LML gradient mismatch")
        print(f"  RBF LML gradient: max_diff={np.max(np.abs(grad_jax - grad_sk)):.2e}")

    def test_lml_gradient_matern52(self):
        """LML gradient for Matern 5/2 kernel matches sklearn."""
        rng = np.random.RandomState(42)
        gpr = make_gpr({"Matern": {"nu": 2.5}}, n_dims=2)
        X = rng.randn(40, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(40)
        fit_gpr(gpr, X, y)

        theta = gpr.fitted_kernel.theta
        lml_jax, grad_jax = gpr.log_marginal_likelihood_with_grad(theta)
        lml_sk, grad_sk = _numpy_log_marginal_likelihood(
            gpr, theta, eval_gradient=True, clone_kernel=True)

        np.testing.assert_allclose(lml_jax, lml_sk, atol=1e-6)
        np.testing.assert_allclose(grad_jax, grad_sk, rtol=1e-4, atol=1e-6)
        print(f"  Matern52 LML gradient: max_diff={np.max(np.abs(grad_jax - grad_sk)):.2e}")

    def test_lml_gradient_matern32(self):
        """LML gradient for Matern 3/2 kernel matches sklearn."""
        rng = np.random.RandomState(42)
        gpr = make_gpr({"Matern": {"nu": 1.5}}, n_dims=2)
        X = rng.randn(40, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(40)
        fit_gpr(gpr, X, y)

        theta = gpr.fitted_kernel.theta
        lml_jax, grad_jax = gpr.log_marginal_likelihood_with_grad(theta)
        lml_sk, grad_sk = _numpy_log_marginal_likelihood(
            gpr, theta, eval_gradient=True, clone_kernel=True)

        np.testing.assert_allclose(lml_jax, lml_sk, atol=1e-6)
        np.testing.assert_allclose(grad_jax, grad_sk, rtol=1e-4, atol=1e-6)
        print(f"  Matern32 LML gradient: max_diff={np.max(np.abs(grad_jax - grad_sk)):.2e}")

    def test_lml_gradient_with_white_noise(self):
        """LML gradient with WhiteKernel (noise_fixed=False) matches sklearn."""
        rng = np.random.RandomState(42)
        gpr = make_gpr("RBF", n_dims=2, noise_fixed=False)
        X = rng.randn(40, 2)
        y = np.sum(np.sin(X), axis=1) + 0.2 * rng.randn(40)
        fit_gpr(gpr, X, y, noise_level=0.2)

        theta = gpr.fitted_kernel.theta
        lml_jax, grad_jax = gpr.log_marginal_likelihood_with_grad(theta)
        lml_sk, grad_sk = _numpy_log_marginal_likelihood(
            gpr, theta, eval_gradient=True, clone_kernel=True)

        np.testing.assert_allclose(lml_jax, lml_sk, atol=1e-6)
        # Looser tolerance for white noise: sklearn zeros gradient for
        # parameters at bounds, but JAX computes the true mathematical gradient
        np.testing.assert_allclose(grad_jax, grad_sk, rtol=1e-3, atol=1e-5)
        print(f"  WhiteNoise LML gradient: max_diff={np.max(np.abs(grad_jax - grad_sk)):.2e}")

    def test_lml_gradient_finite_difference(self):
        """LML gradient matches finite differences (gold standard check).

        We evaluate at a theta with reasonable length scales (~1) rather than
        at the fitted optimum, which can have extreme length scales (e.g. 1e-4)
        where both autodiff and FD suffer from numerical issues (kernel values
        saturate to zero, and catastrophic cancellation in the squared-distance
        computation makes FD unreliable).
        """
        rng = np.random.RandomState(42)
        gpr = make_gpr("RBF", n_dims=2)
        X = rng.randn(30, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(30)
        fit_gpr(gpr, X, y)

        # Use a theta with reasonable length scales so the kernel is non-trivial
        theta = np.array([0.0, 0.5, 0.5])
        _, grad_jax = gpr.log_marginal_likelihood_with_grad(theta)

        # Finite difference gradient
        eps = 1e-5
        grad_fd = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[i] += eps
            theta_m[i] -= eps
            lml_p, _ = gpr.log_marginal_likelihood_with_grad(theta_p)
            lml_m, _ = gpr.log_marginal_likelihood_with_grad(theta_m)
            grad_fd[i] = (lml_p - lml_m) / (2 * eps)

        np.testing.assert_allclose(grad_jax, grad_fd, rtol=1e-4, atol=1e-6)
        print(f"  FD vs JAX LML gradient: max_diff={np.max(np.abs(grad_jax - grad_fd)):.2e}")


# ---------------------------------------------------------------------------
# Predict gradient tests
# ---------------------------------------------------------------------------

class TestPredictGradient:
    """Test JAX auto-diff predict gradients against hand-coded numpy gradients."""

    def test_mean_gradient_rbf(self):
        """Mean gradient for RBF matches hand-coded gradient_x."""
        rng = np.random.RandomState(42)
        gpr = make_gpr("RBF", n_dims=3)
        X = rng.randn(50, 3)
        y = np.sin(X[:, 0]) + 0.1 * rng.randn(50)
        fit_gpr(gpr, X, y)

        x_test = rng.randn(1, 3)
        # JAX gradient
        grad_jax = gpr.predict_mean_grad(x_test[0])

        # Numpy gradient (hand-coded kernel gradient_x)
        grad_kernel = gpr.fitted_kernel.gradient_x(x_test[0], gpr.X_train_)
        grad_np = np.dot(grad_kernel.T, gpr.alpha_)

        np.testing.assert_allclose(grad_jax, grad_np, rtol=1e-4, atol=1e-6)
        print(f"  RBF mean_grad: max_diff={np.max(np.abs(grad_jax - grad_np)):.2e}")

    def test_mean_gradient_matern52(self):
        """Mean gradient for Matern 5/2 matches hand-coded gradient_x."""
        rng = np.random.RandomState(42)
        gpr = make_gpr({"Matern": {"nu": 2.5}}, n_dims=2)
        X = rng.randn(40, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(40)
        fit_gpr(gpr, X, y)

        x_test = rng.randn(1, 2)
        grad_jax = gpr.predict_mean_grad(x_test[0])

        grad_kernel = gpr.fitted_kernel.gradient_x(x_test[0], gpr.X_train_)
        grad_np = np.dot(grad_kernel.T, gpr.alpha_)

        np.testing.assert_allclose(grad_jax, grad_np, rtol=1e-4, atol=1e-6)
        print(f"  Matern52 mean_grad: max_diff={np.max(np.abs(grad_jax - grad_np)):.2e}")

    def test_mean_gradient_matern32(self):
        """Mean gradient for Matern 3/2 matches hand-coded gradient_x."""
        rng = np.random.RandomState(42)
        gpr = make_gpr({"Matern": {"nu": 1.5}}, n_dims=2)
        X = rng.randn(40, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(40)
        fit_gpr(gpr, X, y)

        x_test = rng.randn(1, 2)
        grad_jax = gpr.predict_mean_grad(x_test[0])

        grad_kernel = gpr.fitted_kernel.gradient_x(x_test[0], gpr.X_train_)
        grad_np = np.dot(grad_kernel.T, gpr.alpha_)

        np.testing.assert_allclose(grad_jax, grad_np, rtol=1e-4, atol=1e-6)
        print(f"  Matern32 mean_grad: max_diff={np.max(np.abs(grad_jax - grad_np)):.2e}")

    def test_mean_gradient_matern12(self):
        """Mean gradient for Matern 1/2 matches finite differences.

        Note: kernels.py gradient_x for Matern 1/2 has a pre-existing bug with
        multi-dimensional inputs, so we validate against finite differences.
        """
        rng = np.random.RandomState(42)
        gpr = make_gpr({"Matern": {"nu": 0.5}}, n_dims=2)
        X = rng.randn(40, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(40)
        fit_gpr(gpr, X, y)

        x_test = np.array([2.5, -1.3])
        grad_jax = gpr.predict_mean_grad(x_test)

        # Finite difference
        eps = 1e-5
        grad_fd = np.zeros(2)
        for i in range(2):
            x_p = x_test.copy()
            x_m = x_test.copy()
            x_p[i] += eps
            x_m[i] -= eps
            mean_p = gpr.predict_mean(x_p[None, :])[0]
            mean_m = gpr.predict_mean(x_m[None, :])[0]
            grad_fd[i] = (mean_p - mean_m) / (2 * eps)

        np.testing.assert_allclose(grad_jax, grad_fd, rtol=1e-3, atol=1e-5)
        print(f"  Matern12 mean_grad: max_diff={np.max(np.abs(grad_jax - grad_fd)):.2e}")

    def test_mean_gradient_finite_difference(self):
        """Mean gradient matches finite differences."""
        rng = np.random.RandomState(42)
        gpr = make_gpr("RBF", n_dims=2)
        X = rng.randn(30, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(30)
        fit_gpr(gpr, X, y)

        x_test = rng.randn(2)
        grad_jax = gpr.predict_mean_grad(x_test)

        # Finite difference
        eps = 1e-5
        grad_fd = np.zeros(2)
        for i in range(2):
            x_p = x_test.copy()
            x_m = x_test.copy()
            x_p[i] += eps
            x_m[i] -= eps
            mean_p = gpr.predict_mean(x_p[None, :])[0]
            mean_m = gpr.predict_mean(x_m[None, :])[0]
            grad_fd[i] = (mean_p - mean_m) / (2 * eps)

        np.testing.assert_allclose(grad_jax, grad_fd, rtol=1e-4, atol=1e-6)
        print(f"  FD vs JAX mean_grad: max_diff={np.max(np.abs(grad_jax - grad_fd)):.2e}")

    def test_std_gradient_finite_difference(self):
        """Std gradient matches finite differences."""
        rng = np.random.RandomState(42)
        gpr = make_gpr("RBF", n_dims=2)
        X = rng.randn(30, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(30)
        fit_gpr(gpr, X, y)

        x_test = rng.randn(2)
        grad_jax = gpr.predict_std_grad(x_test)

        # Finite difference
        eps = 1e-5
        grad_fd = np.zeros(2)
        for i in range(2):
            x_p = x_test.copy()
            x_m = x_test.copy()
            x_p[i] += eps
            x_m[i] -= eps
            std_p = gpr.predict_std_native(x_p[None, :])[0]
            std_m = gpr.predict_std_native(x_m[None, :])[0]
            grad_fd[i] = (std_p - std_m) / (2 * eps)

        np.testing.assert_allclose(grad_jax, grad_fd, rtol=1e-3, atol=1e-6)
        print(f"  FD vs JAX std_grad: max_diff={np.max(np.abs(grad_jax - grad_fd)):.2e}")

    def test_std_gradient_matern52_finite_difference(self):
        """Std gradient for Matern 5/2 matches finite differences."""
        rng = np.random.RandomState(42)
        gpr = make_gpr({"Matern": {"nu": 2.5}}, n_dims=2)
        X = rng.randn(30, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(30)
        fit_gpr(gpr, X, y)

        x_test = rng.randn(2)
        grad_jax = gpr.predict_std_grad(x_test)

        eps = 1e-5
        grad_fd = np.zeros(2)
        for i in range(2):
            x_p = x_test.copy()
            x_m = x_test.copy()
            x_p[i] += eps
            x_m[i] -= eps
            std_p = gpr.predict_std_native(x_p[None, :])[0]
            std_m = gpr.predict_std_native(x_m[None, :])[0]
            grad_fd[i] = (std_p - std_m) / (2 * eps)

        np.testing.assert_allclose(grad_jax, grad_fd, rtol=1e-3, atol=1e-6)
        print(f"  FD vs JAX Matern52 std_grad: max_diff={np.max(np.abs(grad_jax - grad_fd)):.2e}")


# ---------------------------------------------------------------------------
# Integration tests: GPR.predict() with gradients uses JAX
# ---------------------------------------------------------------------------

class TestGPRGradientIntegration:
    """Test that gpr.predict(return_mean_grad=True) uses JAX path."""

    def test_predict_mean_grad_via_gpr(self):
        """GPR.predict(return_mean_grad=True) uses JAX and matches numpy."""
        rng = np.random.RandomState(42)
        gpr_jax = make_gpr("RBF", n_dims=2, use_jax=True)
        gpr_np = make_gpr("RBF", n_dims=2, use_jax=False)
        X = rng.randn(40, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(40)
        fit_gpr(gpr_jax, X, y)
        fit_gpr(gpr_np, X, y)
        # Copy fitted kernel to ensure identical hyperparameters
        gpr_np.fitted_kernel = clone(gpr_jax.fitted_kernel)
        gpr_np.fit(X, y, fit_hyperparameters=False, validate=False)

        x_test = rng.randn(1, 2)
        result_jax = gpr_jax.predict(
            x_test, return_std=True, return_mean_grad=True, validate=False)
        result_np = gpr_np.predict(
            x_test, return_std=True, return_mean_grad=True, validate=False)

        # y_mean, y_std, mean_grad
        np.testing.assert_allclose(result_jax[0], result_np[0], atol=1e-6,
                                   err_msg="Mean mismatch")
        np.testing.assert_allclose(result_jax[1], result_np[1], atol=1e-6,
                                   err_msg="Std mismatch")
        np.testing.assert_allclose(result_jax[2], result_np[2], rtol=1e-4, atol=1e-6,
                                   err_msg="Mean gradient mismatch")
        print(f"  GPR predict mean_grad matches: "
              f"max_diff={np.max(np.abs(result_jax[2] - result_np[2])):.2e}")

    def test_predict_std_grad_via_gpr(self):
        """GPR.predict(return_std_grad=True) uses JAX and matches numpy."""
        rng = np.random.RandomState(42)
        gpr_jax = make_gpr("RBF", n_dims=2, use_jax=True)
        X = rng.randn(40, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(40)
        fit_gpr(gpr_jax, X, y)

        x_test = rng.randn(1, 2)
        result = gpr_jax.predict(
            x_test, return_std=True, return_mean_grad=True,
            return_std_grad=True, validate=False)

        # Should return [y_mean, y_std, mean_grad, std_grad]
        assert len(result) == 4, f"Expected 4 return values, got {len(result)}"
        # Verify std_grad with finite differences
        eps = 1e-5
        grad_fd = np.zeros(2)
        for i in range(2):
            x_p = x_test.copy()
            x_m = x_test.copy()
            x_p[0, i] += eps
            x_m[0, i] -= eps
            std_p = gpr_jax.predict(x_p, return_std=True, validate=False)[1][0]
            std_m = gpr_jax.predict(x_m, return_std=True, validate=False)[1][0]
            grad_fd[i] = (std_p - std_m) / (2 * eps)

        np.testing.assert_allclose(result[3], grad_fd, rtol=1e-3, atol=1e-6)
        print(f"  GPR predict std_grad FD check: "
              f"max_diff={np.max(np.abs(result[3] - grad_fd)):.2e}")

    def test_lml_gradient_via_gpr(self):
        """GPR.log_marginal_likelihood(eval_gradient=True) uses JAX."""
        rng = np.random.RandomState(42)
        gpr = make_gpr("RBF", n_dims=2)
        X = rng.randn(40, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(40)
        fit_gpr(gpr, X, y)

        theta = gpr.fitted_kernel.theta
        lml, grad = gpr.log_marginal_likelihood(theta, eval_gradient=True)
        lml_sk, grad_sk = _numpy_log_marginal_likelihood(
            gpr, theta, eval_gradient=True, clone_kernel=True)

        np.testing.assert_allclose(lml, lml_sk, atol=1e-6)
        np.testing.assert_allclose(grad, grad_sk, rtol=1e-4, atol=1e-6)
        print(f"  GPR LML gradient via JAX: max_diff={np.max(np.abs(grad - grad_sk)):.2e}")

    def test_hyperparameter_optimization_with_jax_gradients(self):
        """Full hyperparameter optimization works with JAX gradients."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(50)

        gpr_jax = make_gpr("RBF", n_dims=2, use_jax=True)
        fit_gpr(gpr_jax, X, y)
        lml_jax = gpr_jax.log_marginal_likelihood_value_
        theta_jax = gpr_jax.fitted_kernel.theta.copy()

        gpr_np = make_gpr("RBF", n_dims=2, use_jax=False)
        fit_gpr(gpr_np, X, y)
        lml_np = gpr_np.log_marginal_likelihood_value_
        theta_np = gpr_np.fitted_kernel.theta.copy()

        # Both should converge to similar optima
        np.testing.assert_allclose(lml_jax, lml_np, rtol=1e-3,
                                   err_msg="LML optima differ between JAX and numpy")
        np.testing.assert_allclose(theta_jax, theta_np, atol=0.1,
                                   err_msg="Optimal theta differs between JAX and numpy")
        print(f"  Hyperparam optimization: LML_jax={lml_jax:.4f}, LML_np={lml_np:.4f}")
        print(f"  Theta diff: {np.max(np.abs(theta_jax - theta_np)):.4f}")


# ---------------------------------------------------------------------------
# Higher-dimensional and edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_high_dimensional(self):
        """Gradients work in 5D."""
        rng = np.random.RandomState(42)
        gpr = make_gpr("RBF", n_dims=5)
        X = rng.randn(60, 5)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(60)
        fit_gpr(gpr, X, y)

        x_test = rng.randn(5)

        # Mean gradient
        grad_jax = gpr.predict_mean_grad(x_test)
        grad_kernel = gpr.fitted_kernel.gradient_x(x_test, gpr.X_train_)
        grad_np = np.dot(grad_kernel.T, gpr.alpha_)
        np.testing.assert_allclose(grad_jax, grad_np, rtol=1e-4, atol=1e-6)

        # LML gradient
        _, grad_lml_jax = gpr.log_marginal_likelihood_with_grad(
            gpr.fitted_kernel.theta)
        _, grad_lml_sk = _numpy_log_marginal_likelihood(
            gpr, gpr.fitted_kernel.theta, eval_gradient=True, clone_kernel=True)
        np.testing.assert_allclose(grad_lml_jax, grad_lml_sk, rtol=1e-4, atol=1e-6)
        print(f"  5D gradients: mean_diff={np.max(np.abs(grad_jax - grad_np)):.2e}, "
              f"lml_diff={np.max(np.abs(grad_lml_jax - grad_lml_sk)):.2e}")

    def test_near_training_point(self):
        """Gradients near training data don't blow up or NaN."""
        rng = np.random.RandomState(42)
        gpr = make_gpr("RBF", n_dims=2)
        X = rng.randn(30, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(30)
        fit_gpr(gpr, X, y)

        # Test at a point very near a training point
        x_near = X[0] + 1e-8 * rng.randn(2)
        grad_mean = gpr.predict_mean_grad(x_near)
        grad_std = gpr.predict_std_grad(x_near)

        assert np.all(np.isfinite(grad_mean)), "Mean gradient has non-finite values"
        assert np.all(np.isfinite(grad_std)), "Std gradient has non-finite values"
        print(f"  Near-training: mean_grad={grad_mean}, std_grad={grad_std}")

    def test_predict_with_grads_consistency(self):
        """predict_with_grads returns consistent mean/std with predict_mean/predict_std."""
        rng = np.random.RandomState(42)
        gpr = make_gpr("RBF", n_dims=2)
        X = rng.randn(30, 2)
        y = np.sum(np.sin(X), axis=1) + 0.1 * rng.randn(30)
        fit_gpr(gpr, X, y)

        x_test = rng.randn(2)
        mean, std, mean_grad, std_grad = gpr.predict_with_grads(x_test)

        mean_ref = gpr.predict_mean(x_test[None, :])[0]
        std_ref = gpr.predict_std_native(x_test[None, :])[0]
        mean_grad_ref = gpr.predict_mean_grad(x_test)
        std_grad_ref = gpr.predict_std_grad(x_test)

        np.testing.assert_allclose(mean, mean_ref, atol=1e-10)
        np.testing.assert_allclose(std, std_ref, atol=1e-10)
        np.testing.assert_allclose(mean_grad, mean_grad_ref, atol=1e-10)
        np.testing.assert_allclose(std_grad, std_grad_ref, atol=1e-10)
        print("  predict_with_grads consistency OK")


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("LML Gradient Tests")
    print("=" * 60)
    t = TestLMLGradient()
    t.test_lml_gradient_rbf()
    t.test_lml_gradient_matern52()
    t.test_lml_gradient_matern32()
    t.test_lml_gradient_with_white_noise()
    t.test_lml_gradient_finite_difference()

    print("\n" + "=" * 60)
    print("Predict Gradient Tests")
    print("=" * 60)
    t2 = TestPredictGradient()
    t2.test_mean_gradient_rbf()
    t2.test_mean_gradient_matern52()
    t2.test_mean_gradient_matern32()
    t2.test_mean_gradient_matern12()
    t2.test_mean_gradient_finite_difference()
    t2.test_std_gradient_finite_difference()
    t2.test_std_gradient_matern52_finite_difference()

    print("\n" + "=" * 60)
    print("GPR Integration Tests")
    print("=" * 60)
    t3 = TestGPRGradientIntegration()
    t3.test_predict_mean_grad_via_gpr()
    t3.test_predict_std_grad_via_gpr()
    t3.test_lml_gradient_via_gpr()
    t3.test_hyperparameter_optimization_with_jax_gradients()

    print("\n" + "=" * 60)
    print("Edge Case Tests")
    print("=" * 60)
    t4 = TestEdgeCases()
    t4.test_high_dimensional()
    t4.test_near_training_point()
    t4.test_predict_with_grads_consistency()

    print("\n" + "=" * 60)
    print("ALL GRADIENT TESTS PASSED!")
    print("=" * 60)
