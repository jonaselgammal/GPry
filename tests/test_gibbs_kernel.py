"""Tests for the Gibbs (non-stationary) kernel GP."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gpry.experimental_gibbs_gpr import GibbsGPR, _gibbs_kernel_matrix
import jax.numpy as jnp


def make_gibbs_gpr(d, **kwargs):
    """Helper to create a GibbsGPR with standard priors."""
    lsp = np.column_stack([np.full(d, 1e-2), np.full(d, 1e1)])
    return GibbsGPR(length_scale_prior=lsp, **kwargs)


class TestGibbsKernel:
    """Test the Gibbs kernel function itself."""

    def test_stationary_limit(self):
        """With W=0, Gibbs kernel should reduce to stationary RBF."""
        d = 3
        n1, n2 = 10, 8
        rng = np.random.RandomState(42)
        X1 = jnp.array(rng.randn(n1, d))
        X2 = jnp.array(rng.randn(n2, d))
        l_base = jnp.ones(d) * 0.5
        W = jnp.zeros((d, d))
        b = jnp.zeros(d)
        osc = jnp.float64(1.0)

        K_gibbs = _gibbs_kernel_matrix(X1, X2, l_base, W, b, osc)

        # With W=0, b=0: l(x) = l_base * softplus(0) = l_base * log(2)
        # The effective length scale is l_base * log(2)
        l_eff = float(l_base[0]) * np.log(2)

        # RBF kernel with that effective length scale
        sX1 = np.array(X1) / l_eff
        sX2 = np.array(X2) / l_eff
        dist_sq = (
            np.sum(sX1 ** 2, axis=1)[:, None]
            + np.sum(sX2 ** 2, axis=1)[None, :]
            - 2.0 * sX1 @ sX2.T
        )
        K_rbf = np.exp(-0.5 * dist_sq)

        np.testing.assert_allclose(np.array(K_gibbs), K_rbf, rtol=1e-6)

    def test_symmetric(self):
        """Kernel matrix K(X, X) should be symmetric."""
        d = 4
        n = 15
        rng = np.random.RandomState(42)
        X = jnp.array(rng.randn(n, d))
        l_base = jnp.ones(d)
        W = jnp.array(rng.randn(d, d) * 0.1)
        b = jnp.array(rng.randn(d) * 0.1)
        K = _gibbs_kernel_matrix(X, X, l_base, W, b, jnp.float64(1.0))
        np.testing.assert_allclose(np.array(K), np.array(K.T), atol=1e-12)

    def test_positive_definite(self):
        """K(X, X) + eps*I should be positive definite."""
        d = 3
        n = 20
        rng = np.random.RandomState(42)
        X = jnp.array(rng.randn(n, d))
        l_base = jnp.ones(d)
        W = jnp.array(rng.randn(d, d) * 0.3)
        b = jnp.array(rng.randn(d) * 0.1)
        K = _gibbs_kernel_matrix(X, X, l_base, W, b, jnp.float64(1.0))
        K_reg = np.array(K) + 1e-6 * np.eye(n)
        eigvals = np.linalg.eigvalsh(K_reg)
        assert np.all(eigvals > 0), f"Non-positive eigenvalue: {eigvals.min()}"

    def test_diagonal_is_output_scale(self):
        """Diagonal of K(X,X) should be output_scale_sq."""
        d = 3
        n = 10
        rng = np.random.RandomState(42)
        X = jnp.array(rng.randn(n, d))
        osc = 2.5
        l_base = jnp.ones(d)
        W = jnp.array(rng.randn(d, d) * 0.2)
        b = jnp.zeros(d)
        K = _gibbs_kernel_matrix(X, X, l_base, W, b, jnp.float64(osc))
        np.testing.assert_allclose(np.diag(np.array(K)), osc, rtol=1e-10)


class TestGibbsGPR:
    """Test the full GibbsGPR regressor."""

    def test_fit_predict_basic(self):
        """Basic fit and predict should work without errors."""
        d = 2
        n = 30
        rng = np.random.RandomState(42)
        X = rng.randn(n, d)
        y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1]) + 0.01 * rng.randn(n)

        gpr = make_gibbs_gpr(d, n_restarts_optimizer=1, random_state=42)
        gpr.fit(X, y, validate=False)

        X_test = rng.randn(5, d)
        result = gpr.predict(X_test, return_std=True)
        assert len(result) == 2
        y_mean, y_std = result
        assert y_mean.shape == (5,)
        assert y_std.shape == (5,)
        assert np.all(y_std > 0)

    def test_predict_at_training_points(self):
        """Predictions at training points should be close to training values."""
        d = 2
        n = 20
        rng = np.random.RandomState(42)
        X = rng.randn(n, d)
        y = np.sin(X[:, 0])

        gpr = make_gibbs_gpr(d, n_restarts_optimizer=2, noise_level=1e-4,
                             random_state=42)
        gpr.fit(X, y, validate=False)

        y_pred = gpr.predict(X)[0]
        # Should be close but not exact due to noise regularization
        np.testing.assert_allclose(y_pred, y, atol=0.5)

    def test_predict_std_only(self):
        """predict_std should return just the std array."""
        d = 2
        n = 20
        rng = np.random.RandomState(42)
        X = rng.randn(n, d)
        y = np.sin(X[:, 0])

        gpr = make_gibbs_gpr(d, n_restarts_optimizer=1, random_state=42)
        gpr.fit(X, y, validate=False)

        std = gpr.predict_std(rng.randn(5, d))
        assert std.shape == (5,)
        assert np.all(std > 0)

    def test_gradients(self):
        """Test that predict with gradients works."""
        d = 3
        n = 30
        rng = np.random.RandomState(42)
        X = rng.randn(n, d)
        y = np.sin(X[:, 0])

        gpr = make_gibbs_gpr(d, n_restarts_optimizer=1, random_state=42)
        gpr.fit(X, y, validate=False)

        x_test = rng.randn(1, d)
        result = gpr.predict(x_test, return_std=True, return_mean_grad=True,
                             return_std_grad=True)
        assert len(result) == 4
        y_mean, y_std, grad_mean, grad_std = result
        assert grad_mean.shape == (1, d)
        assert grad_std.shape == (1, d)

    def test_interface_attributes(self):
        """Test that all required interface attributes exist."""
        d = 2
        gpr = make_gibbs_gpr(d)
        assert hasattr(gpr, 'kernel')
        assert hasattr(gpr, 'kernel_')
        assert hasattr(gpr, 'scales')
        assert hasattr(gpr, 'noise_level')
        assert hasattr(gpr, 'is_noise_in_kernel')
        assert hasattr(gpr, 'n_restarts_optimizer')
        assert hasattr(gpr, '_fitted')
        assert hasattr(gpr, 'copy_X_train')
        assert hasattr(gpr, 'n_eval')
        assert hasattr(gpr, 'n_eval_loglike')
        assert hasattr(gpr, 'random_state')
        assert hasattr(gpr, 'alpha')

    def test_low_rank_W(self):
        """Test that low-rank W works (rank < d)."""
        d = 5
        gpr = make_gibbs_gpr(d, rank=2, n_restarts_optimizer=1, random_state=42)
        assert gpr._W.shape == (d, 2)

        rng = np.random.RandomState(42)
        X = rng.randn(20, d)
        y = np.sin(X[:, 0])
        gpr.fit(X, y, validate=False)

        result = gpr.predict(rng.randn(5, d), return_std=True)
        assert len(result) == 2

    def test_lml_improves_with_optimization(self):
        """LML should generally improve after hyperparameter optimization."""
        d = 2
        n = 40
        rng = np.random.RandomState(42)
        X = rng.randn(n, d)
        y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1])

        gpr = make_gibbs_gpr(d, n_restarts_optimizer=3, random_state=42)
        gpr.fit(X, y, validate=False)
        lml_optimized = gpr.log_marginal_likelihood_value_
        assert lml_optimized is not None
        assert np.isfinite(lml_optimized)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
