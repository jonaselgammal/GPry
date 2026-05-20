"""
Cross-backend parity tests between the sklearn-numpy and JAX GPR backends.

This module pins the public-surface contract that the JAX-split refactor
(see ``AGENTS/jax_split_refactor_plan.md``) must preserve. Each test creates two
``GaussianProcessRegressor`` instances with identical configuration but
different ``use_jax`` flags, fits them on the same data, and asserts agreement
on the public methods.

Tolerances follow the plan (Phase 0.2):

- ``predict`` mean: ``atol=1e-6``.
- ``predict_std``, ``predict(return_std=True)`` std: ``atol=1e-5``.
- ``log_marginal_likelihood`` value: ``atol=1e-6``.
- LML gradient: ``rtol=1e-4, atol=1e-5`` componentwise.
- Hyperopt LML at the optimum: ``atol=1e-5``. (We do **not** compare ``theta`` at
  the optimum — scipy ``fmin_l_bfgs_b`` and ``jaxopt.LBFGSB`` use different
  stopping rules and won't agree on θ to better than ~1e-2.)
"""

import os
import sys

import numpy as np
import pytest
from sklearn.base import clone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry.gpr import GaussianProcessRegressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _length_scale_prior(n_dims, bounds_lo=-5.0, bounds_hi=5.0,
                        rel_lo=1e-3, rel_hi=1e1):
    """Per-dim length-scale prior in absolute units."""
    span = bounds_hi - bounds_lo
    return np.column_stack(
        [np.full(n_dims, span * rel_lo), np.full(n_dims, span * rel_hi)]
    )


def _make_gpr(*, n_dims, use_jax, n_hyperopt_restarts=0, kernel="RBF",
              noise_level=0.1, random_state=42):
    """Construct a single GPR with the standard parity-test configuration."""
    return GaussianProcessRegressor(
        kernel=kernel,
        output_scale_prior=[1e-2, 1e3],
        length_scale_prior=_length_scale_prior(n_dims),
        noise_level=noise_level,
        noise_fixed=True,
        n_hyperopt_restarts=n_hyperopt_restarts,
        random_state=random_state,
        use_jax=use_jax,
    )


def _make_paired_gprs(*, n_dims, n_hyperopt_restarts=0, kernel="RBF",
                     noise_level=0.1, random_state=42):
    """A (jax, sklearn) pair of GPRs with otherwise identical configuration."""
    gpr_jax = _make_gpr(
        n_dims=n_dims,
        use_jax=True,
        n_hyperopt_restarts=n_hyperopt_restarts,
        kernel=kernel,
        noise_level=noise_level,
        random_state=random_state,
    )
    gpr_sk = _make_gpr(
        n_dims=n_dims,
        use_jax=False,
        n_hyperopt_restarts=n_hyperopt_restarts,
        kernel=kernel,
        noise_level=noise_level,
        random_state=random_state,
    )
    return gpr_jax, gpr_sk


def _fit_no_hyperopt(gpr, X, y):
    """Fit without running hyperparameter optimization; matches surrogate's
    ``fit_hyperparameters=False`` path. Both backends share the same θ this way,
    so all predict/LML comparisons are at identical hyperparameters."""
    if gpr.fitted_kernel is None:
        # Known initialization quirk for the standalone GPR (see memory note
        # ``feedback_gpry_gpr_init.md``). The Surrogate handles this in normal
        # use; tests that bypass the Surrogate need to do it explicitly.
        gpr.fitted_kernel = clone(gpr.kernel)
    gpr.fit(X, y, fit_hyperparameters=False, validate=False)
    return gpr


def _make_dataset(n_dims, n_train=40, n_test=12, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.uniform(-2.0, 2.0, size=(n_train, n_dims))
    y_train = (
        np.sum(np.sin(X_train), axis=1)
        + 0.05 * rng.standard_normal(n_train)
    )
    X_test = rng.uniform(-2.0, 2.0, size=(n_test, n_dims))
    return X_train, y_train, X_test


# ---------------------------------------------------------------------------
# Public-surface parity at fixed hyperparameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["RBF", {"Matern": {"nu": 2.5}}])
@pytest.mark.parametrize("n_dims", [2, 4])
def test_predict_mean_parity(kernel, n_dims):
    """``predict(X)`` returns the same mean across backends to 1e-6."""
    gpr_jax, gpr_sk = _make_paired_gprs(n_dims=n_dims, kernel=kernel)
    X_train, y_train, X_test = _make_dataset(n_dims, seed=1)
    _fit_no_hyperopt(gpr_jax, X_train, y_train)
    _fit_no_hyperopt(gpr_sk, X_train, y_train)

    mu_jax = gpr_jax.predict(X_test, validate=False)
    mu_sk = gpr_sk.predict(X_test, validate=False)
    np.testing.assert_allclose(mu_jax, mu_sk, atol=1e-6, rtol=0)


@pytest.mark.parametrize("kernel", ["RBF", {"Matern": {"nu": 2.5}}])
@pytest.mark.parametrize("n_dims", [2, 4])
def test_predict_mean_and_std_parity(kernel, n_dims):
    """``predict(X, return_std=True)`` agrees on both mean and std."""
    gpr_jax, gpr_sk = _make_paired_gprs(n_dims=n_dims, kernel=kernel)
    X_train, y_train, X_test = _make_dataset(n_dims, seed=2)
    _fit_no_hyperopt(gpr_jax, X_train, y_train)
    _fit_no_hyperopt(gpr_sk, X_train, y_train)

    mu_jax, std_jax = gpr_jax.predict(X_test, return_std=True, validate=False)
    mu_sk, std_sk = gpr_sk.predict(X_test, return_std=True, validate=False)
    np.testing.assert_allclose(mu_jax, mu_sk, atol=1e-6, rtol=0,
                               err_msg="mean mismatch")
    np.testing.assert_allclose(std_jax, std_sk, atol=1e-5, rtol=0,
                               err_msg="std mismatch")


@pytest.mark.parametrize("kernel", ["RBF", {"Matern": {"nu": 2.5}}])
@pytest.mark.parametrize("n_dims", [2, 4])
def test_predict_std_parity(kernel, n_dims):
    """``predict_std(X)`` agrees across backends to 1e-5."""
    gpr_jax, gpr_sk = _make_paired_gprs(n_dims=n_dims, kernel=kernel)
    X_train, y_train, X_test = _make_dataset(n_dims, seed=3)
    _fit_no_hyperopt(gpr_jax, X_train, y_train)
    _fit_no_hyperopt(gpr_sk, X_train, y_train)

    std_jax = gpr_jax.predict_std(X_test, validate=False)
    std_sk = gpr_sk.predict_std(X_test, validate=False)
    np.testing.assert_allclose(std_jax, std_sk, atol=1e-5, rtol=0)


@pytest.mark.parametrize("kernel", ["RBF", {"Matern": {"nu": 2.5}}])
@pytest.mark.parametrize("n_dims", [2, 4])
def test_log_marginal_likelihood_value_parity(kernel, n_dims):
    """``log_marginal_likelihood(theta)`` agrees at a fixed θ to 1e-6.

    Note: pre-Phase-2, both backends ultimately route the no-gradient LML
    through sklearn (``gpr_jax.py:233``), so this is currently close to a
    tautology. It will become a real parity check once Phase 2 reimplements
    LML natively in both backends.
    """
    gpr_jax, gpr_sk = _make_paired_gprs(n_dims=n_dims, kernel=kernel)
    X_train, y_train, _ = _make_dataset(n_dims, seed=4)
    _fit_no_hyperopt(gpr_jax, X_train, y_train)
    _fit_no_hyperopt(gpr_sk, X_train, y_train)

    theta = gpr_sk.fitted_kernel.theta.copy()
    lml_jax = gpr_jax.log_marginal_likelihood(theta, eval_gradient=False)
    lml_sk = gpr_sk.log_marginal_likelihood(theta, eval_gradient=False)
    np.testing.assert_allclose(lml_jax, lml_sk, atol=1e-6, rtol=0)


@pytest.mark.parametrize("kernel", ["RBF", {"Matern": {"nu": 2.5}}])
@pytest.mark.parametrize("n_dims", [2, 4])
def test_log_marginal_likelihood_gradient_parity(kernel, n_dims):
    """``log_marginal_likelihood(theta, eval_gradient=True)`` value and grad agree.

    JAX returns autodiff gradients; sklearn returns analytic gradients.
    They should agree to ~1e-4 relative (kernel-second-derivative chain).
    """
    gpr_jax, gpr_sk = _make_paired_gprs(n_dims=n_dims, kernel=kernel)
    X_train, y_train, _ = _make_dataset(n_dims, seed=5)
    _fit_no_hyperopt(gpr_jax, X_train, y_train)
    _fit_no_hyperopt(gpr_sk, X_train, y_train)

    theta = gpr_sk.fitted_kernel.theta.copy()
    lml_jax, grad_jax = gpr_jax.log_marginal_likelihood(theta, eval_gradient=True)
    lml_sk, grad_sk = gpr_sk.log_marginal_likelihood(theta, eval_gradient=True)
    np.testing.assert_allclose(lml_jax, lml_sk, atol=1e-6, rtol=0,
                               err_msg="LML value mismatch")
    np.testing.assert_allclose(grad_jax, grad_sk, atol=1e-5, rtol=1e-4,
                               err_msg="LML gradient mismatch")


# ---------------------------------------------------------------------------
# Hyperopt parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["RBF"])
@pytest.mark.parametrize("n_dims", [2, 4])
def test_hyperopt_lml_at_optimum_parity(kernel, n_dims):
    """Run a small hyperopt with one restart and a fixed seed on both backends.

    Compare the *LML at the optimum* (not theta itself) to 1e-5. This catches
    regressions in the optimizer plumbing or in the LML implementation without
    requiring scipy and jaxopt to agree on the literal θ chosen.
    """
    gpr_jax, gpr_sk = _make_paired_gprs(
        n_dims=n_dims, kernel=kernel, n_hyperopt_restarts=1, random_state=42,
    )
    X_train, y_train, _ = _make_dataset(n_dims, seed=6)
    if gpr_jax.fitted_kernel is None:
        gpr_jax.fitted_kernel = clone(gpr_jax.kernel)
    if gpr_sk.fitted_kernel is None:
        gpr_sk.fitted_kernel = clone(gpr_sk.kernel)
    gpr_jax.fit(X_train, y_train, validate=False)
    gpr_sk.fit(X_train, y_train, validate=False)

    lml_at_opt_jax = gpr_jax.log_marginal_likelihood_value_
    lml_at_opt_sk = gpr_sk.log_marginal_likelihood_value_
    np.testing.assert_allclose(
        lml_at_opt_jax, lml_at_opt_sk, atol=1e-5, rtol=0,
        err_msg=(
            f"Hyperopt LML at the optimum disagrees: jax={lml_at_opt_jax}, "
            f"sklearn={lml_at_opt_sk}"
        ),
    )
