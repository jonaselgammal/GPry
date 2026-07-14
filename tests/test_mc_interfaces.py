"""
Regression tests for Lever 1 (gradient-based HMC/NUTS acquisition sampling):
  - the vectorized kernel gradient (``gradient_x_batch`` / ``predict_mean_grad_batch``)
    matches the existing single-point analytic gradient AND finite differences;
  - training-point-seeded HMC covers seeded modes;
  - the BlackJAX NUTS log-density matches an independent numpy reference and
    NUTS covers seeded modes (skipped if JAX/BlackJAX are unavailable).
"""
import numpy as np
import pytest

from gpry.gpr import GaussianProcessRegressor as GPR
from gpry.mc_interfaces import hmc_sample_gp_mean


def _fit(kernel, bounds, X, y, n_restarts=4):
    gpr = GPR(kernel=kernel, bounds=bounds, account_for_inf=None,
              n_restarts_optimizer=n_restarts)
    gpr.append_to_data(X, y, fit_gpr=True, fit_classifier=False)
    return gpr


@pytest.mark.parametrize("kernel", ["RBF", "Matern"])
def test_vectorized_gradient_matches_singlepoint_and_fd(kernel):
    rng = np.random.default_rng(0)
    d = 3
    bounds = np.array([[-3.0, 3.0]] * d)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(40, d))
    y = -0.5 * np.sum(X ** 2, axis=1) + np.sin(X[:, 0]) * X[:, 1]
    gpr = _fit(kernel, bounds, X, y)

    Xq = rng.uniform(bounds[:, 0], bounds[:, 1], size=(20, d))
    _, grad_b = gpr.predict_mean_grad_batch(Xq)

    # vs existing single-point predict(return_mean_grad=True)
    grad_sp = np.array([
        np.ravel(gpr.predict(x[None], return_std=True, return_mean_grad=True)[2])
        for x in Xq
    ])
    assert np.max(np.abs(grad_b - grad_sp)) < 1e-8

    # vs central finite differences of the batch mean
    eps = 1e-5
    grad_fd = np.empty_like(grad_b)
    for k in range(d):
        e = np.zeros(d); e[k] = eps
        mp, _ = gpr.predict_mean_grad_batch(Xq + e)
        mm, _ = gpr.predict_mean_grad_batch(Xq - e)
        grad_fd[:, k] = (mp - mm) / (2 * eps)
    assert np.max(np.abs(grad_b - grad_fd)) < 1e-4


def test_hmc_covers_both_seeded_modes():
    rng = np.random.default_rng(1)
    d = 2
    m1, m2, s = np.array([-2.0, -2.0]), np.array([2.0, 2.0]), 0.6
    bounds = np.array([[-4.0, 4.0]] * d)
    X = np.concatenate([
        m1 + 0.6 * rng.standard_normal((25, d)),
        m2 + 0.6 * rng.standard_normal((25, d)),
        rng.uniform(bounds[:, 0], bounds[:, 1], size=(10, d)),
    ])
    X = np.clip(X, bounds[:, 0], bounds[:, 1])

    def logp(P):
        c = -0.5 * d * np.log(2 * np.pi * s ** 2)
        l1 = c - 0.5 * np.sum((P - m1) ** 2, axis=1) / s ** 2
        l2 = c - 0.5 * np.sum((P - m2) ** 2, axis=1) / s ** 2
        return np.logaddexp(l1 + np.log(0.5), l2 + np.log(0.5))

    gpr = _fit("RBF", bounds, X, logp(X), n_restarts=6)
    res = hmc_sample_gp_mean(gpr, bounds[:, 0], bounds[:, 1],
                             rng=np.random.default_rng(2),
                             n_warmup=40, n_samples=100, thin=4, n_leapfrog=15)
    S = res["X"]
    f1 = np.mean(np.linalg.norm(S - m1, axis=1) < 1.2)
    f2 = np.mean(np.linalg.norm(S - m2, axis=1) < 1.2)
    assert f1 > 0.15 and f2 > 0.15, f"mode coverage too low: {f1:.2f}, {f2:.2f}"


def _bimodal_gpr(rng):
    d = 2
    m1, m2, s = np.array([-2.0, -2.0]), np.array([2.0, 2.0]), 0.6
    bounds = np.array([[-4.0, 4.0]] * d)
    X = np.clip(np.concatenate([
        m1 + 0.6 * rng.standard_normal((25, d)),
        m2 + 0.6 * rng.standard_normal((25, d)),
        rng.uniform(bounds[:, 0], bounds[:, 1], size=(10, d)),
    ]), bounds[:, 0], bounds[:, 1])

    def logp(P):
        c = -0.5 * d * np.log(2 * np.pi * s ** 2)
        l1 = c - 0.5 * np.sum((P - m1) ** 2, axis=1) / s ** 2
        l2 = c - 0.5 * np.sum((P - m2) ** 2, axis=1) / s ** 2
        return np.logaddexp(l1 + np.log(0.5), l2 + np.log(0.5))

    return _fit("RBF", bounds, X, logp(X), n_restarts=6), bounds, m1, m2


def test_nuts_logdensity_matches_numpy_reference():
    pytest.importorskip("blackjax")
    import jax
    jax.config.update("jax_enable_x64", True)
    from gpry import mc_interfaces as mci

    rng = np.random.default_rng(3)
    gpr, bounds, _, _ = _bimodal_gpr(rng)
    lo, hi = bounds[:, 0], bounds[:, 1]
    logdensity_fn, _, _ = mci._build_jax_logdensity(gpr, lo, hi, beta=1.0)
    std_y = float(np.ravel(gpr.preprocessing_y.inverse_transform_scale(np.ones(1)))[0])
    alpha = np.ravel(np.asarray(gpr.alpha_))
    for _ in range(15):
        u = rng.uniform(-3, 3, size=2)
        s = 1.0 / (1.0 + np.exp(-u))
        xn = lo + (hi - lo) * s
        ref = std_y * float((gpr.kernel_(xn[None], gpr.X_train_) @ alpha).ravel()[0]) \
            + np.sum(np.log(hi - lo) + np.log(s) + np.log1p(-s))
        assert abs(ref - float(logdensity_fn(np.asarray(u)))) < 1e-6


def test_nuts_covers_both_seeded_modes():
    pytest.importorskip("blackjax")
    from gpry import mc_interfaces as mci

    rng = np.random.default_rng(4)
    gpr, bounds, m1, m2 = _bimodal_gpr(rng)
    res = mci.nuts_sample_gp_mean(gpr, bounds[:, 0], bounds[:, 1],
                                  rng=np.random.default_rng(5), max_chains=50,
                                  n_warmup=120, n_samples=120)
    S = res["X"]
    f1 = np.mean(np.linalg.norm(S - m1, axis=1) < 1.2)
    f2 = np.mean(np.linalg.norm(S - m2, axis=1) < 1.2)
    assert f1 > 0.15 and f2 > 0.15, f"NUTS mode coverage too low: {f1:.2f}, {f2:.2f}"
    assert res["divergences"] == 0
