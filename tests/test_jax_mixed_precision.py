"""Mixed-precision (float32) predictive-variance kernel (speed-levers Step 4).

The mixed kernel runs the expensive ``V @ K_trans.T`` matmul in float32 (the
GPU-accelerable part) and the cancellation-prone ``k_diag - sum(.**2)``
reduction in float64, with a per-batch float64 guard. These tests assert it is
(a) safe -- never produces negative / NaN variances, the failure mode of
full-float32 -- and (b) accurate -- equal to the full-float64 result wherever
the guard fires, and within float32 matmul tolerance elsewhere. The feature is
opt-in (``predict_f32``); the default path is unchanged (covered by the rest of
the JAX suite).
"""
import os
import sys

import numpy as np
from sklearn.base import clone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry.gpr import GaussianProcessRegressor, JaxGaussianProcessRegressor


def _fit_gpr(X, y, bounds, noise, predict_f32):
    lsp = np.column_stack([bounds[:, 1] - bounds[:, 0]] * 2) * np.array([[1e-3, 1e1]])
    g = GaussianProcessRegressor(
        kernel="RBF", output_scale_prior=[1e-2, 1e3], length_scale_prior=lsp,
        noise_level=noise, noise_fixed=True, n_hyperopt_restarts=1,
        random_state=0, use_jax=True, predict_f32=predict_f32,
    )
    assert isinstance(g, JaxGaussianProcessRegressor)
    g.fitted_kernel = clone(g.kernel)
    g.fit(X, y, validate=False)
    return g


def _make_data(n, d, noise, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, (n, d))
    y = np.sum(np.sin(X), axis=1) + noise * rng.randn(n)
    return X, y, np.array([[-2.0, 2.0]] * d)


def _test_points(X, d, q=200, seed=1):
    # Half near training points (variance ~0, where f32 cancellation bites),
    # half uniformly random.
    rng = np.random.RandomState(seed)
    near = X[rng.randint(0, len(X), q // 2)] + 0.01 * rng.randn(q // 2, d)
    return np.vstack([near, rng.uniform(-2, 2, (q - q // 2, d))])


def test_predict_f32_defaults_off():
    g = GaussianProcessRegressor(
        kernel="RBF", output_scale_prior=[1e-2, 1e3],
        length_scale_prior=np.array([[1e-3, 1e1]]), noise_level=0.1,
        noise_fixed=True, random_state=0, use_jax=True,
    )
    assert g._predict_f32 is False


def test_mixed_precision_safe_regime_matches_f64():
    """d=8, realistic noise: guard does not fire, mixed ~ f64, no negatives."""
    X, y, bounds = _make_data(400, 8, 0.1)
    g64 = _fit_gpr(X, y, bounds, 0.1, predict_f32=False)
    g32 = _fit_gpr(X, y, bounds, 0.1, predict_f32=True)
    Xq = _test_points(X, 8)
    s64 = np.asarray(g64.predict_std(Xq, validate=False))
    s32 = np.asarray(g32.predict_std(Xq, validate=False))
    assert np.all(np.isfinite(s32)) and np.all(s32 >= 0.0)
    # float32 matmul round-off only.
    np.testing.assert_allclose(s32, s64, atol=1e-3)


def test_mixed_precision_guard_no_negative_variance():
    """d=2, tiny noise: full-f32 would give negative variances near training
    points; the mixed guard must keep std finite, non-negative and (here)
    equal to f64 because the whole batch falls back to f64."""
    X, y, bounds = _make_data(800, 2, 1e-4)
    g64 = _fit_gpr(X, y, bounds, 1e-4, predict_f32=False)
    g32 = _fit_gpr(X, y, bounds, 1e-4, predict_f32=True)
    Xq = _test_points(X, 2)
    s64 = np.asarray(g64.predict_std(Xq, validate=False))
    s32 = np.asarray(g32.predict_std(Xq, validate=False))
    assert np.all(np.isfinite(s32)), "mixed-precision guard failed: NaN/inf std"
    assert np.all(s32 >= 0.0), "mixed-precision guard failed: negative variance"
    np.testing.assert_allclose(s32, s64, atol=1e-6)


def test_mixed_precision_mean_and_var_path():
    """The combined predict (mean+std) path also honours the flag and stays
    finite / non-negative."""
    X, y, bounds = _make_data(300, 5, 0.05)
    g32 = _fit_gpr(X, y, bounds, 0.05, predict_f32=True)
    g64 = _fit_gpr(X, y, bounds, 0.05, predict_f32=False)
    Xq = _test_points(X, 5)
    m32, s32 = g32.predict(Xq, return_std=True, validate=False)
    m64, s64 = g64.predict(Xq, return_std=True, validate=False)
    s32, s64 = np.asarray(s32), np.asarray(s64)
    assert np.all(np.isfinite(s32)) and np.all(s32 >= 0.0)
    # Mean path is pure f64 in both -> must match tightly.
    np.testing.assert_allclose(np.asarray(m32), np.asarray(m64), atol=1e-6)
    np.testing.assert_allclose(s32, s64, atol=1e-3)
