import os
import sys

import numpy as np
import pytest

import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry.infinities_classifier import InfinitiesClassifiers, SVM, ThresholdClassifier
from gpry.acquisition_functions import LogExp
from gpry.gp_acquisition import BatchOptimizer
from gpry.preprocessing import DummyPreprocessor
from gpry.surrogate import SurrogateModel


def _make_threshold_problem():
    rng = np.random.default_rng(2027)
    X = rng.normal(size=(60, 3))
    y = np.array(
        [
            -np.inf,
            -8.0,
            -7.1,
            -6.0,
            -5.8,
            -5.2,
            -4.0,
            -3.5,
            -3.0,
            -2.5,
            -2.4,
            -2.2,
            -2.1,
            -1.8,
            -1.6,
            -1.2,
            -1.0,
            -0.9,
            -0.7,
            -0.4,
            -0.1,
            0.0,
            0.1,
            0.3,
            0.5,
            0.8,
            1.0,
            1.15,
            1.22,
            1.3,
            1.35,
            1.4,
            1.42,
            1.45,
            1.48,
            1.49,
            1.5,
            1.51,
            1.52,
            1.53,
            1.54,
            1.545,
            1.55,
            1.555,
            1.56,
            1.57,
            1.58,
            1.585,
            1.59,
            1.595,
            1.6,
            1.605,
            1.61,
            1.615,
            1.62,
            1.625,
            1.63,
            1.635,
            1.64,
            1.645,
        ]
    )
    return X, y


def _make_separable_threshold_problem():
    rng = np.random.default_rng(41)
    X0 = rng.normal(loc=(-1.0, -1.0), scale=0.15, size=(24, 2))
    X1 = rng.normal(loc=(1.0, 1.0), scale=0.15, size=(24, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.linspace(-6.0, -2.0, len(X0)), np.linspace(0.2, 1.8, len(X1))])
    return X, y


class DummyPreprocessorWithFitState(DummyPreprocessor):
    fitted = False


def _make_surrogate(svm_backend="jax"):
    surrogate = SurrogateModel(
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        preprocessing_X=DummyPreprocessorWithFitState,
        preprocessing_y=DummyPreprocessorWithFitState,
        regressor={
            "kernel": "RBF",
            "use_jax": True,
            "noise_level": 0.01,
            "length_scale_prior": [0.01, 100.0],
            "output_scale_prior": [0.01, 100.0],
            "n_hyperopt_restarts": 0,
        },
        infinities_classifier={"svm": {"threshold": 0.3, "backend": svm_backend}},
        verbose=0,
    )
    X = np.array(
        [
            [0.10, 0.10],
            [0.15, 0.20],
            [0.20, 0.10],
            [0.25, 0.15],
            [0.65, 0.70],
            [0.72, 0.68],
            [0.78, 0.76],
            [0.85, 0.82],
        ]
    )
    y = np.array([-5.0, -4.5, -4.0, -3.8, -0.15, -0.05, 0.0, -0.08])
    surrogate.append(X, y, fit_gpr={"n_restarts": 1}, fit_classifier=True)
    return surrogate


def test_svm_backend_parity_on_threshold_problem():
    X, y = _make_separable_threshold_problem()
    i_sorted = np.argsort(y)

    svm_sk = SVM(
        threshold=1.7,
        nstd_calculator=lambda x: x,
        backend="sklearn",
        C=1e4,
        tol=1e-5,
    )
    svm_jx = SVM(
        threshold=1.7,
        nstd_calculator=lambda x: x,
        backend="jax",
        C=1e4,
        tol=1e-5,
    )
    idx_sk = svm_sk.fit(X, y, keep_min=6, i_sorted=i_sorted, validate=True)
    idx_jx = svm_jx.fit(X, y, keep_min=6, i_sorted=i_sorted, validate=True)

    np.testing.assert_array_equal(idx_sk, idx_jx)
    np.testing.assert_array_equal(svm_sk.predict(X), svm_jx.predict(X))

    Xq = np.linspace(-2.0, 2.0, 60).reshape(30, 2)
    np.testing.assert_array_equal(svm_sk.predict(Xq), svm_jx.predict(Xq))


def test_svm_threshold_inf_allfinite_native_path():
    X = np.array([[0.0, 0.0], [0.5, 0.4], [0.8, 0.7]])
    y = np.array([-2.0, -1.0, -0.5])
    svm = SVM(threshold=np.inf, nstd_calculator=lambda x: x, backend="jax")
    svm.fit(X, y, validate=True)
    native = np.asarray(svm.predict_native(jnp.asarray(X)))
    np.testing.assert_array_equal(native, np.ones(len(X), dtype=bool))


def test_infinities_classifiers_native_matches_numpy():
    X, y = _make_threshold_problem()
    clf = InfinitiesClassifiers(
        bounds=np.array([[-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]]),
        nstd_calculator=lambda x: x,
        trust_region={"threshold": 0.25},
        svm={"threshold": 0.25, "backend": "jax", "C": 1e4, "tol": 1e-5},
    )
    clf.fit(X, y, keep_min=8, validate=True)

    Xq = np.vstack([X[:20], np.full((4, 3), 4.0)])
    finite_np = clf.is_finite_X(Xq, validate=True)
    finite_jx = np.asarray(clf.is_finite_X_native(jnp.asarray(Xq)))
    np.testing.assert_array_equal(finite_np, finite_jx)


def test_surrogate_predict_transformed_native_matches_public():
    surrogate = _make_surrogate(svm_backend="jax")
    Xq = np.array(
        [
            [0.12, 0.12],
            [0.20, 0.18],
            [0.75, 0.74],
            [0.95, 0.05],
        ]
    )
    pred_np = surrogate.predict_transformed(Xq, return_std=False, validate=False)
    pred_jx = np.asarray(
        surrogate.predict_transformed_native(Xq, return_std=False)
    )
    np.testing.assert_allclose(pred_jx, pred_np, atol=1e-10)


def test_surrogate_native_adapter_matches_public_pointwise():
    surrogate = _make_surrogate(svm_backend="jax")

    def logp(x):
        return surrogate.predict_transformed(np.asarray(x)[None, :], validate=False)[0]

    adapter = surrogate.make_ns_loglikelihood_adapter(logp)
    assert adapter is not None
    jax_logp = adapter.build_jax_loglikelihood(["x0", "x1"])

    points = np.array(
        [
            [0.16, 0.14],
            [0.70, 0.72],
            [0.98, 0.02],
        ]
    )
    for point in points:
        params = {"x0": point[0], "x1": point[1]}
        native = float(jax_logp(params))
        public = float(surrogate.predict_transformed(point[None, :], validate=False)[0])
        assert native == pytest.approx(public, abs=1e-10)


def test_surrogate_native_adapter_absent_with_sklearn_svm():
    surrogate = _make_surrogate(svm_backend="sklearn")

    def logp(x):
        return surrogate.predict_transformed(np.asarray(x)[None, :], validate=False)[0]

    assert surrogate.make_ns_loglikelihood_adapter(logp) is None


def test_surrogate_native_acquisition_objective_matches_public_and_vetoes():
    surrogate = _make_surrogate(svm_backend="jax")
    acq_func = LogExp(dimension=2, zeta=0.5)
    native = surrogate.make_native_acquisition_objective(acq_func)
    assert native is not None

    good = np.array([0.70, 0.72])
    grid = np.array(
        [[x0, x1] for x0 in np.linspace(0.0, 1.0, 11) for x1 in np.linspace(0.0, 1.0, 11)]
    )
    finite = surrogate.infinities_classifier.is_finite_X(grid, validate=False)
    bad = grid[np.flatnonzero(~finite)[0]]

    native_good = float(native(jnp.asarray(good)))
    native_bad = float(native(jnp.asarray(bad)))
    public_good = float(-acq_func(good[None, :], surrogate, validate=False)[0])
    public_bad = float(-acq_func(bad[None, :], surrogate, validate=False)[0])

    assert native_good == pytest.approx(public_good, abs=1e-10)
    assert np.isinf(public_bad)
    assert native_bad > 1e20


def test_batch_optimizer_native_path_uses_surrogate_objective_not_gp(monkeypatch):
    surrogate = _make_surrogate(svm_backend="jax")
    optimizer = BatchOptimizer(
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        acq_func={"LogExp": {"zeta": 0.5}},
        verbose=0,
    )

    def _fail(*args, **kwargs):
        raise AssertionError("GP-native acquisition optimizer should not be used")

    monkeypatch.setattr(surrogate.gpr, "optimize_acquisition_native", _fail)
    optimizer.optimize_acquisition_function(
        surrogate,
        i=0,
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        rng=np.random.default_rng(0),
    )
