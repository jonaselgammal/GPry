import time

import numpy as np
import pytest
from sklearn.svm import SVC

from gpry.infinities_classifier import ThresholdClassifier
from gpry.jax_svm import JaxBinaryRBFSVC


def _make_binary_problem():
    rng = np.random.default_rng(1234)
    X0 = rng.normal(loc=(-1.0, -0.7), scale=0.55, size=(18, 2))
    X1 = rng.normal(loc=(1.1, 0.8), scale=0.65, size=(22, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1))
    X_test = rng.normal(loc=0.1, scale=1.1, size=(25, 2))
    return X, y, X_test


def _make_threshold_problem():
    rng = np.random.default_rng(2026)
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


def test_gamma_scale_matches_sklearn():
    X, y, _ = _make_binary_problem()
    ref = SVC(kernel="rbf", gamma="scale", C=2.5).fit(X, y)
    clf = JaxBinaryRBFSVC(gamma="scale", C=2.5).fit(X, y)
    assert clf._gamma == pytest.approx(ref._gamma, rel=0, abs=1e-15)


def test_gamma_auto_matches_sklearn():
    X, y, _ = _make_binary_problem()
    ref = SVC(kernel="rbf", gamma="auto", C=2.5).fit(X, y)
    clf = JaxBinaryRBFSVC(gamma="auto", C=2.5).fit(X, y)
    assert clf._gamma == pytest.approx(ref._gamma, rel=0, abs=1e-15)


def test_public_decision_function_matches_manual_reconstruction():
    X, y, X_test = _make_binary_problem()
    clf = JaxBinaryRBFSVC(gamma="scale", C=3.0).fit(X, y)
    decision = clf.decision_function(X_test)

    sqdist = np.maximum(
        np.sum(X_test**2, axis=1, keepdims=True)
        + np.sum(clf.support_vectors_**2, axis=1)[None, :]
        - 2.0 * X_test @ clf.support_vectors_.T,
        0.0,
    )
    kernel = np.exp(-clf._gamma * sqdist)
    manual = kernel @ clf.dual_coef_[0] + clf.intercept_[0]
    np.testing.assert_allclose(decision, manual, rtol=1e-10, atol=1e-10)


def test_matches_sklearn_predictions_and_decision_function_on_unweighted_problem():
    X, y, X_test = _make_binary_problem()
    ref = SVC(kernel="rbf", gamma="scale", C=4.0, tol=1e-5).fit(X, y)
    clf = JaxBinaryRBFSVC(gamma="scale", C=4.0, tol=1e-5, max_iter=3000).fit(X, y)

    np.testing.assert_array_equal(clf.predict(X), ref.predict(X))
    np.testing.assert_array_equal(clf.predict(X_test), ref.predict(X_test))
    np.testing.assert_allclose(
        clf.decision_function(X),
        ref.decision_function(X),
        rtol=0,
        atol=5e-2,
    )
    np.testing.assert_allclose(
        clf.decision_function(X_test),
        ref.decision_function(X_test),
        rtol=0,
        atol=5e-2,
    )


def test_matches_sklearn_with_class_and_sample_weight():
    X, y, X_test = _make_binary_problem()
    sample_weight = np.linspace(0.8, 1.4, len(X))
    class_weight = {0: 0.7, 1: 1.6}
    ref = SVC(
        kernel="rbf",
        gamma=0.4,
        C=2.0,
        tol=1e-5,
        class_weight=class_weight,
    ).fit(X, y, sample_weight=sample_weight)
    clf = JaxBinaryRBFSVC(
        gamma=0.4,
        C=2.0,
        tol=1e-5,
        max_iter=4000,
        class_weight=class_weight,
    ).fit(X, y, sample_weight=sample_weight)

    np.testing.assert_array_equal(clf.predict(X), ref.predict(X))
    np.testing.assert_array_equal(clf.predict(X_test), ref.predict(X_test))
    np.testing.assert_allclose(
        clf.decision_function(X_test),
        ref.decision_function(X_test),
        rtol=0,
        atol=7e-2,
    )


def test_rejects_non_binary_target():
    X = np.eye(3)
    y = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="binary classification only"):
        JaxBinaryRBFSVC().fit(X, y)


def test_matches_sklearn_on_gpry_threshold_labels():
    X, y = _make_threshold_problem()
    i_sorted = np.argsort(y)
    finite_idx, used_threshold = ThresholdClassifier.i_finite_threshold(
        y,
        threshold=0.25,
        keep_min=8,
        i_sorted=i_sorted,
        validate=True,
    )
    labels = np.zeros(len(y), dtype=int)
    labels[finite_idx] = 1

    ref = SVC(kernel="rbf", gamma="scale", C=1e4, tol=1e-5).fit(X, labels)
    clf = JaxBinaryRBFSVC(gamma="scale", C=1e4, tol=1e-5, max_iter=4000).fit(X, labels)

    np.testing.assert_array_equal(clf.predict(X), ref.predict(X))
    np.testing.assert_allclose(
        clf.decision_function(X),
        ref.decision_function(X),
        rtol=0,
        atol=7e-2,
    )
    assert used_threshold >= 0.25
    assert labels.sum() >= 8


def test_handles_duplicate_points_with_soft_margin_overlap():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.1, 0.0],
            [0.9, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 1])
    ref = SVC(kernel="rbf", gamma=1.5, C=10.0, tol=1e-6).fit(X, y)
    clf = JaxBinaryRBFSVC(gamma=1.5, C=10.0, tol=1e-6, max_iter=5000).fit(X, y)

    np.testing.assert_array_equal(clf.predict(X), ref.predict(X))
    np.testing.assert_allclose(
        clf.decision_function(X),
        ref.decision_function(X),
        rtol=0,
        atol=7e-2,
    )


def test_all_one_class_after_threshold_is_rejected_cleanly():
    X = np.arange(10.0)[:, None]
    labels = np.ones(len(X), dtype=int)
    with pytest.raises(ValueError, match="binary classification only"):
        JaxBinaryRBFSVC().fit(X, labels)


def test_runtime_stays_reasonable_on_representative_small_problem():
    rng = np.random.default_rng(77)
    X0 = rng.normal(loc=-0.7, scale=0.8, size=(90, 4))
    X1 = rng.normal(loc=0.8, scale=0.9, size=(90, 4))
    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1))

    t0 = time.perf_counter()
    ref = SVC(kernel="rbf", gamma="scale", C=10.0, tol=1e-4).fit(X, y)
    sklearn_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    clf = JaxBinaryRBFSVC(gamma="scale", C=10.0, tol=1e-4, max_iter=6000).fit(X, y)
    jax_fit = time.perf_counter() - t0

    np.testing.assert_array_equal(clf.predict(X), ref.predict(X))
    # This is a smoke bound, not a benchmark claim.
    assert sklearn_fit < 0.2
    assert jax_fit < 1.0
