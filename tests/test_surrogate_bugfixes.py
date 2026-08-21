"""
Regression tests for small, self-contained bugs in the surrogate/preprocessing stack.

Deliberately free of any ``cobaya`` import (directly or via ``model_generator``), so that
this module collects and runs in a bare environment.
"""

import numpy as np
import pytest

from gpry.preprocessing import (
    DummyPreprocessor,
    NormalizeBounds,
    NormalizeY,
    PipelineX,
    Whitening,
)
from gpry.surrogate import SurrogateModel

def regressor():
    """
    Full regressor spec, as a fresh dict per call.

    ``SurrogateModel`` has no defaults for this sub-dict (those live in
    ``run.Runner._construct_surrogate``), so a partial dict raises ``KeyError``. It also
    modifies the dict in place (e.g. it unfolds ``length_scale_prior`` per dimension), so
    a shared one must not be reused across surrogates of different dimensionality.
    """
    return {
        "kernel": "RBF",
        "output_scale_prior": [1e-2, 1e3],
        "length_scale_prior": [1e-3, 1e2],
        "noise_level": 1e-2,
        "optimizer": "fmin_l_bfgs_b",
        "n_restarts_optimizer": 4,
    }


def svm_classifier():
    """Infinities-classifier spec, as a fresh dict per call (also modified in place)."""
    return {"svm": {"threshold": "20s"}}


def gaussian_training_set(d=4, n=100, seed=0, half_width=4.0):
    """Returns ``(bounds, X, y)`` for a standard normal log-posterior."""
    rng = np.random.default_rng(seed)
    bounds = np.array([[-half_width, half_width]] * d)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, d))
    y = -0.5 * np.sum(X**2, axis=1)
    return bounds, X, y


def test_is_finite_X_uses_classifier_is_finite_X():
    """
    ``SurrogateModel.is_finite_X`` used to call ``infinities_classifier.predict``, which
    does not exist on the ``InfinitiesClassifiers`` container, so it raised
    ``AttributeError`` unconditionally whenever a classifier was configured.
    """
    d = 4
    bounds, X, y = gaussian_training_set(d=d)
    surrogate = SurrogateModel(
        bounds=bounds,
        preprocessing_X=NormalizeBounds(bounds),
        preprocessing_y=NormalizeY(),
        regressor=regressor(),
        clip_factor=1.1,
        infinities_classifier=svm_classifier(),
        random_state=42,
        verbose=1,
    )
    surrogate.append(X, y)
    rng = np.random.default_rng(1)
    X_test = rng.uniform(bounds[:, 0], bounds[:, 1], size=(10, d))
    is_finite = surrogate.is_finite_X(X_test)
    assert isinstance(is_finite, np.ndarray)
    assert is_finite.dtype == bool
    assert is_finite.shape == (len(X_test),)
    # It must agree with the classifier's own method on the transformed points.
    expected = surrogate.infinities_classifier.is_finite_X(
        surrogate.preprocessing_X.transform(X_test)
    )
    assert np.array_equal(is_finite, np.asarray(expected, dtype=bool))


def test_dummy_preprocessor_is_fitted_on_the_class():
    """
    ``DummyPreprocessor`` is used un-instantiated (all its methods are classmethods), so
    ``fitted`` has to be readable on the class itself.
    """
    from gpry.preprocessing import DummyPreprocessor

    assert DummyPreprocessor.fitted is True
    assert DummyPreprocessor().fitted is True


def test_surrogate_without_preprocessing_y():
    """
    With ``preprocessing_y=None`` the surrogate falls back to ``DummyPreprocessor``, which
    used to lack a ``fitted`` attribute, so ``SurrogateModel.__init__`` raised
    ``AttributeError`` on the ``if self.preprocessing_y.fitted`` guard whenever an
    infinities classifier was configured.
    """
    d = 3
    bounds, X, y = gaussian_training_set(d=d, n=80, seed=2)
    surrogate = SurrogateModel(
        bounds=bounds,
        regressor=regressor(),
        infinities_classifier=svm_classifier(),
        random_state=42,
    )
    # The guard must have selected the dummy preprocessor itself.
    from gpry.preprocessing import DummyPreprocessor

    assert surrogate.preprocessing_y is DummyPreprocessor
    surrogate.append(X, y)
    rng = np.random.default_rng(3)
    X_test = rng.uniform(bounds[:, 0], bounds[:, 1], size=(10, d))
    y_pred = surrogate.predict(X_test)
    assert y_pred.shape == (len(X_test),)
    assert np.all(np.isfinite(y_pred))


def test_whitening_compute_mean_cov_orientation():
    """
    ``Whitening.compute_mean_cov`` used to call ``np.cov`` without ``rowvar=False``, i.e.
    treating rows of ``X`` as variables, while the ``mean`` line on the preceding line
    treats them as samples. For ``n_samples != n_features`` that raises; for
    ``n_samples == n_features`` it would silently return a meaningless matrix.
    """
    d, n = 8, 20000
    X = np.random.default_rng(0).normal(size=(n, d))
    logp = -0.5 * np.sum(X**2, axis=1)
    mean, cov = Whitening.compute_mean_cov(X, logp)
    assert mean.shape == (d,)
    assert cov.shape == (d, d)
    assert np.allclose(cov, cov.T)
    # Symmetric positive-definite: Cholesky succeeds.
    np.linalg.cholesky(cov)
    # Standard normal samples weighted by exp(-|x|^2/2) are distributed as N(0, I/2).
    assert np.allclose(np.diag(cov), 0.5, atol=0.1)


def test_whitening_compute_mean_cov_square_case():
    """``n_samples == n_features`` used to silently return an unrelated matrix."""
    d = 30
    X = np.random.default_rng(1).normal(size=(d, d))
    logp = -0.5 * np.sum(X**2, axis=1)
    _, cov = Whitening.compute_mean_cov(X, logp)
    assert cov.shape == (d, d)
    expected = np.cov(X, rowvar=False, aweights=np.exp(logp - np.max(logp)), ddof=0)
    assert np.allclose(cov, expected)


def test_whitening_in_surrogate_pipeline():
    """
    End-to-end: exercise ``Whitening`` through a ``PipelineX`` inside a
    ``SurrogateModel``. ``Whitening`` is opt-in and unreferenced by ``run.py`` and
    ``surrogate.py``, which is why the covariance-orientation bug survived.
    """
    d = 4
    bounds, X, y = gaussian_training_set(d=d, n=120, seed=4, half_width=3.0)
    surrogate = SurrogateModel(
        bounds=bounds,
        preprocessing_X=PipelineX([NormalizeBounds(bounds), Whitening(bounds, learn=True)]),
        preprocessing_y=NormalizeY(),
        regressor=regressor(),
        infinities_classifier=svm_classifier(),
        random_state=42,
    )
    surrogate.append(X, y)
    # The whitening step must actually have learnt a covariance.
    whitening = surrogate.preprocessing_X.preprocessors[-1]
    assert whitening.cov is not None
    assert np.asarray(whitening.cov).shape == (d, d)
    rng = np.random.default_rng(5)
    X_test = rng.uniform(bounds[:, 0], bounds[:, 1], size=(10, d))
    y_pred = surrogate.predict(X_test)
    assert y_pred.shape == (len(X_test),)
    assert np.all(np.isfinite(y_pred))


def test_surrogate_without_preprocessing_y_or_classifier():
    """
    With neither a ``preprocessing_y`` nor an infinities classifier, the local
    y-preprocessor handle used to transform the noise level was never assigned — it was
    only set inside the infinities-classifier branch — so it stayed at the raw `None`
    argument and `__init__` raised `AttributeError`.
    """
    d = 4
    bounds, X, y = gaussian_training_set(d=d, n=80, seed=6)
    surrogate = SurrogateModel(bounds=bounds, regressor=regressor(), random_state=42)
    assert surrogate.infinities_classifier is None
    assert surrogate.preprocessing_y is DummyPreprocessor
    surrogate.append(X, y)
    assert surrogate.n_total == len(X)
    y_pred = surrogate.predict(X[:5])
    assert y_pred.shape == (5,)
    assert np.all(np.isfinite(y_pred))


@pytest.mark.parametrize(
    "infinities_classifier", [None, {"svm": {"threshold": "20s"}}]
)
def test_predict_without_infinities_classifier(infinities_classifier):
    """
    `predict` and `predict_std` dereferenced `self.infinities_classifier` without the
    `is None` guard that `is_finite_X` already had, so every prediction entry point
    raised `AttributeError` on a surrogate built without a classifier.
    """
    d = 4
    bounds, X, y = gaussian_training_set(d=d, n=120, seed=7, half_width=3.0)
    surrogate = SurrogateModel(
        bounds=bounds,
        preprocessing_X=NormalizeBounds(bounds),
        preprocessing_y=NormalizeY(),
        regressor=regressor(),
        infinities_classifier=infinities_classifier,
        random_state=42,
    )
    surrogate.append(X, y)
    rng = np.random.default_rng(8)
    X_test = rng.uniform(-2.0, 2.0, size=(10, d))
    mean = surrogate.predict(X_test)
    assert mean.shape == (len(X_test),)
    assert np.all(np.isfinite(mean))
    # The other entry points go through the same code path.
    assert np.array_equal(surrogate.logp(X_test), mean)
    std = surrogate.predict_std(X_test)
    assert std.shape == (len(X_test),)
    assert np.all(std >= 0)
    mean2, std2 = surrogate.predict(X_test, return_std=True)
    assert np.array_equal(mean2, mean)
    assert std2.shape == (len(X_test),)


def test_predict_without_classifier_is_accurate():
    """
    Not just non-crashing: with no classifier every point is finite, so the prediction
    must be the plain GP fit. Checked against the true log-posterior it was trained on.
    """
    d = 2
    bounds, X, y = gaussian_training_set(d=d, n=200, seed=9, half_width=3.0)
    surrogate = SurrogateModel(
        bounds=bounds,
        preprocessing_X=NormalizeBounds(bounds),
        preprocessing_y=NormalizeY(),
        regressor=regressor(),
        infinities_classifier=None,
        random_state=42,
    )
    surrogate.append(X, y)
    rng = np.random.default_rng(10)
    X_test = rng.uniform(-1.5, 1.5, size=(30, d))
    y_true = -0.5 * np.sum(X_test**2, axis=1)
    y_pred = surrogate.predict(X_test)
    # No point may be silently classified away to -inf.
    assert np.all(np.isfinite(y_pred))
    assert np.max(np.abs(y_pred - y_true)) < 0.5
