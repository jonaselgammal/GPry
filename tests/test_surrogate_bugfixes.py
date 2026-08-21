"""
Regression tests for small, self-contained bugs in the surrogate/preprocessing stack.

Deliberately free of any ``cobaya`` import (directly or via ``model_generator``), so that
this module collects and runs in a bare environment.
"""

import numpy as np
import pytest

from gpry.preprocessing import NormalizeBounds, NormalizeY
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
