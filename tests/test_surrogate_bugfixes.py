"""
Regression tests for small, self-contained bugs in the surrogate/preprocessing stack.

Deliberately free of any ``cobaya`` import (directly or via ``model_generator``), so that
this module collects and runs in a bare environment.
"""

import numpy as np
import pytest

from gpry.preprocessing import NormalizeBounds, NormalizeY
from gpry.surrogate import SurrogateModel

# Full regressor spec: SurrogateModel has no defaults for this sub-dict (those live in
# run.Runner._construct_surrogate), so a partial dict raises KeyError.
REGRESSOR = {
    "kernel": "RBF",
    "output_scale_prior": [1e-2, 1e3],
    "length_scale_prior": [1e-3, 1e2],
    "noise_level": 1e-2,
    "optimizer": "fmin_l_bfgs_b",
    "n_restarts_optimizer": 4,
}


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
        regressor=REGRESSOR,
        clip_factor=1.1,
        infinities_classifier={"svm": {"threshold": "20s"}},
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
