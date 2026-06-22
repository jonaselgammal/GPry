import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry.preprocessing import DummyPreprocessor
from gpry.surrogate import SurrogateModel


class DummyPreprocessorWithFitState(DummyPreprocessor):
    fitted = False


def make_surrogate_with_all_finite_svm():
    return SurrogateModel(
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        preprocessing_X=DummyPreprocessorWithFitState,
        preprocessing_y=DummyPreprocessorWithFitState,
        regressor={
            "kernel": "RBF",
            "noise_level": 0.01,
            "length_scale_prior": [0.01, 100.0],
            "output_scale_prior": [0.01, 100.0],
            "n_hyperopt_restarts": 0,
        },
        infinities_classifier={"svm": {"threshold": np.inf}},
        verbose=0,
    )


def make_surrogate_without_classifier():
    """A surrogate with ``infinities_classifier=None`` (classifier disabled)."""
    return SurrogateModel(
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        preprocessing_X=DummyPreprocessorWithFitState,
        preprocessing_y=DummyPreprocessorWithFitState,
        regressor={
            "kernel": "RBF",
            "noise_level": 0.01,
            "length_scale_prior": [0.01, 100.0],
            "output_scale_prior": [0.01, 100.0],
            "n_hyperopt_restarts": 0,
        },
        infinities_classifier=None,
        verbose=0,
    )


def test_surrogate_without_infinities_classifier_core_api():
    """Construct/append/predict must work with ``infinities_classifier=None``.

    Regression guard for a chain of latent bugs on the no-classifier path:
    - construction crashed transforming the noise level with an unfit
      y-preprocessor (the rebinding to a DummyPreprocessor only happened
      inside the classifier branch);
    - ``append`` raised ``AttributeError`` because ``_i_y_sorted`` was set to
      None and then ``.copy()``-d in the ``infinities_classifier is None``
      branch;
    - ``predict``/``predict_std`` (and their transformed variants) called
      ``is_finite_X`` on the ``None`` classifier.
    With no classifier every point is treated as finite.
    """
    surrogate = make_surrogate_without_classifier()
    assert surrogate.infinities_classifier is None

    X = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.6, 0.7]])
    y = np.array([0.0, 3.0, 1.0, 2.0, -1.0])
    surrogate.append(X, y, fit_gpr={"n_restarts": 1}, fit_classifier=False)

    # No classifier => every appended point is used for regression, and the
    # sorted-index / regression-index bookkeeping is populated (not None).
    assert surrogate.n_regress == 5
    assert surrogate._i_y_sorted is not None
    assert len(surrogate._i_y_sorted) == 5
    assert len(surrogate._i_regress) == 5
    # _i_y_sorted is a valid argsort of y (highest y last); y_max reads it.
    assert surrogate._i_y_sorted[-1] == int(np.argmax(y))
    assert np.isclose(surrogate.y_max, y.max())

    # All four numpy predict-family entry points must run and return finite,
    # right-shaped output (no point is vetoed to -inf without a classifier).
    Xt = np.array([[0.15, 0.15], [0.5, 0.5], [0.9, 0.1]])
    mean, std = surrogate.predict(Xt, return_std=True)
    assert mean.shape == (3,) and std.shape == (3,)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std))
    assert np.all(std >= 0.0)

    std_only = surrogate.predict_std(Xt)
    assert std_only.shape == (3,)
    np.testing.assert_allclose(std_only, std, atol=1e-8)

    # Transformed-space variants (DummyPreprocessor X-transform is identity).
    mean_t = surrogate.predict_transformed(Xt)
    std_t = surrogate.predict_std_transformed(Xt)
    assert mean_t.shape == (3,) and std_t.shape == (3,)
    assert np.all(np.isfinite(mean_t)) and np.all(np.isfinite(std_t))


def test_append_refits_gpr_when_classifier_indices_are_unsorted():
    """Unsorted finite indices must not make the surrogate skip GP refits.

    The SVM threshold path returns finite indices in y-sorted order. That ordering is
    intentionally not index-sorted, so counting newly finite points with searchsorted
    would incorrectly report zero and leave the GP trained on stale data.
    """
    surrogate = make_surrogate_with_all_finite_svm()

    X0 = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
        ]
    )
    y0 = np.array([0.0, 3.0, 1.0, 2.0])
    surrogate.append(X0, y0, fit_gpr={"n_restarts": 1}, fit_classifier=True)

    # Surrogate-level accessor (was ``surrogate.gpr.X_train_.shape[0]``). The
    # sklearn-named attribute is being phased out for callers outside the GPR
    # backend per ``AGENTS/jax_split_refactor_plan.md`` Phase 1.
    assert surrogate.n_regress == 4
    assert not np.all(surrogate._i_regress[:-1] <= surrogate._i_regress[1:])

    X1 = np.array([[0.5, 0.5], [0.6, 0.6]])
    y1 = np.array([-1.0, -2.0])
    surrogate.append(X1, y1, fit_gpr={"n_restarts": 1}, fit_classifier=True)

    assert surrogate.n_last_appended_finite == 2
    np.testing.assert_array_equal(surrogate._i_last_appended_regress, np.array([4, 5]))
    np.testing.assert_allclose(surrogate.X_last_appended_regress, X1)
    np.testing.assert_allclose(surrogate.y_last_appended_regress, y1)
    assert surrogate.n_regress == 6
