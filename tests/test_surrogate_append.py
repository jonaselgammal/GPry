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
            "n_restarts_optimizer": 0,
        },
        infinities_classifier={"svm": {"threshold": np.inf}},
        verbose=0,
    )


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

    assert surrogate.gpr.X_train_.shape[0] == 4
    assert not np.all(surrogate._i_regress[:-1] <= surrogate._i_regress[1:])

    X1 = np.array([[0.5, 0.5], [0.6, 0.6]])
    y1 = np.array([-1.0, -2.0])
    surrogate.append(X1, y1, fit_gpr={"n_restarts": 1}, fit_classifier=True)

    assert surrogate.n_last_appended_finite == 2
    np.testing.assert_array_equal(surrogate._i_last_appended_regress, np.array([4, 5]))
    np.testing.assert_allclose(surrogate.X_last_appended_regress, X1)
    np.testing.assert_allclose(surrogate.y_last_appended_regress, y1)
    assert surrogate.gpr.X_train_.shape[0] == 6
