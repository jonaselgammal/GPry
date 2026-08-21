"""
Regression tests: GPry must not modify caller-supplied inputs in place.

Config dicts passed by a user belong to the user. Mutating them makes the same dict
unusable for a second object, and silently changes what the caller thinks they passed.

Deliberately free of any ``cobaya`` import, so this module collects in a bare
environment.
"""

from copy import deepcopy

import numpy as np
import pytest

from gpry.surrogate import SurrogateModel


def regressor():
    """Full regressor spec, as a fresh dict per call."""
    return {
        "kernel": "RBF",
        "output_scale_prior": [1e-2, 1e3],
        "length_scale_prior": [1e-3, 1e2],
        "noise_level": 1e-2,
        "optimizer": "fmin_l_bfgs_b",
        "n_restarts_optimizer": 4,
    }


def assert_unchanged(actual, expected, path="config"):
    """Recursively asserts equality, tolerating numpy arrays as values."""
    assert type(actual) is type(expected), f"{path}: type changed"
    if isinstance(expected, dict):
        assert set(actual) == set(expected), f"{path}: keys changed"
        for key in expected:
            assert_unchanged(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected), f"{path}: array changed"
    else:
        assert actual == expected, f"{path}: value changed"


def test_surrogate_does_not_mutate_regressor_dict():
    """
    `SurrogateModel.__init__` used to write the per-dimension-unfolded
    `length_scale_prior` back into the caller's `regressor` dict.
    """
    spec = regressor()
    expected = deepcopy(spec)
    SurrogateModel(
        bounds=np.array([[-4.0, 4.0]] * 4), regressor=spec, random_state=42
    )
    assert_unchanged(spec, expected, "regressor")


def test_surrogate_does_not_mutate_infinities_classifier_dict():
    """
    `InfinitiesClassifiers.__init__` used to inject `nstd_calculator` into the caller's
    per-classifier options dict (and to rename `inf_threshold` in place).
    """
    inf_spec = {"svm": {"threshold": "20s"}}
    expected = deepcopy(inf_spec)
    SurrogateModel(
        bounds=np.array([[-4.0, 4.0]] * 4),
        regressor=regressor(),
        infinities_classifier=inf_spec,
        random_state=42,
    )
    assert_unchanged(inf_spec, expected, "infinities_classifier")


def test_surrogate_config_reusable_across_dimensionalities():
    """
    The practical consequence: the same config dicts must build a second surrogate of a
    different dimensionality. This used to raise TypeError on `length_scale_prior`.
    """
    spec = regressor()
    inf_spec = {"svm": {"threshold": "20s"}}
    for d in (4, 3, 5):
        surrogate = SurrogateModel(
            bounds=np.array([[-4.0, 4.0]] * d),
            regressor=spec,
            infinities_classifier=inf_spec,
            random_state=42,
        )
        assert surrogate.d == d


def test_surrogate_does_not_alias_a_supplied_length_scale_array():
    """A caller-supplied array must not end up shared with the regressor's kernel."""
    spec = regressor()
    lsp = np.array([[1e-3, 1e2]] * 4)
    spec["length_scale_prior"] = lsp
    SurrogateModel(bounds=np.array([[-4.0, 4.0]] * 4), regressor=spec, random_state=42)
    # Still the very same object, with the same contents.
    assert spec["length_scale_prior"] is lsp
    assert np.array_equal(lsp, np.array([[1e-3, 1e2]] * 4))
