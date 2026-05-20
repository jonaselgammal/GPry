"""Unit tests for the shared hyperparameter-restart helpers in
``gpry.gpr_optim``.

These tests exercise the backend-agnostic helpers directly on a fixed sklearn
kernel + training set, without going through the GPR class. They check
structural properties (determinism under a fixed RNG seed, bounds clipping,
ordering of scored candidates) rather than specific theta values — the GPR
parity tests already cover the latter end-to-end.
"""

import os
import sys

import numpy as np
import pytest
from sklearn.base import clone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry import gpr_optim
from gpry.kernels_sklearn import RBF, ConstantKernel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_kernel(n_dims=2):
    """A constant * RBF compound kernel matching what GPry constructs."""
    return ConstantKernel(
        constant_value=1.0, constant_value_bounds=(1e-2, 1e3)
    ) * RBF(
        length_scale=np.ones(n_dims),
        length_scale_bounds=np.column_stack(
            [np.full(n_dims, 1e-3), np.full(n_dims, 1e1)]
        ),
    )


def _make_dataset(n_dims=2, n_train=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n_train, n_dims))
    y = -0.5 * np.sum(X**2, axis=1) + 0.05 * rng.standard_normal(n_train)
    return X, y


@pytest.fixture
def kernel_and_data():
    kernel = _make_kernel(n_dims=2)
    X, y = _make_dataset(n_dims=2, n_train=40, seed=0)
    bounds = np.asarray(kernel.bounds, dtype=float)
    return kernel, X, y, bounds


# ---------------------------------------------------------------------------
# kernel_structure / top_training_subset
# ---------------------------------------------------------------------------


def test_kernel_structure_unpacks_compound_kernel(kernel_and_data):
    kernel, *_ = kernel_and_data
    constant_k, length_k, noise_k = gpr_optim.kernel_structure(kernel)
    assert constant_k is not None and hasattr(constant_k, "constant_value")
    assert length_k is not None and hasattr(length_k, "length_scale")
    assert noise_k is None  # no WhiteKernel in this fixture


def test_top_training_subset_returns_largest_y(kernel_and_data):
    _, X, y, _ = kernel_and_data
    X_sub, y_sub = gpr_optim.top_training_subset(X, y)
    assert X_sub is not None
    assert y_sub is not None
    # All selected y's should be >= every non-selected y.
    selected_min = y_sub.min()
    # Subset size must be smaller than full for this fixture to be meaningful.
    assert len(y_sub) <= len(y)
    sorted_y = np.sort(y)
    # The selected items must be the top-k by y.
    expected = sorted_y[-len(y_sub):]
    assert np.allclose(np.sort(y_sub), expected)


def test_top_training_subset_returns_none_when_too_few():
    X = np.zeros((3, 2))
    y = np.zeros(3)
    X_sub, y_sub = gpr_optim.top_training_subset(X, y)
    assert X_sub is None and y_sub is None


# ---------------------------------------------------------------------------
# build_restart_candidates
# ---------------------------------------------------------------------------


def test_build_restart_candidates_deterministic_under_fixed_seed(kernel_and_data):
    kernel, X, y, bounds = kernel_and_data
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    cands_a = gpr_optim.build_restart_candidates(
        clone(kernel), X, y,
        hyperparameter_bounds=bounds,
        n_random=8,
        rng=rng_a,
    )
    cands_b = gpr_optim.build_restart_candidates(
        clone(kernel), X, y,
        hyperparameter_bounds=bounds,
        n_random=8,
        rng=rng_b,
    )
    assert len(cands_a) == len(cands_b)
    for a, b in zip(cands_a, cands_b):
        np.testing.assert_allclose(a, b)


def test_build_restart_candidates_counts(kernel_and_data):
    kernel, X, y, bounds = kernel_and_data
    rng = np.random.default_rng(0)
    cands = gpr_optim.build_restart_candidates(
        clone(kernel), X, y,
        hyperparameter_bounds=bounds,
        n_random=5,
        rng=rng,
    )
    # local-cov + local-quadratic + 5 random, with prev_theta not provided.
    # Both deterministic seeds should fire on this 40-point 2D fixture.
    assert len(cands) == 2 + 5


def test_build_restart_candidates_with_prev_theta(kernel_and_data):
    kernel, X, y, bounds = kernel_and_data
    rng = np.random.default_rng(0)
    prev = np.array(clone(kernel).theta, copy=True)
    cands = gpr_optim.build_restart_candidates(
        clone(kernel), X, y,
        hyperparameter_bounds=bounds,
        n_random=3,
        rng=rng,
        prev_theta=prev,
        start_from_current=True,
    )
    # prev + local-cov + local-quadratic + 3 random.
    assert len(cands) == 1 + 2 + 3
    np.testing.assert_allclose(cands[0], prev)


def test_build_restart_candidates_clip_to_bounds(kernel_and_data):
    kernel, X, y, bounds = kernel_and_data
    rng = np.random.default_rng(0)
    cands = gpr_optim.build_restart_candidates(
        clone(kernel), X, y,
        hyperparameter_bounds=bounds,
        n_random=10,
        rng=rng,
    )
    for theta in cands:
        # Each component must lie within the (clipped) bounds (random samples
        # are LHS in-bounds by construction; deterministic seeds are clipped
        # in build_theta_guess; previous-theta is not clipped, not tested here).
        assert np.all(theta >= bounds[:, 0] - 1e-12)
        assert np.all(theta <= bounds[:, 1] + 1e-12)


# ---------------------------------------------------------------------------
# score_and_filter_candidates
# ---------------------------------------------------------------------------


def test_score_and_filter_orders_by_neg_lml(kernel_and_data):
    _, _, _, bounds = kernel_and_data
    # Three candidates inside the bounds. Their pseudo-LML is the negative of
    # the first component, so the ordering (ascending on neg_lml) is determined
    # entirely by theta[0]: higher theta[0] => more negative neg_lml => first.
    thetas = [
        np.array([np.log(0.5), 0.0, 0.0]),
        np.array([np.log(2.0), 0.0, 0.0]),
        np.array([np.log(1.0), 0.0, 0.0]),
    ]

    def neg_lml(theta):
        return -float(theta[0])

    out = gpr_optim.score_and_filter_candidates(
        thetas,
        neg_lml,
        hyperparameter_bounds=bounds,
        n_select=3,
    )
    assert len(out) == 3
    # Highest theta[0] first (lowest neg_lml).
    assert out[0][0] == thetas[1][0]
    assert out[1][0] == thetas[2][0]
    assert out[2][0] == thetas[0][0]


def test_score_and_filter_skips_non_finite(kernel_and_data):
    _, _, _, bounds = kernel_and_data
    thetas = [
        np.array([np.log(1.0), 0.0, 0.0]),
        np.array([np.log(2.0), 0.0, 0.0]),
    ]

    def neg_lml(theta):
        if theta[0] > 0.1:
            return float("inf")
        return -float(theta[0])

    out = gpr_optim.score_and_filter_candidates(
        thetas,
        neg_lml,
        hyperparameter_bounds=bounds,
        n_select=3,
    )
    assert len(out) == 1
    np.testing.assert_allclose(out[0], thetas[0])


def test_score_and_filter_deduplicates(kernel_and_data):
    _, _, _, bounds = kernel_and_data
    a = np.array([np.log(1.5), 0.1, 0.0])
    thetas = [a, a.copy(), a + 1e-15]  # identical at 12 decimals

    def neg_lml(theta):
        return 0.0

    out = gpr_optim.score_and_filter_candidates(
        thetas,
        neg_lml,
        hyperparameter_bounds=bounds,
        n_select=5,
    )
    assert len(out) == 1


def test_score_and_filter_raises_when_no_finite(kernel_and_data):
    _, _, _, bounds = kernel_and_data
    thetas = [np.array([np.log(1.0), 0.0, 0.0])]

    def neg_lml(theta):
        return float("inf")

    with pytest.raises(RuntimeError):
        gpr_optim.score_and_filter_candidates(
            thetas, neg_lml,
            hyperparameter_bounds=bounds,
            n_select=3,
        )


def test_score_and_filter_respects_n_select(kernel_and_data):
    _, _, _, bounds = kernel_and_data
    thetas = [
        np.array([np.log(0.5), 0.0, 0.0]),
        np.array([np.log(2.0), 0.0, 0.0]),
        np.array([np.log(1.0), 0.0, 0.0]),
        np.array([np.log(0.1), 0.0, 0.0]),
    ]

    def neg_lml(theta):
        return -float(theta[0])

    out = gpr_optim.score_and_filter_candidates(
        thetas, neg_lml,
        hyperparameter_bounds=bounds,
        n_select=2,
    )
    assert len(out) == 2


# ---------------------------------------------------------------------------
# boundary_penalty / is_pathological_optimum
# ---------------------------------------------------------------------------


def test_boundary_penalty_zero_in_interior(kernel_and_data):
    _, _, _, bounds = kernel_and_data
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    assert gpr_optim.boundary_penalty(midpoint, bounds) == 0.0


def test_boundary_penalty_positive_at_edge(kernel_and_data):
    _, _, _, bounds = kernel_and_data
    edge = bounds[:, 0].copy()
    assert gpr_optim.boundary_penalty(edge, bounds) > 0.0


def test_is_pathological_at_majority_bounds(kernel_and_data):
    _, _, _, bounds = kernel_and_data
    # Pin every component to the lower bound.
    theta = bounds[:, 0].copy()
    assert gpr_optim.is_pathological_optimum(theta, bounds)


def test_is_pathological_false_at_interior(kernel_and_data):
    _, _, _, bounds = kernel_and_data
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    assert not gpr_optim.is_pathological_optimum(midpoint, bounds)
