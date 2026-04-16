"""Tests for the adaptive hyperparameter optimization scheduler."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gpry.experimental_hyperopt_scheduler import HyperparameterScheduler


def test_basic_scheduling():
    """Test that the scheduler follows the basic fit_full/fit_simple pattern."""
    sched = HyperparameterScheduler(
        fit_full_every=3, fit_simple_every=1, n_restarts_base=6,
    )
    # Iteration 0: fit_simple_every=1 triggers (0 % 1 == 0)
    result = sched.should_fit(0, n_points=20)
    assert result is not False
    assert result["n_restarts"] == 1  # simple fit

    # Iteration 2: fit_full_every=3 triggers (2 % 3 == 2)
    result = sched.should_fit(2, n_points=20)
    assert result is not False
    assert result["n_restarts"] >= 2  # full fit


def test_lml_stability_skip():
    """Test that fits are skipped when LML is stable."""
    sched = HyperparameterScheduler(
        fit_full_every=5, fit_simple_every=1, n_restarts_base=5,
        lml_tol=0.1, lml_window=3,
    )
    # Record stable LML
    for lml in [-50.0, -50.02, -50.01, -50.03]:
        sched.record_lml(lml)

    # Simple fit should be skipped (LML range = 0.03 < tol=0.1)
    result = sched.should_fit(3, n_points=50)
    assert result is False
    assert sched._n_skipped == 1

    # But full fit should NOT be skipped
    result = sched.should_fit(4, n_points=50)
    assert result is not False


def test_lml_instability_triggers_fit():
    """Test that unstable LML triggers a fit even on simple schedule."""
    sched = HyperparameterScheduler(
        fit_full_every=10, fit_simple_every=1, n_restarts_base=5,
        lml_tol=0.1, lml_window=3,
    )
    # Record unstable LML
    for lml in [-50.0, -48.0, -52.0]:
        sched.record_lml(lml)

    # Simple fit should proceed (LML range = 4.0 > tol=0.1)
    result = sched.should_fit(0, n_points=50)
    assert result is not False


def test_restart_decay():
    """Test that restarts decrease over time."""
    sched = HyperparameterScheduler(
        fit_full_every=2, fit_simple_every=None, n_restarts_base=10,
        restart_decay=0.5, min_restarts=2, decay_period=3,
    )
    restarts = []
    for i in range(20):
        result = sched.should_fit(i, n_points=50)
        if result is not False and result["n_restarts"] > 1:
            restarts.append(result["n_restarts"])
    # Restarts should decrease
    assert restarts[0] >= restarts[-1]
    # Should never go below min_restarts
    assert all(r >= 2 for r in restarts)


def test_subset_selection():
    """Test that training subset is selected correctly for large N."""
    sched = HyperparameterScheduler(subset_threshold=50, subset_size=20)
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y = rng.randn(100)

    X_sub, y_sub, _ = sched.get_training_subset(X, y)
    assert len(X_sub) == 20
    assert len(y_sub) == 20
    # Top 10 highest-y points should be in the subset
    top10_idx = np.argsort(y)[-10:]
    for idx in top10_idx:
        assert any(np.allclose(X_sub[j], X[idx]) for j in range(len(X_sub)))


def test_no_subset_for_small_n():
    """Test that no subsetting happens when N is small."""
    sched = HyperparameterScheduler(subset_threshold=100)
    X = np.random.randn(50, 3)
    y = np.random.randn(50)
    X_sub, y_sub, _ = sched.get_training_subset(X, y)
    assert X_sub is X  # should be the same object
    assert y_sub is y


def test_stats():
    """Test that stats are tracked correctly."""
    sched = HyperparameterScheduler(
        fit_full_every=3, fit_simple_every=1, n_restarts_base=5,
    )
    for i in range(6):
        sched.should_fit(i, 50)
    sched.record_lml(-50.0)
    sched.record_lml(-49.5)
    stats = sched.stats
    assert stats["n_lml_recorded"] == 2
    assert stats["n_full_fits"] == 2  # iterations 2 and 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
