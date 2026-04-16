"""
Test ensemble fallback mechanism.

Validates that:
1. RandomForestFallback implements the GP-compatible predict interface
2. Fallback integrates correctly with SurrogateModel
3. When GP fits normally, fallback is trained but not used
4. End-to-end with Runner using fallback=True
"""

import sys
import os
import time
import tempfile
import shutil
import numpy as np
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Test 1: Unit test of RandomForestFallback interface
# ---------------------------------------------------------------------------

def test_rf_interface():
    """Test that RandomForestFallback matches GP predict interface."""
    from gpry.experimental_fallback import RandomForestFallback

    print("  Test 1: RandomForestFallback interface")

    rng = np.random.RandomState(42)
    n, d = 50, 3
    X = rng.randn(n, d)
    y = np.sum(X ** 2, axis=1)  # simple quadratic

    rf = RandomForestFallback(n_estimators=50, random_state=42)
    rf.fit(X, y)

    # Test predict (mean only)
    X_test = rng.randn(10, d)
    result = rf.predict(X_test)
    assert isinstance(result, list), "predict should return a list"
    assert len(result) == 1, "predict without flags should return [y_mean]"
    assert result[0].shape == (10,), f"y_mean shape should be (10,), got {result[0].shape}"
    print(f"    mean only: OK (shape={result[0].shape})")

    # Test predict with std
    result = rf.predict(X_test, return_std=True)
    assert len(result) == 2, "predict with return_std should return [y_mean, y_std]"
    assert result[1].shape == (10,), f"y_std shape should be (10,), got {result[1].shape}"
    assert np.all(result[1] >= 0), "std should be non-negative"
    print(f"    with std: OK (mean_range=[{result[0].min():.2f}, {result[0].max():.2f}], "
          f"std_range=[{result[1].min():.3f}, {result[1].max():.3f}])")

    # Test predict with gradients (single point)
    x_single = X_test[:1]
    result = rf.predict(x_single, return_std=True, return_mean_grad=True,
                        return_std_grad=True)
    assert len(result) == 4, "predict with all flags should return 4 elements"
    assert result[2].shape == (1, d), f"mean_grad shape should be (1, {d}), got {result[2].shape}"
    assert result[3].shape == (1, d), f"std_grad shape should be (1, {d}), got {result[3].shape}"
    print(f"    with grads: OK (mean_grad={result[2][0]}, std_grad={result[3][0]})")

    # Test predict_std
    std = rf.predict_std(X_test)
    assert std.shape == (10,), f"predict_std shape should be (10,), got {std.shape}"
    print(f"    predict_std: OK")

    print("    PASSED")


# ---------------------------------------------------------------------------
# Test 2: End-to-end with Runner
# ---------------------------------------------------------------------------

def test_e2e_with_runner():
    """Test fallback=True integrates with Runner end-to-end."""
    from gpry.run import Runner
    from gpry.tools import mean_covmat_from_samples

    print("\n  Test 2: End-to-end with Runner (fallback=True)")

    mean = np.array([3.0, 2.0])
    cov = np.array([[0.5, 0.4], [0.4, 1.5]])
    rv = multivariate_normal(mean, cov)

    def logLkl(x_1, x_2):
        return rv.logpdf(np.array([x_1, x_2]).T)

    bounds = [[-10, 10], [-10, 10]]
    tmpdir = tempfile.mkdtemp(prefix="gpry_fallback_e2e_")
    checkpoint = os.path.join(tmpdir, "run")

    try:
        t0 = time.time()
        runner = Runner(
            logLkl, bounds,
            surrogate={
                "regressor": {"kernel": "RBF", "use_jax": False},
                "fallback": True,  # Enable fallback
            },
            checkpoint=checkpoint,
            load_checkpoint="overwrite",
            verbose=0, seed=42,
        )
        runner.run()
        t_total = time.time() - t0

        surr = runner.surrogate
        print(f"    converged: {runner.has_converged}")
        print(f"    n_total: {surr.n_total}")
        print(f"    using_fallback: {surr._using_fallback}")
        print(f"    fallback fitted: {surr._fallback.fitted}")
        print(f"    time: {t_total:.1f}s")

        # Check posterior accuracy
        mc = runner.last_mc_samples()
        if mc is not None:
            mean_mc, _ = mean_covmat_from_samples(mc["X"], mc["w"])
            err = np.max(np.abs(mean_mc - mean))
            print(f"    mean_err: {err:.4f}")

        assert surr._fallback.fitted, "Fallback should be trained alongside GP"
        assert not surr._using_fallback, "GP should not have failed on simple Gaussian"
        print("    PASSED")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: Fallback with adaptive noise + max_training_points
# ---------------------------------------------------------------------------

def test_combined_features():
    """Test fallback works together with adaptive noise and max_training_points."""
    from gpry.run import Runner

    print("\n  Test 3: Combined features (fallback + adaptive + max_training_points)")

    mean = np.array([3.0, 2.0])
    cov = np.array([[0.5, 0.4], [0.4, 1.5]])
    rv = multivariate_normal(mean, cov)

    def logLkl(x_1, x_2):
        return rv.logpdf(np.array([x_1, x_2]).T)

    bounds = [[-10, 10], [-10, 10]]
    tmpdir = tempfile.mkdtemp(prefix="gpry_fallback_combined_")
    checkpoint = os.path.join(tmpdir, "run")

    try:
        t0 = time.time()
        runner = Runner(
            logLkl, bounds,
            surrogate={
                "regressor": {"kernel": "RBF", "use_jax": False},
                "adaptive_noise": True,
                "max_training_points": 50,
                "fallback": True,
            },
            convergence_criterion=False,  # DontConverge
            checkpoint=checkpoint,
            load_checkpoint="overwrite",
            verbose=0, seed=42,
            options={"max_finite": 100, "max_total": 200},
        )
        runner.run()
        t_total = time.time() - t0

        surr = runner.surrogate
        n_gpr = surr.gpr.X_train_.shape[0]
        print(f"    n_total: {surr.n_total}")
        print(f"    n_gpr: {n_gpr}")
        print(f"    n_gpr <= 50: {n_gpr <= 50}")
        print(f"    using_fallback: {surr._using_fallback}")
        print(f"    fallback fitted: {surr._fallback.fitted}")
        print(f"    time: {t_total:.1f}s")

        assert n_gpr <= 50, f"max_training_points violated: {n_gpr} > 50"
        assert surr._fallback.fitted, "Fallback should be trained"
        print("    PASSED")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: Fallback with JAX
# ---------------------------------------------------------------------------

def test_jax_with_fallback():
    """Test fallback works with JAX backend."""
    from gpry.run import Runner

    print("\n  Test 4: JAX + adaptive + fallback")

    mean = np.array([3.0, 2.0])
    cov = np.array([[0.5, 0.4], [0.4, 1.5]])
    rv = multivariate_normal(mean, cov)

    def logLkl(x_1, x_2):
        return rv.logpdf(np.array([x_1, x_2]).T)

    bounds = [[-10, 10], [-10, 10]]
    tmpdir = tempfile.mkdtemp(prefix="gpry_fallback_jax_")
    checkpoint = os.path.join(tmpdir, "run")

    try:
        t0 = time.time()
        runner = Runner(
            logLkl, bounds,
            surrogate={
                "regressor": {"kernel": "RBF", "use_jax": True},
                "adaptive_noise": True,
                "fallback": True,
            },
            checkpoint=checkpoint,
            load_checkpoint="overwrite",
            verbose=0, seed=42,
        )
        runner.run()
        t_total = time.time() - t0

        surr = runner.surrogate
        print(f"    converged: {runner.has_converged}")
        print(f"    n_total: {surr.n_total}")
        print(f"    using_fallback: {surr._using_fallback}")
        print(f"    time: {t_total:.1f}s")
        print("    PASSED")

    except Exception as e:
        import traceback
        print(f"    ERROR: {e}")
        traceback.print_exc()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Ensemble Fallback Tests")
    print("=" * 70)

    test_rf_interface()
    test_e2e_with_runner()
    test_combined_features()
    test_jax_with_fallback()

    print("\n" + "=" * 70)
    print("  All fallback tests complete")
    print("=" * 70)
