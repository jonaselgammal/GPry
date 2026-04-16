"""
End-to-end test of JAX-accelerated GPry.

Tests the full pipeline: Runner -> SurrogateModel -> GPR (with JAX) -> Acquisition.
Validates that JAX acceleration produces identical results to numpy and measures speedups.
"""

import sys
import os
import time
import tempfile
import shutil
import numpy as np
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry.run import Runner


def test_full_runner_jax_numpy_consistency():
    """Run GPry with JAX and verify JAX predictions match numpy exactly."""
    mean = [3, 2]
    cov = [[0.5, 0.4], [0.4, 1.5]]
    rv = multivariate_normal(mean, cov)

    def logLkl(x_1, x_2):
        return rv.logpdf(np.array([x_1, x_2]).T)

    bounds = [[-10, 10], [-10, 10]]
    tmpdir = tempfile.mkdtemp(prefix="gpry_jax_test_")

    try:
        checkpoint = os.path.join(tmpdir, "simple")
        runner = Runner(
            logLkl, bounds,
            checkpoint=checkpoint,
            load_checkpoint="overwrite",
            verbose=0,
        )
        runner.run()

        gpr = runner.surrogate.gpr
        print(f"JAX enabled: {gpr.use_jax}")
        print(f"JAX ready: {gpr._jax_accel.ready if gpr._jax_accel else False}")
        print(f"Kernel: {gpr.kernel_}")
        print(f"N training: {gpr.X_train_.shape[0]}")

        assert gpr.use_jax, "JAX should be enabled"
        assert gpr._jax_accel is not None and gpr._jax_accel.ready, \
            "JAX accelerator should be ready"

        # Compare JAX vs numpy predictions on a grid
        rng = np.random.RandomState(42)
        X_test_prepro = runner.surrogate.preprocessing_X.transform(
            rng.uniform(-8, 8, size=(200, 2))
        )

        # Get JAX predictions
        y_mean_jax = gpr.predict(X_test_prepro, return_std=True, validate=False)

        # Disable JAX and get numpy predictions
        accel_backup = gpr._jax_accel
        gpr._jax_accel = None
        y_mean_np = gpr.predict(X_test_prepro, return_std=True, validate=False)
        gpr._jax_accel = accel_backup

        # They should match very closely
        np.testing.assert_allclose(
            y_mean_jax[0], y_mean_np[0], atol=5e-7,
            err_msg="Mean prediction mismatch between JAX and numpy"
        )
        np.testing.assert_allclose(
            y_mean_jax[1], y_mean_np[1], atol=1e-6,
            err_msg="Std prediction mismatch between JAX and numpy"
        )

        max_mean_diff = np.max(np.abs(y_mean_jax[0] - y_mean_np[0]))
        max_std_diff = np.max(np.abs(y_mean_jax[1] - y_mean_np[1]))
        print(f"Max mean diff (JAX vs numpy): {max_mean_diff:.2e}")
        print(f"Max std diff (JAX vs numpy): {max_std_diff:.2e}")

        # Also test predict_std
        y_std_jax = gpr.predict_std(X_test_prepro, validate=False)
        gpr._jax_accel = None
        y_std_np = gpr.predict_std(X_test_prepro, validate=False)
        gpr._jax_accel = accel_backup

        np.testing.assert_allclose(
            y_std_jax, y_std_np, atol=1e-6,
            err_msg="predict_std mismatch between JAX and numpy"
        )

        print("\nFull Runner JAX-NumPy consistency test PASSED!")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_jax_active_during_acquisition():
    """Verify JAX acceleration is active during the acquisition loop."""
    from scipy.stats import multivariate_normal

    mean = [0, 0]
    cov = [[1.0, 0.0], [0.0, 1.0]]
    rv = multivariate_normal(mean, cov)

    def logLkl(x_1, x_2):
        return rv.logpdf(np.array([x_1, x_2]).T)

    bounds = [[-5, 5], [-5, 5]]
    tmpdir = tempfile.mkdtemp(prefix="gpry_jax_active_")

    try:
        checkpoint = os.path.join(tmpdir, "test")
        runner = Runner(
            logLkl, bounds,
            checkpoint=checkpoint,
            load_checkpoint="overwrite",
            options={"max_finite": 30},
            verbose=0,
        )
        runner.run()

        gpr = runner.surrogate.gpr
        assert gpr.use_jax, "JAX should be enabled"

        # The GP should have been called many times during acquisition
        # (n_eval tracks total predict calls)
        print(f"Total GP predict calls: {gpr.n_eval}")
        print(f"Training points: {gpr.X_train_.shape[0]}")

        # Verify accelerator is ready and kernel was detected
        assert gpr._jax_accel.ready, "JAX accelerator should be ready after run"

        print("JAX active during acquisition test PASSED!")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_performance_comparison():
    """Measure speedup of JAX vs numpy predict calls in a realistic scenario."""
    from sklearn.base import clone
    from gpry.gpr import GaussianProcessRegressor

    rng = np.random.RandomState(42)
    n_train = 100
    n_dims = 3
    X_train = rng.randn(n_train, n_dims)
    y_train = np.sum(np.sin(X_train), axis=1) + 0.1 * rng.randn(n_train)

    bounds = np.array([[-5, 5]] * n_dims)
    length_scale_prior = np.column_stack([
        bounds[:, 1] - bounds[:, 0],
        bounds[:, 1] - bounds[:, 0],
    ]) * np.array([[1e-3, 1e1]])

    gpr = GaussianProcessRegressor(
        kernel="RBF",
        output_scale_prior=[1e-2, 1e3],
        length_scale_prior=length_scale_prior,
        noise_level=0.1,
        noise_fixed=True,
        n_restarts_optimizer=2,
        random_state=42,
        use_jax=True,
    )
    gpr.kernel_ = clone(gpr.kernel)
    gpr.fit(X_train, y_train, validate=False)

    X_test = rng.randn(500, n_dims)

    # Warmup JAX
    _ = gpr.predict(X_test, return_std=True, validate=False)

    # Time JAX predict (mean+std)
    n_repeats = 200
    t_start = time.time()
    for _ in range(n_repeats):
        result = gpr.predict(X_test, return_std=True, validate=False)
    t_jax = time.time() - t_start

    # Disable JAX and time numpy
    gpr_accel = gpr._jax_accel
    gpr._jax_accel = None
    t_start = time.time()
    for _ in range(n_repeats):
        result = gpr.predict(X_test, return_std=True, validate=False)
    t_numpy = time.time() - t_start
    gpr._jax_accel = gpr_accel

    speedup = t_numpy / t_jax
    print(f"\nPerformance ({n_train} train, {len(X_test)} test, {n_dims}D, {n_repeats} repeats):")
    print(f"  NumPy: {t_numpy/n_repeats*1000:.2f} ms/call")
    print(f"  JAX:   {t_jax/n_repeats*1000:.2f} ms/call")
    print(f"  Speedup: {speedup:.1f}x")

    # Also test predict_std only
    _ = gpr.predict_std(X_test, validate=False)  # warmup
    t_start = time.time()
    for _ in range(n_repeats):
        _ = gpr.predict_std(X_test, validate=False)
    t_jax_std = time.time() - t_start

    gpr._jax_accel = None
    t_start = time.time()
    for _ in range(n_repeats):
        _ = gpr.predict_std(X_test, validate=False)
    t_numpy_std = time.time() - t_start
    gpr._jax_accel = gpr_accel

    speedup_std = t_numpy_std / t_jax_std
    print(f"\n  predict_std only:")
    print(f"  NumPy: {t_numpy_std/n_repeats*1000:.2f} ms/call")
    print(f"  JAX:   {t_jax_std/n_repeats*1000:.2f} ms/call")
    print(f"  Speedup: {speedup_std:.1f}x")


def test_various_problem_sizes():
    """Test JAX accuracy across different problem sizes."""
    from sklearn.base import clone
    from gpry.gpr import GaussianProcessRegressor

    rng = np.random.RandomState(42)

    sizes = [
        (20, 2, 50),    # small 2D
        (50, 3, 100),   # medium 3D
        (100, 5, 200),  # larger 5D
        (200, 2, 1000), # many test points
    ]

    for n_train, n_dims, n_test in sizes:
        X_train = rng.randn(n_train, n_dims)
        y_train = np.sum(np.sin(X_train), axis=1) + 0.1 * rng.randn(n_train)
        X_test = rng.randn(n_test, n_dims)

        bounds = np.array([[-5, 5]] * n_dims)
        length_scale_prior = np.column_stack([
            bounds[:, 1] - bounds[:, 0],
            bounds[:, 1] - bounds[:, 0],
        ]) * np.array([[1e-3, 1e1]])

        gpr = GaussianProcessRegressor(
            kernel="RBF",
            output_scale_prior=[1e-2, 1e3],
            length_scale_prior=length_scale_prior,
            noise_level=0.1,
            noise_fixed=True,
            n_restarts_optimizer=2,
            random_state=42,
            use_jax=True,
        )
        gpr.kernel_ = clone(gpr.kernel)
        gpr.fit(X_train, y_train, validate=False)

        # JAX predictions
        y_mean_jax, y_std_jax = gpr._jax_accel.predict_mean_std(X_test)

        # NumPy predictions
        gpr._jax_accel_bak = gpr._jax_accel
        gpr._jax_accel = None
        result = gpr.predict(X_test, return_std=True, validate=False)
        y_mean_np, y_std_np = result[0], result[1]
        gpr._jax_accel = gpr._jax_accel_bak

        max_mean_err = np.max(np.abs(y_mean_jax - y_mean_np))
        max_std_err = np.max(np.abs(y_std_jax - y_std_np))

        np.testing.assert_allclose(y_mean_jax, y_mean_np, atol=1e-6)
        np.testing.assert_allclose(y_std_jax, y_std_np, atol=1e-5)

        print(f"  n_train={n_train}, n_dims={n_dims}, n_test={n_test}: "
              f"mean_err={max_mean_err:.2e}, std_err={max_std_err:.2e} OK")

    print("\nVarious problem sizes test PASSED!")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Full Runner JAX-NumPy Consistency")
    print("=" * 60)
    test_full_runner_jax_numpy_consistency()

    print("\n" + "=" * 60)
    print("Test 2: JAX Active During Acquisition")
    print("=" * 60)
    test_jax_active_during_acquisition()

    print("\n" + "=" * 60)
    print("Test 3: Various Problem Sizes")
    print("=" * 60)
    test_various_problem_sizes()

    print("\n" + "=" * 60)
    print("Test 4: Performance Comparison")
    print("=" * 60)
    test_performance_comparison()
