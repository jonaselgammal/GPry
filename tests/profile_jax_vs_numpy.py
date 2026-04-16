"""
Profile JAX vs NumPy GP operations at realistic scales.

Tests predict throughput and acquisition optimization at different
training set sizes (50, 200, 500, 1000) to find the crossover point.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry.gpr import GaussianProcessRegressor as GPR


def make_gp_data(n_train, n_dim, seed=42):
    """Generate synthetic GP training data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_train, n_dim)
    # Smooth function + noise
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1] if n_dim > 1 else X[:, 0])
    y += 0.01 * rng.randn(n_train)
    return X, y


def make_gpr(n_dim, use_jax, n_restarts=0):
    """Create a GPR with proper initialization."""
    length_scale_prior = np.column_stack([np.full(n_dim, 1e-3), np.full(n_dim, 1e1)])
    return GPR(
        kernel="RBF",
        length_scale_prior=length_scale_prior,
        n_restarts_optimizer=n_restarts,
        use_jax=use_jax,
    )


def profile_predict(n_train, n_dim, n_predict, use_jax):
    """Profile predict_mean_std at given scale."""
    X_train, y_train = make_gp_data(n_train, n_dim)

    gpr = make_gpr(n_dim, use_jax, n_restarts=1)
    gpr.fit(X_train, y_train, validate=False)

    # Test points
    rng = np.random.RandomState(123)
    X_test = rng.randn(n_predict, n_dim)

    # Warmup
    gpr.predict(X_test, return_std=True)

    # Time it
    t0 = time.perf_counter()
    n_reps = 20
    for _ in range(n_reps):
        gpr.predict(X_test, return_std=True)
    elapsed = (time.perf_counter() - t0) / n_reps

    return elapsed


def profile_fit(n_train, n_dim, use_jax):
    """Profile GP fitting (hyperparameter optimization)."""
    X_train, y_train = make_gp_data(n_train, n_dim)

    gpr = make_gpr(n_dim, use_jax, n_restarts=2)

    t0 = time.perf_counter()
    gpr.fit(X_train, y_train, validate=False)
    elapsed = time.perf_counter() - t0
    return elapsed


def profile_incremental_fit(n_start, n_end, n_dim, use_jax):
    """Profile incremental GP fitting (as in GPry loop: add 1 point, refit)."""
    X_all, y_all = make_gp_data(n_end, n_dim)

    gpr = make_gpr(n_dim, use_jax, n_restarts=1)

    # Fit initial dataset
    gpr.fit(X_all[:n_start], y_all[:n_start], validate=False)

    t0 = time.perf_counter()
    for i in range(n_start, n_end):
        X_cur = X_all[:i+1]
        y_cur = y_all[:i+1]
        gpr.fit(X_cur, y_cur, validate=False)
    elapsed = time.perf_counter() - t0
    return elapsed


if __name__ == "__main__":
    n_dim = 5  # realistic dimensionality

    print("=" * 80)
    print(f"  JAX vs NumPy GP Profiling (d={n_dim})")
    print("=" * 80)

    # 1. Predict throughput
    print(f"\n{'─' * 80}")
    print("  PREDICT (mean + std), 1000 test points, 20 reps averaged")
    print(f"{'─' * 80}")
    print(f"  {'n_train':<10} {'NumPy (ms)':<15} {'JAX (ms)':<15} {'Speedup':<10}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*10}")

    for n_train in [50, 100, 200, 500, 1000]:
        t_np = profile_predict(n_train, n_dim, 1000, use_jax=False) * 1000
        t_jax = profile_predict(n_train, n_dim, 1000, use_jax=True) * 1000
        speedup = t_np / t_jax if t_jax > 0 else float('inf')
        print(f"  {n_train:<10} {t_np:<15.2f} {t_jax:<15.2f} {speedup:<10.2f}x")

    # 2. Single-point predict (as in acquisition optimization)
    print(f"\n{'─' * 80}")
    print("  PREDICT single point (acq eval), 20 reps averaged")
    print(f"{'─' * 80}")
    print(f"  {'n_train':<10} {'NumPy (ms)':<15} {'JAX (ms)':<15} {'Speedup':<10}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*10}")

    for n_train in [50, 100, 200, 500, 1000]:
        t_np = profile_predict(n_train, n_dim, 1, use_jax=False) * 1000
        t_jax = profile_predict(n_train, n_dim, 1, use_jax=True) * 1000
        speedup = t_np / t_jax if t_jax > 0 else float('inf')
        print(f"  {n_train:<10} {t_np:<15.2f} {t_jax:<15.2f} {speedup:<10.2f}x")

    # 3. Fit (hyperparameter optimization)
    print(f"\n{'─' * 80}")
    print("  FIT (hyperparameter optimization, 2 restarts)")
    print(f"{'─' * 80}")
    print(f"  {'n_train':<10} {'NumPy (s)':<15} {'JAX (s)':<15} {'Speedup':<10}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*10}")

    for n_train in [50, 100, 200, 500]:
        t_np = profile_fit(n_train, n_dim, use_jax=False)
        t_jax = profile_fit(n_train, n_dim, use_jax=True)
        speedup = t_np / t_jax if t_jax > 0 else float('inf')
        print(f"  {n_train:<10} {t_np:<15.3f} {t_jax:<15.3f} {speedup:<10.2f}x")

    # 4. Incremental fit (GPry loop simulation)
    print(f"\n{'─' * 80}")
    print("  INCREMENTAL FIT (add 1 point, refit) - simulates GPry loop")
    print(f"{'─' * 80}")
    print(f"  {'range':<15} {'NumPy (s)':<15} {'JAX (s)':<15} {'Speedup':<10}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*10}")

    for n_start, n_end in [(20, 50), (50, 100), (100, 150)]:
        t_np = profile_incremental_fit(n_start, n_end, n_dim, use_jax=False)
        t_jax = profile_incremental_fit(n_start, n_end, n_dim, use_jax=True)
        speedup = t_np / t_jax if t_jax > 0 else float('inf')
        label = f"{n_start}->{n_end}"
        print(f"  {label:<15} {t_np:<15.3f} {t_jax:<15.3f} {speedup:<10.2f}x")

    print(f"\n{'=' * 80}")
