"""
Deep dive: WHY is JAX slower on CPU? Separate recompilation from compute.
Also test vectorized operations (vmap) vs sequential.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
from jax import jit, vmap
jax.config.update("jax_enable_x64", True)

# ============================================================
# 1. Raw kernel matrix: JAX vs NumPy BLAS
# ============================================================
def numpy_rbf_kernel(X1, X2, length_scale):
    """NumPy RBF kernel using BLAS."""
    sX1 = X1 / length_scale
    sX2 = X2 / length_scale
    sq1 = np.sum(sX1**2, axis=1)
    sq2 = np.sum(sX2**2, axis=1)
    dist_sq = sq1[:, None] + sq2[None, :] - 2.0 * sX1 @ sX2.T
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.exp(-0.5 * dist_sq)

@jit
def jax_rbf_kernel(X1, X2, length_scale):
    sX1 = X1 / length_scale
    sX2 = X2 / length_scale
    sq1 = jnp.sum(sX1**2, axis=1)
    sq2 = jnp.sum(sX2**2, axis=1)
    dist_sq = sq1[:, None] + sq2[None, :] - 2.0 * sX1 @ sX2.T
    dist_sq = jnp.maximum(dist_sq, 0.0)
    return jnp.exp(-0.5 * dist_sq)

print("=" * 80)
print("  1. RAW KERNEL MATRIX COMPUTATION (no GP overhead)")
print("=" * 80)
print(f"  {'n_train x n_test':<20} {'NumPy (ms)':<12} {'JAX (ms)':<12} {'Ratio':<10}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")

for n_train, n_test in [(50, 50), (100, 100), (200, 200), (500, 500),
                         (500, 1000), (1000, 1000), (100, 10000)]:
    n_dim = 5
    rng = np.random.RandomState(42)
    X1 = rng.randn(n_train, n_dim)
    X2 = rng.randn(n_test, n_dim)
    ls = np.ones(n_dim)

    X1j = jnp.array(X1)
    X2j = jnp.array(X2)
    lsj = jnp.array(ls)

    # Warmup JAX
    _ = jax_rbf_kernel(X1j, X2j, lsj).block_until_ready()

    n_reps = 100
    t0 = time.perf_counter()
    for _ in range(n_reps):
        numpy_rbf_kernel(X1, X2, ls)
    t_np = (time.perf_counter() - t0) / n_reps * 1000

    t0 = time.perf_counter()
    for _ in range(n_reps):
        jax_rbf_kernel(X1j, X2j, lsj).block_until_ready()
    t_jax = (time.perf_counter() - t0) / n_reps * 1000

    ratio = t_np / t_jax
    label = f"{n_train}x{n_test}"
    print(f"  {label:<20} {t_np:<12.3f} {t_jax:<12.3f} {ratio:<10.2f}x")

# ============================================================
# 2. Full GP predict: JAX vs NumPy (same shape, no recompilation)
# ============================================================
print(f"\n{'=' * 80}")
print("  2. GP PREDICT (fixed shape, no recompilation)")
print("=" * 80)

@jit
def jax_gp_predict(X_new, X_train, alpha_, V_, length_scale, output_scale_sq, white_noise):
    """Full GP predict mean+var, JIT compiled."""
    K_trans = output_scale_sq * jax_rbf_kernel(X_new, X_train, length_scale)
    y_mean = K_trans @ alpha_
    k_diag = (output_scale_sq + white_noise) * jnp.ones(X_new.shape[0])
    M = V_ @ K_trans.T  # (n_train, n_new)
    y_var = k_diag - jnp.sum(M**2, axis=0)
    y_var = jnp.maximum(y_var, 1e-30)
    return y_mean, jnp.sqrt(y_var)

def numpy_gp_predict(X_new, X_train, alpha_, V_, length_scale, output_scale_sq, white_noise):
    """NumPy GP predict mean+var."""
    K_trans = output_scale_sq * numpy_rbf_kernel(X_new, X_train, length_scale)
    y_mean = K_trans @ alpha_
    k_diag = (output_scale_sq + white_noise) * np.ones(X_new.shape[0])
    M = V_ @ K_trans.T
    y_var = k_diag - np.sum(M**2, axis=0)
    y_var = np.maximum(y_var, 1e-30)
    return y_mean, np.sqrt(y_var)

print(f"  {'n_train, n_test':<20} {'NumPy (ms)':<12} {'JAX (ms)':<12} {'Ratio':<10}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")

for n_train, n_test in [(50, 1), (100, 1), (500, 1), (1000, 1),
                         (50, 100), (100, 1000), (500, 1000), (1000, 5000)]:
    n_dim = 5
    rng = np.random.RandomState(42)
    X_train = rng.randn(n_train, n_dim)
    y_train = np.sin(X_train[:, 0])
    X_test = rng.randn(n_test, n_dim)
    ls = np.ones(n_dim)
    osc = 1.0
    wn = 1e-4

    # Fake precomputed arrays
    K = numpy_rbf_kernel(X_train, X_train, ls)
    K = osc * K + (wn + 1e-6) * np.eye(n_train)
    L = np.linalg.cholesky(K)
    V = np.linalg.inv(L)
    alpha_ = np.linalg.solve(K, y_train)

    # JAX versions
    X_train_j = jnp.array(X_train)
    X_test_j = jnp.array(X_test)
    alpha_j = jnp.array(alpha_)
    V_j = jnp.array(V)
    ls_j = jnp.array(ls)

    # Warmup
    _ = jax_gp_predict(X_test_j, X_train_j, alpha_j, V_j, ls_j, osc, wn)
    _[0].block_until_ready()

    n_reps = 200
    t0 = time.perf_counter()
    for _ in range(n_reps):
        numpy_gp_predict(X_test, X_train, alpha_, V, ls, osc, wn)
    t_np = (time.perf_counter() - t0) / n_reps * 1000

    t0 = time.perf_counter()
    for _ in range(n_reps):
        r = jax_gp_predict(X_test_j, X_train_j, alpha_j, V_j, ls_j, osc, wn)
        r[0].block_until_ready()
    t_jax = (time.perf_counter() - t0) / n_reps * 1000

    label = f"{n_train}, {n_test}"
    print(f"  {label:<20} {t_np:<12.3f} {t_jax:<12.3f} {t_np/t_jax:<10.2f}x")

# ============================================================
# 3. JIT RECOMPILATION COST
# ============================================================
print(f"\n{'=' * 80}")
print("  3. JIT RECOMPILATION: cost of shape change vs reuse")
print("=" * 80)

n_dim = 5
rng = np.random.RandomState(42)

# Time a single compilation
X1 = jnp.array(rng.randn(100, n_dim))
X2 = jnp.array(rng.randn(100, n_dim))
ls = jnp.ones(n_dim)

# Force fresh compile by creating new function
def make_fresh_kernel():
    @jit
    def kern(X1, X2, ls):
        sX1 = X1 / ls
        sX2 = X2 / ls
        sq1 = jnp.sum(sX1**2, axis=1)
        sq2 = jnp.sum(sX2**2, axis=1)
        dist_sq = sq1[:, None] + sq2[None, :] - 2.0 * sX1 @ sX2.T
        return jnp.exp(-0.5 * jnp.maximum(dist_sq, 0.0))
    return kern

# Measure one compilation
fn = make_fresh_kernel()
t0 = time.perf_counter()
_ = fn(X1, X2, ls).block_until_ready()
t_compile = (time.perf_counter() - t0) * 1000
print(f"  First call (compile):    {t_compile:.1f} ms")

# Measure reuse (same shape)
t0 = time.perf_counter()
for _ in range(100):
    fn(X1, X2, ls).block_until_ready()
t_reuse = (time.perf_counter() - t0) / 100 * 1000
print(f"  Reuse (same shape):      {t_reuse:.3f} ms")

# Measure shape changes (100->101->102...)
fn2 = make_fresh_kernel()
t0 = time.perf_counter()
for n in range(100, 130):
    X1_n = jnp.array(rng.randn(n, n_dim))
    _ = fn2(X1_n, X2, ls).block_until_ready()
t_change = (time.perf_counter() - t0) / 30 * 1000
print(f"  Shape change (retrace):  {t_change:.1f} ms per call")
print(f"  Recompilation overhead:  {t_change / t_reuse:.0f}x")

# ============================================================
# 4. VECTORIZED OPTIMIZATION (vmap)
# ============================================================
print(f"\n{'=' * 80}")
print("  4. VECTORIZED ACQUISITION OPTIMIZATION (vmap)")
print("=" * 80)

from scipy.optimize import minimize as scipy_minimize

n_train = 200
n_dim = 5
rng = np.random.RandomState(42)
X_train = rng.randn(n_train, n_dim)
y_train = np.sin(X_train[:, 0]) + 0.01 * rng.randn(n_train)

# Build GP state
ls = np.ones(n_dim)
osc = 1.0
wn = 1e-4
K = numpy_rbf_kernel(X_train, X_train, ls)
K = osc * K + (wn + 1e-6) * np.eye(n_train)
L = np.linalg.cholesky(K)
V = np.linalg.inv(L)
alpha_ = np.linalg.solve(K, y_train)

# Acquisition function (LogExp style)
def numpy_acq(x, X_train, alpha_, V, ls, osc, wn, zeta, baseline):
    x = x.reshape(1, -1)
    K_trans = osc * numpy_rbf_kernel(x, X_train, ls)
    mu = (K_trans @ alpha_)[0]
    k_diag = osc + wn
    M = V @ K_trans.T
    var = k_diag - np.sum(M**2)
    var = max(var, 1e-30)
    return -(2.0 * zeta * (mu - baseline) + 0.5 * np.log(var))

@jit
def jax_acq(x, X_train, alpha_, V, ls, osc, wn, zeta, baseline):
    x = x[None, :]
    K_trans = osc * jax_rbf_kernel(x, X_train, ls)
    mu = (K_trans @ alpha_)[0]
    k_diag = osc + wn
    M = V @ K_trans.T
    var = k_diag - jnp.sum(M**2)
    var = jnp.maximum(var, 1e-30)
    return -(2.0 * zeta * (mu - baseline) + 0.5 * jnp.log(var))

jax_acq_grad = jit(jax.grad(jax_acq))

# JAX arrays
X_train_j = jnp.array(X_train)
alpha_j = jnp.array(alpha_)
V_j = jnp.array(V)
ls_j = jnp.array(ls)

n_starts = 20
bounds_np = np.array([[-5, 5]] * n_dim)
starts = rng.uniform(-5, 5, (n_starts, n_dim))

# A. Sequential scipy L-BFGS-B (NumPy, current approach)
t0 = time.perf_counter()
results_scipy = []
for x0 in starts:
    res = scipy_minimize(numpy_acq, x0, method='L-BFGS-B',
                        bounds=bounds_np,
                        args=(X_train, alpha_, V, ls, osc, wn, 1.0, 0.0))
    results_scipy.append(res.x)
t_scipy_seq = (time.perf_counter() - t0) * 1000
best_scipy = min(numpy_acq(r, X_train, alpha_, V, ls, osc, wn, 1.0, 0.0) for r in results_scipy)
print(f"  A. Scipy L-BFGS-B sequential ({n_starts} starts):  {t_scipy_seq:.1f} ms  best={best_scipy:.4f}")

# B. Sequential scipy with JAX gradient
def jax_acq_for_scipy(x, *args):
    x_j = jnp.array(x)
    args_j = tuple(jnp.array(a) if isinstance(a, np.ndarray) else jnp.float64(a) for a in args)
    val = float(jax_acq(x_j, *args_j))
    g = np.array(jax_acq_grad(x_j, *args_j))
    return val, g

# Warmup
_ = jax_acq_for_scipy(starts[0], X_train, alpha_, V, ls, osc, wn, 1.0, 0.0)

t0 = time.perf_counter()
results_jax_scipy = []
for x0 in starts:
    res = scipy_minimize(jax_acq_for_scipy, x0, method='L-BFGS-B', jac=True,
                        bounds=bounds_np,
                        args=(X_train, alpha_, V, ls, osc, wn, 1.0, 0.0))
    results_jax_scipy.append(res.x)
t_jax_scipy = (time.perf_counter() - t0) * 1000
best_jax_scipy = min(numpy_acq(r, X_train, alpha_, V, ls, osc, wn, 1.0, 0.0) for r in results_jax_scipy)
print(f"  B. Scipy + JAX grad sequential ({n_starts} starts): {t_jax_scipy:.1f} ms  best={best_jax_scipy:.4f}")

# C. jaxopt L-BFGS-B sequential
import jaxopt
import logging
for name in ("jaxopt", "jaxopt._src", "absl"):
    logging.getLogger(name).setLevel(logging.ERROR)

bounds_jax = (jnp.array(bounds_np[:, 0]), jnp.array(bounds_np[:, 1]))

# Wrap acq fn so bounds are baked into the function, avoiding arg collision
def make_bounded_acq(acq_fn, bounds):
    """Wrap acq_fn so jaxopt only sees (x, *gp_args)."""
    @jit
    def bounded_acq(x, X_train, alpha_, V, ls, osc, wn, zeta, baseline):
        # Clip x to bounds (jaxopt LBFGSB handles this, but just in case)
        return acq_fn(x, X_train, alpha_, V, ls, osc, wn, zeta, baseline)
    return bounded_acq

bounded_jax_acq = make_bounded_acq(jax_acq, bounds_jax)
solver = jaxopt.LBFGSB(fun=bounded_jax_acq, maxiter=50, tol=1e-5)
acq_args = (X_train_j, alpha_j, V_j, ls_j, jnp.float64(osc),
            jnp.float64(wn), jnp.float64(1.0), jnp.float64(0.0))

# Warmup
_ = solver.run(jnp.array(starts[0]), bounds_jax, *acq_args)

t0 = time.perf_counter()
results_jaxopt = []
for x0 in starts:
    result = solver.run(jnp.array(x0), bounds_jax, *acq_args)
    results_jaxopt.append(np.array(result.params))
t_jaxopt_seq = (time.perf_counter() - t0) * 1000
best_jaxopt = min(numpy_acq(r, X_train, alpha_, V, ls, osc, wn, 1.0, 0.0) for r in results_jaxopt)
print(f"  C. jaxopt L-BFGS-B sequential ({n_starts} starts):  {t_jaxopt_seq:.1f} ms  best={best_jaxopt:.4f}")

# D. vmap'd jaxopt (all starts in parallel)
@jit
def vmap_optimize(all_x0, X_train, alpha_, V, ls, osc, wn, zeta, baseline):
    def single_opt(x0):
        result = solver.run(x0, bounds_jax,
                           X_train, alpha_, V, ls, osc, wn, zeta, baseline)
        return result.params, result.state.value
    return vmap(single_opt)(all_x0)

all_starts_j = jnp.array(starts)

# Warmup (includes compilation)
t0 = time.perf_counter()
opt_x, opt_val = vmap_optimize(all_starts_j, X_train_j, alpha_j, V_j, ls_j,
                                jnp.float64(osc), jnp.float64(wn),
                                jnp.float64(1.0), jnp.float64(0.0))
opt_x.block_until_ready()
t_vmap_compile = (time.perf_counter() - t0) * 1000
best_vmap_compile = float(jnp.min(opt_val))
print(f"  D. vmap jaxopt (first call, compile):             {t_vmap_compile:.1f} ms  best={best_vmap_compile:.4f}")

# Reuse (same shapes)
t0 = time.perf_counter()
n_reps = 10
for _ in range(n_reps):
    opt_x, opt_val = vmap_optimize(all_starts_j, X_train_j, alpha_j, V_j, ls_j,
                                    jnp.float64(osc), jnp.float64(wn),
                                    jnp.float64(1.0), jnp.float64(0.0))
    opt_x.block_until_ready()
t_vmap_reuse = (time.perf_counter() - t0) / n_reps * 1000
print(f"  D. vmap jaxopt (reuse, {n_starts} starts):             {t_vmap_reuse:.1f} ms  best={float(jnp.min(opt_val)):.4f}")

# E. vmap'd with different n_train (simulating shape change)
print(f"\n  E. vmap cost with different n_train (recompilation):")
for nt in [50, 100, 200, 500]:
    X_tr = jnp.array(rng.randn(nt, n_dim))
    y_tr = jnp.sin(X_tr[:, 0])
    K_tr = jax_rbf_kernel(X_tr, X_tr, ls_j)
    K_full = osc * K_tr + (wn + 1e-6) * jnp.eye(nt)
    L_tr = jnp.linalg.cholesky(K_full)
    V_tr = jnp.linalg.inv(L_tr)
    a_tr = jnp.linalg.solve(K_full, y_tr)

    t0 = time.perf_counter()
    opt_x, opt_val = vmap_optimize(all_starts_j, X_tr, a_tr, V_tr, ls_j,
                                    jnp.float64(osc), jnp.float64(wn),
                                    jnp.float64(1.0), jnp.float64(0.0))
    opt_x.block_until_ready()
    t_retrace = (time.perf_counter() - t0) * 1000

    # Second call (cached)
    t0 = time.perf_counter()
    opt_x, opt_val = vmap_optimize(all_starts_j, X_tr, a_tr, V_tr, ls_j,
                                    jnp.float64(osc), jnp.float64(wn),
                                    jnp.float64(1.0), jnp.float64(0.0))
    opt_x.block_until_ready()
    t_cached = (time.perf_counter() - t0) * 1000
    print(f"     n_train={nt:<5}  compile: {t_retrace:.0f} ms,  cached: {t_cached:.1f} ms")

print(f"\n{'=' * 80}")
