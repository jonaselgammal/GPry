"""Detailed breakdown of predict overhead: JAX vs NumPy."""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry.gpr import GaussianProcessRegressor as GPR
import jax.numpy as jnp


def make_gpr(n_dim, use_jax):
    length_scale_prior = np.column_stack([np.full(n_dim, 1e-3), np.full(n_dim, 1e1)])
    return GPR(kernel="RBF", length_scale_prior=length_scale_prior,
               n_hyperopt_restarts=1, use_jax=use_jax)


n_dim = 5
n_train = 500
n_predict = 1000
n_reps = 50

rng = np.random.RandomState(42)
X_train = rng.randn(n_train, n_dim)
y_train = np.sin(X_train[:, 0]) + 0.01 * rng.randn(n_train)
X_test = rng.randn(n_predict, n_dim)

# Fit JAX GP
gpr_jax = make_gpr(n_dim, True)
gpr_jax.fit(X_train, y_train, validate=False)

# Warmup
_ = gpr_jax.predict_native(X_test, return_std=True)
_ = gpr_jax.predict(X_test, return_std=True)

# Already-JAX array test
X_test_jax = jnp.array(X_test, dtype=jnp.float64)

print(f"n_train={n_train}, n_predict={n_predict}, n_dim={n_dim}, {n_reps} reps\n")

# 1. Raw JAX predict (no conversion)
t0 = time.perf_counter()
for _ in range(n_reps):
    m, s = gpr_jax.predict_native(X_test_jax, return_std=True)
    m.block_until_ready()
t_raw_jax = (time.perf_counter() - t0) / n_reps * 1000
print(f"JAX predict (JAX input):     {t_raw_jax:.3f} ms")

# 2. JAX predict with numpy->jax conversion
t0 = time.perf_counter()
for _ in range(n_reps):
    m, s = gpr_jax.predict_native(X_test, return_std=True)
    m.block_until_ready()
t_jax_conv = (time.perf_counter() - t0) / n_reps * 1000
print(f"JAX predict (np input):      {t_jax_conv:.3f} ms  (+{t_jax_conv - t_raw_jax:.3f} ms conversion)")

# 3. JAX predict returning numpy (what GPR.predict does)
t0 = time.perf_counter()
for _ in range(n_reps):
    m, s = gpr_jax.predict(X_test, return_std=True)
t_jax_np = (time.perf_counter() - t0) / n_reps * 1000
print(f"JAX predict (np in+out):     {t_jax_np:.3f} ms  (+{t_jax_np - t_jax_conv:.3f} ms jax->np)")

# 4. Full GPR.predict (JAX path)
t0 = time.perf_counter()
for _ in range(n_reps):
    m, s = gpr_jax.predict(X_test, return_std=True)
t_gpr_jax = (time.perf_counter() - t0) / n_reps * 1000
print(f"GPR.predict (JAX path):      {t_gpr_jax:.3f} ms  (+{t_gpr_jax - t_jax_np:.3f} ms overhead)")

# 5. NumPy baseline
gpr_np = make_gpr(n_dim, False)
gpr_np.fit(X_train, y_train, validate=False)
_ = gpr_np.predict(X_test, return_std=True)  # warmup

t0 = time.perf_counter()
for _ in range(n_reps):
    m, s = gpr_np.predict(X_test, return_std=True)
t_gpr_np = (time.perf_counter() - t0) / n_reps * 1000
print(f"GPR.predict (NumPy path):    {t_gpr_np:.3f} ms")

print(f"\nOverall: NumPy {t_gpr_np:.2f}ms vs JAX {t_gpr_jax:.2f}ms = {t_gpr_np/t_gpr_jax:.2f}x")
print(f"Pure compute: NumPy {t_gpr_np:.2f}ms vs JAX {t_raw_jax:.2f}ms = {t_gpr_np/t_raw_jax:.2f}x")
