"""
Verify the persistent-JIT fix: with X_train/alpha padded to a fixed capacity and
n_chains fixed, repeated NUTS calls at DIFFERENT n_train should reuse the same
compiled executable -- only the FIRST call (and a capacity-tier crossing) should
pay the compile cost. Compare against the old per-call ~1.5-1.9s overhead.
"""
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from gpry.gpr import GaussianProcessRegressor as GPR
from gpry import mc_interfaces as mci

BOUNDS = np.array([[-3.0, 3.0]] * 2)


def rosen(X):
    X = np.atleast_2d(X)
    v = np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2, axis=1)
    return -v / 20.0


def fresh_gpr(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 2))
    g = GPR(kernel="RBF", bounds=BOUNDS, account_for_inf=None, n_restarts_optimizer=2)
    g.append_to_data(X, rosen(X), fit_gpr=True, fit_classifier=False)
    return g


def timed(n, seed):
    g = fresh_gpr(n, seed)
    t0 = time.time()
    res = mci.nuts_sample_gp_mean(g, BOUNDS[:, 0], BOUNDS[:, 1],
                                  rng=np.random.default_rng(1), max_chains=32,
                                  n_warmup=80, n_samples=60)
    dt = time.time() - t0
    print(f"  n_train={n:4d}  capacity={res['capacity']:4d}  "
          f"wall={dt:6.2f}s  accept={res['accept_rate']:.2f}  div={res['divergences']}")
    return dt, res["capacity"]


def main():
    print("Persistent-JIT verification (same process, runner cached globally):\n")
    print("Within capacity tier 64 (n_train 40..50):")
    t_cold, _ = timed(40, 10)      # first call: JAX init + compile
    within = [timed(n, 10 + i)[0] for i, n in enumerate([41, 42, 45, 50], start=1)]
    print("\nCross into capacity tier 128 (n_train 70):")
    t_tier, _ = timed(70, 20)      # new capacity -> one recompile
    print("\nBack within tier 128 (n_train 71..80):")
    within2 = [timed(n, 30 + i)[0] for i, n in enumerate([71, 80], start=1)]

    print("\n" + "-" * 60)
    print(f"  first call (init+compile)     : {t_cold:6.2f}s")
    print(f"  within-tier reuse (mean)      : {np.mean(within):6.2f}s  "
          f"(was ~1.5-1.9s/call before the fix)")
    print(f"  tier-crossing recompile       : {t_tier:6.2f}s")
    print(f"  within-tier reuse after (mean): {np.mean(within2):6.2f}s")
    speedup = 1.7 / max(1e-6, np.mean(within))
    print(f"\n  -> steady-state per-call cost dropped ~{speedup:.0f}x "
          f"(from ~1.7s to ~{np.mean(within):.2f}s).")


if __name__ == "__main__":
    main()
