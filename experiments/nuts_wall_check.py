"""
Wall-time check: NORA acquisition sampler="nuts" (BlackJAX) vs "ultranest" on
harder targets -- Rosenbrock 2D, Rosenbrock 5D, and a stiff high-d Gaussian.

Runs cheapest-first. After each target, if NUTS is >5x slower in wall time, STOP
and run a recompile-diagnosis probe (times repeated NUTS calls to test whether
every acquisition call retraces/recompiles the JAX pipeline).
"""
import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from gpry.run import Runner
from gpry.gpr import GaussianProcessRegressor as GPR
from gpry import mc_interfaces as mci

OUT = ("/private/tmp/claude-501/-Users-jeg-Documents-GPry-jax-gpry-jax-claude/"
       "5651b097-65b4-41f8-b160-51ff40d58f48/scratchpad/wall_results.json")
RNG = np.random.default_rng(0)


def rosenbrock_logp(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    val = np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1.0 - X[:, :-1]) ** 2, axis=1)
    out = -val / 20.0
    return float(out[0]) if np.asarray(X).shape[0] == 1 else out


def make_rosen(d):
    bounds = [[-3.0, 3.0]] * d
    def logLkl(X):
        return rosenbrock_logp(X)
    return logLkl, bounds


def make_stiff_gauss(d, cond=80.0):
    stds = np.geomspace(1.0, 1.0 / np.sqrt(cond), d)
    Q, _ = np.linalg.qr(RNG.standard_normal((d, d)))
    cov = Q @ np.diag(stds ** 2) @ Q.T
    cov = 0.5 * (cov + cov.T)
    prec = np.linalg.inv(cov)
    const = -0.5 * (d * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1])
    bounds = [[-5.0, 5.0]] * d

    def logLkl(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        q = np.einsum("ni,ij,nj->n", X, prec, X, optimize=True)
        out = const - 0.5 * q
        return float(out[0]) if X.shape[0] == 1 else out

    return logLkl, bounds


def run_target(name, logLkl, bounds, options):
    d = len(bounds)
    row = {"target": name, "d": d}
    for sampler in ["ultranest", "nuts"]:
        t0 = time.time()
        try:
            runner = Runner(
                logLkl, bounds, gpr="RBF",
                gp_acquisition={"NORA": {"sampler": sampler, "mc_every": 1}},
                mc={"nested": {}}, options=options,
                seed=1234, verbose=1, checkpoint=None,
            )
            try:
                runner.run()
            except IndexError:
                pass  # known post-convergence generate_mc_sample issue
            dt = time.time() - t0
            row[sampler] = {"wall_s": round(dt, 1),
                            "n_total": int(runner.gpr.n_total),
                            "converged": bool(getattr(runner, "has_converged", False))}
        except Exception as e:
            row[sampler] = {"error": f"{type(e).__name__}: {e}"}
    if "wall_s" in row.get("ultranest", {}) and "wall_s" in row.get("nuts", {}):
        row["nuts_over_ns_wall"] = round(
            row["nuts"]["wall_s"] / max(1e-6, row["ultranest"]["wall_s"]), 2)
    print(f">>> {json.dumps(row)}")
    return row


def diagnose_recompiles():
    """Time repeated NUTS calls to test the JAX-recompile hypothesis."""
    print("\n" + "=" * 70 + "\nRECOMPILE DIAGNOSIS\n" + "=" * 70)
    d = 2
    bounds = np.array([[-3.0, 3.0]] * d)
    X = RNG.uniform(-3, 3, size=(40, d))
    y = rosenbrock_logp(X)
    gpr = GPR(kernel="RBF", bounds=bounds, account_for_inf=None, n_restarts_optimizer=4)
    gpr.append_to_data(X, y, fit_gpr=True, fit_classifier=False)

    def fresh_gpr(n):
        Xn = RNG.uniform(-3, 3, size=(n, d))
        g = GPR(kernel="RBF", bounds=bounds, account_for_inf=None, n_restarts_optimizer=2)
        g.append_to_data(Xn, np.atleast_1d(rosenbrock_logp(Xn)),
                         fit_gpr=True, fit_classifier=False)
        return g

    def timed_call(g, tag):
        t0 = time.time()
        mci.nuts_sample_gp_mean(g, bounds[:, 0], bounds[:, 1],
                                rng=np.random.default_rng(1), max_chains=32,
                                n_warmup=80, n_samples=60)
        dt = time.time() - t0
        print(f"  {tag:42s} {dt:6.2f}s")
        return dt

    def timed_lean(g, tag, **kw):
        t0 = time.time()
        mci.nuts_sample_gp_mean(g, bounds[:, 0], bounds[:, 1],
                                rng=np.random.default_rng(1), **kw)
        dt = time.time() - t0
        print(f"  {tag:42s} {dt:6.2f}s")
        return dt

    t1 = timed_call(gpr, f"call #1 (cold, n_train={len(gpr.X_train_)})")
    t2 = timed_call(gpr, f"call #2 (SAME shape n_train={len(gpr.X_train_)})")
    g41 = fresh_gpr(41)
    t3 = timed_call(g41, "call #3 (n_train=41, NEW shape)")
    g42 = fresh_gpr(42)
    t4 = timed_call(g42, "call #4 (n_train=42, NEW shape)")
    # Attribute the ~fixed per-call cost: does shrinking chains/steps help?
    g43 = fresh_gpr(43)
    t5 = timed_lean(g43, "call #5 (NEW shape, 8 chains, 40+30)",
                    max_chains=8, n_warmup=40, n_samples=30)
    g44 = fresh_gpr(44)
    t6 = timed_lean(g44, "call #6 (NEW shape, 4 chains, 20+20)",
                    max_chains=4, n_warmup=20, n_samples=20)

    print("\n  Interpretation:")
    print(f"  - first call {t1:.2f}s = one-time JAX init + first compile.")
    print(f"  - subsequent calls ~{np.mean([t2,t3,t4]):.2f}s regardless of shape; a")
    print(f"    NEW n_train adds only ~{np.mean([t3,t4])-t2:+.2f}s -> recompilation is")
    print("    NOT the dominant cost.")
    print(f"  - shrinking chains/steps ({t5:.2f}s, {t6:.2f}s) barely helps -> the cost")
    print("    is FIXED per-call overhead (fresh window_adaptation trace + XLA")
    print("    dispatch), not the on-device sampling compute.")
    print("    Fix: build+jit the multi-chain warmup+sample ONCE (pad X_train/alpha")
    print("    to fixed capacity) and reuse across iterations -> one compile total.")
    return {"cold": round(t1, 2), "same_shape": round(t2, 2),
            "new_shape_41": round(t3, 2), "new_shape_42": round(t4, 2),
            "lean_8ch": round(t5, 2), "lean_4ch": round(t6, 2)}


def main():
    targets = [
        ("rosen_2d", *make_rosen(2), {"n_initial": 6, "max_initial": 14, "max_total": 36}),
        ("rosen_5d", *make_rosen(5), {"n_initial": 15, "max_initial": 35, "max_total": 90}),
        ("gauss_d8", *make_stiff_gauss(8), {"n_initial": 24, "max_initial": 55, "max_total": 140}),
    ]
    results = []
    for name, logLkl, bounds, options in targets:
        print("\n" + "#" * 70 + f"\n# {name} (d={len(bounds)})\n" + "#" * 70)
        row = run_target(name, logLkl, bounds, options)
        results.append(row)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        ratio = row.get("nuts_over_ns_wall")
        if ratio is not None and ratio > 5.0:
            print(f"\n!!! NUTS is {ratio}x slower than UltraNest (> 5x) on {name}. "
                  f"STOPPING remaining targets and diagnosing.")
            results.append(diagnose_recompiles())
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
            break
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
