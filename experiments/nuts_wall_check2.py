"""
Wall-time + eval comparison, post persistent-JIT fix: NORA sampler="nuts" vs
"ultranest" on Rosenbrock 2D/5D and a stiff high-d (d=8) Gaussian.

Init setup fixed vs the first attempt: Gaussian bounds are set per-axis to +-6
marginal-sigma (so uniform init finds finite points), and Rosenbrock gets a
larger max_initial. We report wall, true-eval count, convergence, and the
nuts/ultranest wall ratio.
"""
import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from gpry.run import Runner

OUT = ("/private/tmp/claude-501/-Users-jeg-Documents-GPry-jax-gpry-jax-claude/"
       "5651b097-65b4-41f8-b160-51ff40d58f48/scratchpad/wall2_results.json")
RNG = np.random.default_rng(0)


def make_rosen(d):
    bounds = [[-2.0, 2.0]] * d
    def logLkl(X):
        X = np.atleast_2d(np.asarray(X, float))
        v = np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2
                   + (1 - X[:, :-1]) ** 2, axis=1)
        out = -v / 200.0  # gentler scale so uniform init finds finite points
        return float(out[0]) if X.shape[0] == 1 else out
    return logLkl, bounds


def make_stiff_gauss(d, cond=20.0):
    stds = np.geomspace(1.0, 1.0 / np.sqrt(cond), d)
    Q, _ = np.linalg.qr(RNG.standard_normal((d, d)))
    cov = Q @ np.diag(stds ** 2) @ Q.T
    cov = 0.5 * (cov + cov.T)
    prec = np.linalg.inv(cov)
    const = -0.5 * (d * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1])
    marg = np.sqrt(np.diag(cov))
    bounds = [[-6 * m, 6 * m] for m in marg]  # +-6 sigma per axis -> init finds finite pts

    def logLkl(X):
        X = np.atleast_2d(np.asarray(X, float))
        q = np.einsum("ni,ij,nj->n", X, prec, X, optimize=True)
        out = const - 0.5 * q
        return float(out[0]) if X.shape[0] == 1 else out

    return logLkl, bounds


def run_target(name, logLkl, bounds, options):
    row = {"target": name, "d": len(bounds)}
    for sampler in ["ultranest", "nuts"]:
        t0 = time.time()
        try:
            r = Runner(logLkl, bounds, gpr="RBF",
                       gp_acquisition={"NORA": {"sampler": sampler, "mc_every": 1}},
                       mc={"nested": {}}, options=options,
                       seed=1234, verbose=1, checkpoint=None)
            note = None
            try:
                r.run()
            except (IndexError, AssertionError) as e:
                # Post-convergence final-MC issues (Runner-internal, sampler-
                # agnostic); the active-learning loop itself already finished.
                note = f"{type(e).__name__} in final MC"
            row[sampler] = {"wall_s": round(time.time() - t0, 1),
                            "n_total": int(r.gpr.n_total),
                            "converged": bool(getattr(r, "has_converged", False)),
                            "note": note}
        except Exception as e:
            row[sampler] = {"error": f"{type(e).__name__}: {str(e)[:70]}"}
    u, n = row.get("ultranest", {}), row.get("nuts", {})
    if "wall_s" in u and "wall_s" in n:
        row["nuts_over_ns_wall"] = round(n["wall_s"] / max(1e-6, u["wall_s"]), 2)
    print(f">>> {json.dumps(row)}", flush=True)
    return row


def main():
    targets = [
        ("rosen_2d", *make_rosen(2), {"n_initial": 6, "max_initial": 20, "max_total": 40}),
        ("rosen_5d", *make_rosen(5), {"n_initial": 10, "max_initial": 60, "max_total": 120}),
        ("gauss_d8", *make_stiff_gauss(8), {"n_initial": 16, "max_initial": 100, "max_total": 140}),
    ]
    results = []
    for name, logLkl, bounds, options in targets:
        print("\n" + "#" * 70 + f"\n# {name} (d={len(bounds)})\n" + "#" * 70, flush=True)
        results.append(run_target(name, logLkl, bounds, options))
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    print(f"{'target':10s} {'US wall':>8s} {'NUTS wall':>10s} {'ratio':>7s} "
          f"{'US evals':>9s} {'NUTS evals':>11s}")
    for r in results:
        u, n = r.get("ultranest", {}), r.get("nuts", {})
        if "wall_s" in u and "wall_s" in n:
            print(f"{r['target']:10s} {u['wall_s']:>8.1f} {n['wall_s']:>10.1f} "
                  f"{r.get('nuts_over_ns_wall'):>7} {u['n_total']:>9d} {n['n_total']:>11d}")
        else:
            err = u.get("error") or n.get("error")
            print(f"{r['target']:10s}  ERROR: {err}")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
