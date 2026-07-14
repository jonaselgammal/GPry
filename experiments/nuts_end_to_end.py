"""
End-to-end GPry comparison: NORA acquisition with sampler="nuts" (BlackJAX)
vs sampler="ultranest", on the same target with the same seed.

The acquisition sampler is the ONLY thing that differs. We measure, for each:
  - n_total : true-likelihood evaluations the active-learning loop used,
  - converged : whether GPry's own convergence criterion fired,
  - wall time,
  - KL(surrogate-posterior || truth): posterior accuracy of the FINAL GP,
    scored with a common evaluator (UltraNest sampling of the final GP mean),
    so differences reflect where each run PLACED its points, not the scorer.

This is the arbiter the earlier per-acquisition eval-count proxies could not be.
"""
import json
import time
import warnings

import numpy as np
from scipy.stats import multivariate_normal

warnings.filterwarnings("ignore")

from gpry.run import Runner
from gpry.gp_acquisition import NORA

OUT = ("/private/tmp/claude-501/-Users-jeg-Documents-GPry-jax-gpry-jax-claude/"
       "5651b097-65b4-41f8-b160-51ff40d58f48/scratchpad/e2e_results.json")

MEAN = np.array([3.0, 2.0])
COV = np.array([[0.5, 0.4], [0.4, 1.5]])
RV = multivariate_normal(MEAN, COV)
BOUNDS = [[-10, 10], [-10, 10]]
# Small but coherent budget: cap the initial random phase low so most of the
# budget is the acquisition phase (the only place the sampler choice matters).
OPTIONS = {"n_initial": 8, "max_initial": 20, "max_total": 90}


def logLkl(x_1, x_2):
    return RV.logpdf(np.array([x_1, x_2]).T)


def gaussian_kl(m0, c0, m1, c1):
    """KL( N(m0,c0) || N(m1,c1) )."""
    c1i = np.linalg.inv(c1)
    k = len(m0)
    dm = m1 - m0
    return 0.5 * (np.trace(c1i @ c0) + dm @ c1i @ dm - k
                  + np.log(np.linalg.det(c1) / np.linalg.det(c0)))


def surrogate_moments(runner):
    """Weighted mean/cov of the final GP surrogate posterior, via UltraNest."""
    gpr = runner.gpr
    acq = NORA(bounds=np.array(BOUNDS, dtype=float),
               preprocessing_X=gpr.preprocessing_X, acq_func="LogExp",
               sampler="ultranest", verbose=0)
    X, y, sy, w = acq.do_MC_sample(gpr, bounds=np.array(BOUNDS, dtype=float),
                                   rng=np.random.default_rng(999), sampler="ultranest")
    w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
    w = w / w.sum()
    m = np.average(X, axis=0, weights=w)
    c = np.cov(X.T, aweights=w)
    return m, c


def run_one(sampler):
    t0 = time.time()
    runner = Runner(
        logLkl, BOUNDS, gpr="RBF",
        gp_acquisition={"NORA": {"sampler": sampler, "mc_every": 1}},
        mc={"nested": {}},  # final MC sampler (Runner's default {} is broken)
        options=OPTIONS,
        seed=1234, verbose=1, checkpoint=None,
    )
    try:
        runner.run()
    except IndexError:
        # Pre-existing issue in the post-convergence final MC-sample step; the
        # active-learning loop itself has already finished. Proceed with the
        # trained GPR and our own surrogate-accuracy evaluator below.
        pass
    dt = time.time() - t0
    m, c = surrogate_moments(runner)
    kl = float(gaussian_kl(m, c, MEAN, COV))
    res = dict(
        sampler=sampler,
        n_total=int(runner.gpr.n_total),
        n_finite=int(runner.gpr.n_finite),
        converged=bool(getattr(runner, "has_converged", getattr(runner, "converged", False))),
        wall_s=round(dt, 1),
        kl_surrogate_vs_truth=round(kl, 4),
        surrogate_mean=[round(float(v), 3) for v in m],
        true_mean=MEAN.tolist(),
    )
    print(f"\n>>> [{sampler}] {json.dumps(res)}")
    return res


def main():
    results = []
    for sampler in ["ultranest", "nuts"]:
        print("\n" + "#" * 74 + f"\n# END-TO-END RUN: sampler = {sampler}\n" + "#" * 74)
        try:
            results.append(run_one(sampler))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append(dict(sampler=sampler, error=f"{type(e).__name__}: {e}"))
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 74 + "\nSUMMARY\n" + "=" * 74)
    print(f"{'sampler':12s} {'n_total':>8s} {'converged':>10s} {'wall(s)':>8s} "
          f"{'KL(surrogate||truth)':>22s}")
    for r in results:
        if "error" in r:
            print(f"{r['sampler']:12s}  ERROR: {r['error']}")
            continue
        print(f"{r['sampler']:12s} {r['n_total']:>8d} {str(r['converged']):>10s} "
              f"{r['wall_s']:>8.1f} {r['kl_surrogate_vs_truth']:>22.4f}")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
