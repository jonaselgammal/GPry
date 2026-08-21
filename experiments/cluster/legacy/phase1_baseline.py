"""
PHASE 1 re-baseline on current main + the NUTS acquisition branch.

Answers the decision gate for the speed work:
  (a) With `start_from_cov` active (main's improved hyperparameter restarts) and
      SANE budgets (n_initial=3d, convergence-driven), how much of the wall time
      is still the GP hyperparameter refit? That is what a staleness-triggered
      refit would attack. The pre-rebase measurement was ~72% at d=2, but that
      was on stale code with a 10x-oversized initial design.
  (b) What is the NUTS vs nested-sampling cost ratio at matched budget now?

Uses GPry's own per-iteration accounting (`runner.progress.data`) rather than
monkeypatching, so the numbers are the ones GPry itself reports.

Usage:  python phase1_baseline.py <d> <sampler> [max_total] [seed] [outdir]
"""
import os
import sys
import json
import time
import warnings

import numpy as np

# `common.py` lives in ../run. `setup_env.sh` copies it next to this script on
# the cluster, so this only matters when running from the repo checkout.
# (This file moved into legacy/ on 2026-08-20; behaviour is unchanged.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "run"))
import common as C

from gpry.run import Runner


def main():
    d = int(sys.argv[1])
    sampler = sys.argv[2].lower()
    max_total = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    outdir = sys.argv[5] if len(sys.argv) > 5 else "phase1"
    os.makedirs(outdir, exist_ok=True)

    # Generous cap; we WANT convergence to stop the run so we learn the budget
    # actually needed (and can stop over-sizing it by hand).
    if not max_total:
        max_total = {2: 200, 4: 400, 8: 900, 16: 1600}.get(d, int(70 * d ** 1.5))

    tgt = C.make_gaussian(d)
    n_initial = 3 * d
    r = Runner(
        tgt["logLkl"], tgt["bounds"].tolist(), ref_bounds=tgt["ref_bounds"].tolist(),
        gpr="RBF", gp_acquisition={"NORA": {"sampler": sampler, "mc_every": 1}},
        mc={"nested": {}},
        options={"n_initial": n_initial,
                 "max_initial": min(3 * n_initial, max_total),
                 "max_total": max_total},
        seed=seed, verbose=1, checkpoint=None,
    )
    if sampler == "nuts":
        r.acquisition._nuts_kwargs = {"pad_multiple": 512, "max_num_doublings": 7}

    t0 = time.time()
    converged = False
    try:
        r.run()
        converged = bool(getattr(r, "has_converged", False))
    except Exception:
        import traceback
        print("run() ended with exception:\n" + traceback.format_exc(), flush=True)
    wall = time.time() - t0

    df = r.progress.data
    tot = lambda col: float(np.nansum(df[col].values)) if col in df else float("nan")
    t_acq, t_fit = tot("time_acquire"), tot("time_fit")
    t_truth, t_mc = tot("time_truth"), tot("time_mc")
    t_conv = tot("time_convergence")
    # The acquisition LOOP excludes the final MC sample (not part of the loop) and
    # the true-posterior evaluations (which a real, expensive likelihood dominates).
    loop = t_acq + t_fit
    n_iter = int(len(df))
    res = dict(
        d=d, sampler=sampler, seed=seed, max_total=max_total, converged=converged,
        n_total_final=int(r.surrogate.n_total), n_iterations=n_iter,
        wall_s=round(wall, 1),
        t_acquire_s=round(t_acq, 2), t_fit_s=round(t_fit, 2),
        t_truth_s=round(t_truth, 2), t_mc_s=round(t_mc, 2),
        t_convergence_s=round(t_conv, 2),
        fit_share_of_loop=round(t_fit / loop, 4) if loop > 0 else None,
        acq_share_of_loop=round(t_acq / loop, 4) if loop > 0 else None,
        per_acq_call_s=round(t_acq / max(1, n_iter), 4),
        per_fit_call_s=round(t_fit / max(1, n_iter), 4),
    )
    tag = f"d{d}_{sampler}_seed{seed}"
    with open(os.path.join(outdir, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    df.to_csv(os.path.join(outdir, f"{tag}_iters.csv"), index=False)
    print(
        f"[PHASE1] d={d} {sampler} seed={seed}: n_total={res['n_total_final']} "
        f"({n_iter} iters, converged={converged}) wall={res['wall_s']}s | "
        f"acq={res['t_acquire_s']}s ({100*res['acq_share_of_loop']:.0f}% of loop) "
        f"fit={res['t_fit_s']}s ({100*res['fit_share_of_loop']:.0f}% of loop) | "
        f"per-call acq={res['per_acq_call_s']}s fit={res['per_fit_call_s']}s",
        flush=True,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
