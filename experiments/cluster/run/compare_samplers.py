"""
Head-to-head of the acquisition samplers on a d-dim anisotropic Gaussian,
scoring each with a sampler-INDEPENDENT read-out and always plotting the FULL
corner (all dimensions, GP vs truth).

Motivation: the earlier UltraNest arm was found to be broken at d>=16 (ESS
collapse to 1, and instant NS termination inside the loop), so every previous
NUTS-vs-nested number was measured against a collapsed baseline. This re-runs
the comparison against samplers that actually work.

The FINAL posterior extraction is matched to the acquisition sampler (NUTS
acquisition -> NUTS final MC, BlackJAX -> BlackJAX), rather than always using
UltraNest: driving a sequential slice-stepper over the numpy surrogate made the
final MC dominate the whole run at d>=16. Loop and final-MC timings are recorded
separately, and the surrogate is checkpointed BEFORE the final MC so a different
final sampler can be re-run on exactly the same GP later.

Usage:
    python compare_samplers.py <d> <sampler> <seed> [max_total] [outdir] [final_mc]

  <sampler>  = nuts | blackjax | ultranest | polychord
  <final_mc> = match (default) | nuts | blackjax | nested
"""
import os
import sys
import json
import time
import warnings
from copy import deepcopy

import numpy as np

import common as C

from gpry.run import Runner


def corner_all_dims(sur, target, path, label, nuts_kwargs=None):
    """
    Corner plot of the GP posterior vs the truth, over ALL dimensions.

    The GP is read out with many-chain NUTS (sampler-independent: the same
    read-out is used whatever built the GP), so the plot compares surrogates,
    not samplers.
    """
    import matplotlib
    matplotlib.use("Agg")
    from getdist import MCSamples, plots
    X, w, _ = C.nuts_corner_samples(sur, target["bounds"], **(nuts_kwargs or {}))
    d = target["d"]
    names = [f"x{i}" for i in range(d)]
    ref = np.random.default_rng(1).multivariate_normal(
        target["mean"], target["cov"], 40000)
    mcs = [MCSamples(samples=ref, names=names, label="truth"),
           MCSamples(samples=np.asarray(X), weights=np.asarray(w), names=names,
                     label=f"GP ({label})")]
    g = plots.get_subplot_plotter(width_inch=min(2 + 0.7 * d, 26))
    g.triangle_plot(mcs, filled=[True, False], legend_labels=[m.label for m in mcs])
    g.export(path)
    return X, w


def main():
    d = int(sys.argv[1])
    sampler = sys.argv[2].lower()
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    max_total = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    outdir = sys.argv[5] if len(sys.argv) > 5 else "compare"
    final_mc = (sys.argv[6] if len(sys.argv) > 6 else "match").lower()
    os.makedirs(outdir, exist_ok=True)
    if not max_total:
        max_total = {2: 200, 4: 400, 8: 900, 16: 1600, 30: 3000}.get(
            d, int(70 * d ** 1.5))
    # Match the final posterior extraction to the acquisition sampler.
    if final_mc == "match":
        final_mc = {"nuts": "nuts", "hmc": "hmc", "blackjax": "blackjax"}.get(
            sampler, "nested")
    mc_opts = {final_mc: {}}

    tgt = C.make_gaussian(d)
    n_initial = 3 * d
    r = Runner(
        tgt["logLkl"], tgt["bounds"].tolist(), ref_bounds=tgt["ref_bounds"].tolist(),
        gpr="RBF", gp_acquisition={"NORA": {"sampler": sampler, "mc_every": 1}},
        mc=mc_opts,
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
        print("run() ended:\n" + traceback.format_exc(), flush=True)
    wall = time.time() - t0

    tag = f"d{d}_{sampler}_mc{final_mc}_seed{seed}"
    sur = r.surrogate
    # Checkpoint the GP: the final MC does not modify it, so this is exactly the
    # surrogate the acquisition produced and a different final sampler can be
    # re-run on it later without repeating the (expensive) acquisition.
    C.save_gp(deepcopy(sur), os.path.join(outdir, f"{tag}_surrogate.pkl"))
    # Persist the final MC sample itself, for later comparison of final samplers.
    try:
        _s = r.last_mc_samples()
        _w = _s["w"]
        np.savez_compressed(
            os.path.join(outdir, f"{tag}_finalmc.npz"),
            X=np.asarray(_s["X"]),
            w=(np.ones(len(_s["X"])) if _w is None else np.asarray(_w, float)),
        )
    except Exception as exc:
        print(f"  [warn] could not save final MC sample: {exc!r}", flush=True)
    r.progress.data.to_csv(os.path.join(outdir, f"{tag}_iters.csv"), index=False)

    # Sampler-independent scoring + FULL corner plot
    rec, _ = C.evaluate_recovery(
        sur, tgt["bounds"], tgt["mean"], tgt["cov"], tgt["marg"],
        nuts_kwargs=dict(n_chains=32, n_warmup=250, n_samples=250))
    try:
        corner_all_dims(sur, dict(bounds=tgt["bounds"], mean=tgt["mean"],
                                  cov=tgt["cov"], d=d),
                        os.path.join(outdir, f"{tag}_corner.png"), sampler,
                        nuts_kwargs=dict(n_chains=32, n_warmup=250, n_samples=250))
    except Exception as exc:
        print(f"  [warn] corner plot failed: {exc!r}", flush=True)

    df = r.progress.data
    tot = lambda c: float(np.nansum(df[c].values)) if c in df else float("nan")
    res = dict(d=d, sampler=sampler, final_mc=final_mc, seed=seed,
               converged=converged,
               n_total=int(sur.n_total), n_iterations=int(len(df)),
               wall_s=round(wall, 1),
               t_acquire_s=round(tot("time_acquire"), 2),
               t_fit_s=round(tot("time_fit"), 2),
               # Record EVERY timing column: wall must be fully accounted for.
               # time_mc is GPry's FINAL posterior extraction (the `mc=` sampler,
               # identical in every arm) and is NOT part of the acquisition loop --
               # including it in a per-sampler comparison is meaningless.
               t_mc_s=round(tot("time_mc"), 2),
               t_truth_s=round(tot("time_truth"), 2),
               t_convergence_s=round(tot("time_convergence"), 2),
               t_loop_s=round(tot("time_acquire") + tot("time_fit"), 2),
               evals_mc_total=int(np.nansum(df.get("evals_mc", [0]))),
               evals_acquire_total=int(np.nansum(df.get("evals_acquire", [0]))),
               kernel=str(sur.gpr.kernel_), **{k: v for k, v in rec.items()
                                               if k != "mean"})
    with open(os.path.join(outdir, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[CMP] {tag}: n={res['n_total']} conv={converged} wall={res['wall_s']}s "
          f"| LOOP={res['t_loop_s']}s (acq={res['t_acquire_s']} fit={res['t_fit_s']}) "
          f"finalMC={res['t_mc_s']}s | "
          f"KL={rec['kl_nuts']} max|mean|={rec['max_mean_in_sigma']}sig "
          f"std_relerr={rec['std_relerr_nuts']} ESS={rec['ess']:.0f}", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
