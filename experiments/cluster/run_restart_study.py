"""
Restart-efficiency study: does a cheaper hyperparameter-fit budget stay robust?

Motivation.  In the d=16/d=30 sampler comparison the hyperparameter fit was the
dominant cost of the acquisition loop -- 52% of the loop at d=16 and 85% at
d=30 -- and a bench measurement on a real d=16 surrogate found that all restart
strategies converge to the SAME optimum (LML=460.2), with the two informed
starts (previous theta, covariance guess) doing all the work: dropping to
n_restarts=2 was 47x faster for an identical fit.  This study asks whether that
holds up across dimension AND on targets where a wrong length scale actually
breaks the posterior, rather than only on easy Gaussians.

Arms (`restart_strategy` x `n_restarts_optimizer`):
    S0  uniform 10+2d   -- control: GPry's current default
    S1  uniform 8       -- fewer restarts, unchanged draw
    S2  uniform 2       -- informed starts only (the 47x candidate)
    S3  local   8       -- log-normal around the covariance guess
    S4  screen  8       -- draw 8x, rank by one gradient-free LML, keep the best

Targets: d=5 uses NON-Gaussian targets (a chained banana and a 4-mode mixture)
so the study measures robustness, not just speed; d=8/16/30 use the anisotropic
Gaussian for the speed/scaling curve.

Usage:
    python run_restart_study.py <target> <d> <arm> <seed> [outdir] [max_total]
      <target> = gauss | curved | multimode
      <arm>    = S0 | S1 | S2 | S3 | S4
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


# strategy, n_restarts ("Xd" strings are resolved against d by GPry)
ARMS = {
    "S0": ("uniform", None),   # None -> 10 + 2d, the current default
    "S1": ("uniform", 8),
    "S2": ("uniform", 2),      # informed starts only
    "S3": ("local", 8),
    "S4": ("screen", 8),
}

# Point budget. The d=5 non-Gaussian targets get a larger budget than a
# unimodal Gaussian of the same dimension would need: four separated basins (or
# a curved ridge) simply need more points to resolve.
MAX_TOTAL = {("gauss", 8): 900, ("gauss", 16): 1600, ("gauss", 30): 3000,
             ("curved", 5): 600, ("multimode", 5): 600}


# Target difficulty is calibrated so that the CONTROL arm (S0) actually solves
# the target: if the control already fails, a cheaper arm failing carries no
# information about the restart budget. Overridable for difficulty sweeps.
CURVED_B = float(os.environ.get("RST_B", 0.35))
MULTIMODE_SEP = float(os.environ.get("RST_SEP", 4.0))
MULTIMODE_K = int(os.environ.get("RST_K", 4))


def build_target(kind, d):
    if kind == "gauss":
        return C.make_gaussian(d)
    if kind == "curved":
        return C.make_curved(d, b=CURVED_B)
    if kind == "multimode":
        return C.make_multimode(d, n_modes=MULTIMODE_K, sep=MULTIMODE_SEP)
    raise ValueError(f"unknown target {kind!r}")


def score(kind, sur, tgt, nuts_kwargs):
    if kind == "gauss":
        return C.evaluate_recovery(sur, tgt["bounds"], tgt["mean"], tgt["cov"],
                                   tgt["marg"], nuts_kwargs=nuts_kwargs)
    if kind == "curved":
        return C.evaluate_recovery_curved(sur, tgt, nuts_kwargs=nuts_kwargs)
    return C.evaluate_recovery_mixture(sur, tgt, nuts_kwargs=nuts_kwargs)


def reference_samples(kind, tgt, n=40000):
    if kind == "gauss":
        return np.random.default_rng(1).multivariate_normal(
            tgt["mean"], tgt["cov"], n)
    if kind == "curved":
        return C.curved_reference_samples(tgt, n=n, seed=1)
    return C.mixture_reference_samples(tgt, n=n, seed=1)


def corner_all_dims(kind, tgt, X, w, path, label):
    """Corner over ALL dimensions -- never a subset."""
    import matplotlib
    matplotlib.use("Agg")
    from getdist import MCSamples, plots
    d = tgt["d"]
    names = [f"x{i}" for i in range(d)]
    ref = reference_samples(kind, tgt)
    mcs = [MCSamples(samples=ref, names=names, label="truth"),
           MCSamples(samples=np.asarray(X), weights=np.asarray(w), names=names,
                     label=f"GP ({label})")]
    g = plots.get_subplot_plotter(width_inch=min(2 + 0.7 * d, 26))
    g.triangle_plot(mcs, filled=[True, False], legend_labels=[m.label for m in mcs])
    g.export(path)


def main():
    kind = sys.argv[1].lower()
    d = int(sys.argv[2])
    arm = sys.argv[3].upper()
    seed = int(sys.argv[4])
    outdir = sys.argv[5] if len(sys.argv) > 5 else "restart_out"
    max_total = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    strategy, n_restarts = ARMS[arm]
    if n_restarts is None:
        n_restarts = 10 + 2 * d
    os.makedirs(outdir, exist_ok=True)
    if not max_total:
        max_total = MAX_TOTAL.get((kind, d), int(70 * d ** 1.5))

    tgt = build_target(kind, d)
    n_initial = 3 * d
    r = Runner(
        tgt["logLkl"], tgt["bounds"].tolist(), ref_bounds=tgt["ref_bounds"].tolist(),
        surrogate={"regressor": {"kernel": "RBF",
                                 "n_restarts_optimizer": n_restarts,
                                 "restart_strategy": strategy}},
        gp_acquisition={"NORA": {"sampler": "nuts", "mc_every": 1}},
        mc={"nuts": {}},
        options={"n_initial": n_initial,
                 "max_initial": min(3 * n_initial, max_total),
                 "max_total": max_total},
        seed=seed, verbose=1, checkpoint=None,
    )
    r.acquisition._nuts_kwargs = {"pad_multiple": 512, "max_num_doublings": 7}
    # Assert the arm actually took effect: a silently-ignored option would make
    # every arm identical and the whole study a null result.
    got_s = r.surrogate.gpr.restart_strategy
    got_n = r.surrogate.gpr.n_restarts_optimizer
    if (got_s, got_n) != (strategy, n_restarts):
        raise SystemExit(f"ARM NOT APPLIED: asked ({strategy},{n_restarts}) "
                         f"got ({got_s},{got_n})")
    print(f"[ARM] {arm}: strategy={got_s} n_restarts={got_n} "
          f"target={kind} d={d} seed={seed} max_total={max_total}", flush=True)

    t0 = time.time()
    converged, err = False, None
    try:
        r.run()
        converged = bool(getattr(r, "has_converged", False))
    except Exception as exc:
        import traceback
        err = repr(exc)
        print("run() ended:\n" + traceback.format_exc(), flush=True)
    wall = time.time() - t0

    tag = f"{kind}_d{d}_{arm}_seed{seed}"
    sur = r.surrogate
    C.save_gp(deepcopy(sur), os.path.join(outdir, f"{tag}_surrogate.pkl"))
    r.progress.data.to_csv(os.path.join(outdir, f"{tag}_iters.csv"), index=False)

    nuts_kwargs = dict(n_chains=64, n_warmup=300, n_samples=300)
    rec, (X, w) = score(kind, sur, tgt, nuts_kwargs)

    df = r.progress.data
    tot = lambda c: float(np.nansum(df[c].values)) if c in df else float("nan")
    res = dict(
        target=kind, d=d, arm=arm, strategy=strategy, n_restarts=int(n_restarts),
        target_param=(CURVED_B if kind == "curved" else
                      MULTIMODE_SEP if kind == "multimode" else None),
        seed=seed, converged=converged, error=err,
        n_total=int(sur.n_total), n_iterations=int(len(df)),
        wall_s=round(wall, 1),
        t_loop_s=round(tot("time_acquire") + tot("time_fit"), 2),
        t_fit_s=round(tot("time_fit"), 2),
        t_acquire_s=round(tot("time_acquire"), 2),
        t_mc_s=round(tot("time_mc"), 2),
        # The headline cost metric: LML evaluations spent on hyperparameter fits.
        evals_fit_total=int(np.nansum(df.get("evals_fit", [0]))),
        evals_acquire_total=int(np.nansum(df.get("evals_acquire", [0]))),
        lml_final=float(sur.gpr.log_marginal_likelihood_value_)
        if hasattr(sur.gpr, "log_marginal_likelihood_value_") else None,
        kernel=str(sur.gpr.kernel_),
        **{k: v for k, v in rec.items() if k != "mean"})
    with open(os.path.join(outdir, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)

    try:
        corner_all_dims(kind, tgt, X, w, os.path.join(outdir, f"{tag}_corner.png"), arm)
    except Exception as exc:
        print(f"  [warn] corner plot failed: {exc!r}", flush=True)

    head = {"gauss": f"KL={rec.get('kl_nuts')}",
            "curved": f"KL_z={rec.get('kl_z')} curv={rec.get('curv_gp')}"
                      f"/{rec.get('curv_true')}",
            "multimode": f"modes={rec.get('n_modes_found')}/{rec.get('n_modes_true')}"
                         f" w_relerr={rec.get('w_relerr')}"}[kind]
    print(f"[RST] {tag}: n={res['n_total']} conv={converged} "
          f"loop={res['t_loop_s']}s fit={res['t_fit_s']}s "
          f"evals_fit={res['evals_fit_total']} LML={res['lml_final']} | {head}",
          flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
