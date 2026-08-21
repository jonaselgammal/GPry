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

The ACQUISITION sampler is selected with RST_SAMPLER (default "nuts"); the final
posterior MC is always matched to it (nuts->nuts, ultranest->ultranest), because
scoring an UltraNest acquisition with a NUTS read-out would not be a comparison
of acquisition at all.  RST_MAXWALL sets a soft wall-clock limit in seconds: on
expiry the run raises, so a result JSON with error="TIMEOUT" is still written
rather than SLURM killing the task and leaving nothing behind.
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

# Acquisition sampler, and the FINAL MC matched to it.  "nested" is deliberately
# NOT used: it resolves to "PolyChord if importable, else UltraNest", so an
# UltraNest arm could silently be scored with a PolyChord read-out.  Naming the
# interface explicitly makes the arm reproducible.
GRADIENT_SAMPLERS = ("nuts", "hmc", "blackjax")


# Point budget. The d=5 non-Gaussian targets get a larger budget than a
# unimodal Gaussian of the same dimension would need: four separated basins (or
# a curved ridge) simply need more points to resolve.
MAX_TOTAL = {("gauss", 8): 900, ("gauss", 16): 1600, ("gauss", 30): 3000,
             ("curved", 5): 300, ("multimode", 5): 300}

# The d=5 non-Gaussian arms run to a FIXED budget with convergence disabled.
# Reason: on the multimode target GPry's convergence criterion fires at wildly
# inconsistent points (control arm stopped at n = 95, 115 and 280 across three
# seeds, with the mode-weight error swinging 0.056 -> 0.79), so a
# quality-at-convergence comparison is dominated by *when the criterion fired*
# rather than by the hyperparameter fit under test.  Matching the budget
# removes that noise and makes the robustness question well posed.  The
# Gaussian arms keep natural convergence -- that is where the SPEED claim
# lives, and there convergence is stable (d=16: n=304 on all three v3 seeds).
# Only the MULTIMODE target needs this. The single-bend banana already recovers
# with very low seed variance under natural convergence, and disabling
# convergence there just pushes the run past GP saturation into
# GPAcquisitionError ("Acquisition returning no values"), which is a different
# arbitrary stopping rule rather than a matched budget.
FIXED_BUDGET = set(os.environ.get("RST_FIXED", "multimode").split(",")) - {""}


# Target difficulty is calibrated so that the CONTROL arm (S0) actually solves
# the target: if the control already fails, a cheaper arm failing carries no
# information about the restart budget. Overridable for difficulty sweeps.
CURVED_B = float(os.environ.get("RST_B", 1.2))
CURVED_NTWIST = int(os.environ.get("RST_NTWIST", 1))
MULTIMODE_SEP = float(os.environ.get("RST_SEP", 4.0))
MULTIMODE_K = int(os.environ.get("RST_K", 4))


def build_target(kind, d):
    if kind == "gauss":
        return C.make_gaussian(d)
    if kind == "curved":
        return C.make_curved(d, b=CURVED_B, n_twist=CURVED_NTWIST)
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
    sampler = os.environ.get("RST_SAMPLER", "nuts").lower()
    final_mc = sampler          # matched, and named explicitly (never "nested")
    strategy, n_restarts = ARMS[arm]
    if n_restarts is None:
        n_restarts = 10 + 2 * d
    os.makedirs(outdir, exist_ok=True)
    if not max_total:
        max_total = MAX_TOTAL.get((kind, d), int(70 * d ** 1.5))

    tgt = build_target(kind, d)
    # The sampler is part of the identity of a run, so it is part of the tag;
    # otherwise a nuts and an ultranest arm would overwrite each other.
    tag = (f"{kind}_d{d}_{arm}_seed{seed}" if sampler == "nuts"
           else f"{kind}_d{d}_{arm}_{sampler}_seed{seed}")
    # Per-iteration convergence checkpointing. Under DontConverge (used to match
    # arms on evaluation budget) the run records nothing about where it WOULD
    # have stopped, so answering that -- or drawing a convergence curve -- would
    # otherwise mean re-running everything. Saving the surrogate state and the
    # acquisition chain at each point the criterion would have been evaluated
    # makes both recoverable a posteriori, for any criterion. Default ON;
    # RST_NO_CKPT=1 disables.
    conv_ckpt = None
    if not os.environ.get("RST_NO_CKPT"):
        conv_ckpt = C.ConvergenceCheckpointer(
            outdir, tag,
            pickle_every=int(os.environ.get("RST_CKPT_PICKLE_EVERY", 0)),
        )
    n_initial = 3 * d
    opts = {"n_initial": n_initial, "max_initial": min(3 * n_initial, max_total),
            "max_total": max_total}
    # The per-iteration "simple" fit is a full L-BFGS from the current theta and
    # costs 15 s each at d=30 (vs 0.08 s at d=8), forming a floor that no restart
    # policy can touch. RST_SIMPLE_EVERY thins it.
    if os.environ.get("RST_SIMPLE_EVERY"):
        opts["fit_simple_every"] = int(os.environ["RST_SIMPLE_EVERY"])
    if os.environ.get("RST_FULL_EVERY"):
        opts["fit_full_every"] = float(os.environ["RST_FULL_EVERY"])
    # Length-scale prior. The merged default is [1e-3, 1e2]; at d=30 the fit
    # rails against that ceiling (a Gaussian log-posterior is quadratic, so the
    # RBF legitimately wants a very long correlation length), so allow it to be
    # widened for the diagnosis.
    ls_max = float(os.environ.get("RST_LSMAX", 0)) or None
    reg = {"kernel": "RBF", "n_restarts_optimizer": n_restarts,
           "restart_strategy": strategy}
    # L-BFGS ftol; SciPy's default sits far below the LML's ~3e-5 noise floor.
    if os.environ.get("RST_FTOL"):
        reg["optimizer_ftol"] = float(os.environ["RST_FTOL"])
    if ls_max:
        reg["length_scale_prior"] = [1e-3, ls_max]
    r = Runner(
        tgt["logLkl"], tgt["bounds"].tolist(), ref_bounds=tgt["ref_bounds"].tolist(),
        surrogate={"regressor": reg},
        gp_acquisition={"NORA": {"sampler": sampler, "mc_every": 1}},
        mc={final_mc: {}},
        options=opts,
        convergence_criterion=("DontConverge" if kind in FIXED_BUDGET else None),
        callback=conv_ckpt,
        seed=seed, verbose=1, checkpoint=None,
    )
    if sampler in ("nuts", "hmc"):
        r.acquisition._nuts_kwargs = {"pad_multiple": 512, "max_num_doublings": 7}
    # Assert the arm actually took effect: a silently-ignored option would make
    # every arm identical and the whole study a null result.
    got_s = r.surrogate.gpr.restart_strategy
    got_n = r.surrogate.gpr.n_restarts_optimizer
    if (got_s, got_n) != (strategy, n_restarts):
        raise SystemExit(f"ARM NOT APPLIED: asked ({strategy},{n_restarts}) "
                         f"got ({got_s},{got_n})")
    # Same for the sampler axis. `Runner` accepts the acquisition dict wholesale,
    # so a typo or an unsupported name would fall back to the default sampler and
    # make both arms of the head-to-head identical -- a fabricated null result.
    got_acq = getattr(r.acquisition, "sampler", None)
    if str(got_acq).lower() != sampler:
        raise SystemExit(f"SAMPLER NOT APPLIED: asked {sampler!r} "
                         f"got {got_acq!r}")
    # A nested sampler must also have a live interface object; the gradient
    # samplers legitimately have none (they are not nested samplers).
    got_iface = type(getattr(r.acquisition, "sampler_interface", None)).__name__
    if sampler not in GRADIENT_SAMPLERS and got_iface == "NoneType":
        raise SystemExit(f"SAMPLER NOT APPLIED: {sampler!r} has no ns interface")
    print(f"[ARM] {arm}: strategy={got_s} n_restarts={got_n} "
          f"target={kind} d={d} seed={seed} max_total={max_total}", flush=True)
    print(f"[SAMPLER] acquisition={got_acq} interface={got_iface} "
          f"final_mc={final_mc}", flush=True)

    # Soft wall-clock limit, below the SLURM --time cap, so a task that would be
    # killed instead raises and still writes a JSON. A timeout is a RESULT (the
    # configuration did not finish in the budget), not a missing data point.
    max_wall = float(os.environ.get("RST_MAXWALL", 0))
    if max_wall:
        import signal

        def _timeout(signum, frame):
            raise TimeoutError(f"TIMEOUT after {max_wall:.0f}s soft wall limit")

        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(int(max_wall))
        print(f"[LIMIT] soft wall limit {max_wall:.0f}s armed", flush=True)

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
    if max_wall:
        import signal
        signal.alarm(0)          # scoring/plotting must not be interrupted

    sur = r.surrogate
    C.save_gp(deepcopy(sur), os.path.join(outdir, f"{tag}_surrogate.pkl"))
    r.progress.data.to_csv(os.path.join(outdir, f"{tag}_iters.csv"), index=False)

    nuts_kwargs = dict(n_chains=64, n_warmup=300, n_samples=300)
    rec, (X, w) = score(kind, sur, tgt, nuts_kwargs)

    df = r.progress.data
    tot = lambda c: float(np.nansum(df[c].values)) if c in df else float("nan")
    res = dict(
        target=kind, d=d, arm=arm, sampler=sampler, final_mc=final_mc,
        timed_out=bool(err and "TIMEOUT" in err),
        strategy=strategy, n_restarts=int(n_restarts),
        ls_max=ls_max, optimizer_ftol=reg.get("optimizer_ftol"),
        fit_simple_every=opts.get("fit_simple_every", 1),
        fit_full_every=opts.get("fit_full_every"),
        target_param=(CURVED_B if kind == "curved" else
                      MULTIMODE_SEP if kind == "multimode" else None),
        seed=seed, converged=converged, error=err,
        fixed_budget=bool(kind in FIXED_BUDGET),
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

    # Persist the read-out chain itself. Without this the corner plot can only be
    # remade by re-running the NUTS read-out on the surrogate, so a plotting tweak
    # costs a compute job. With it, replot_corner.py regenerates from disk.
    try:
        np.savez_compressed(os.path.join(outdir, f"{tag}_readout.npz"),
                            X=np.asarray(X), w=np.asarray(w, float))
    except Exception as exc:
        print(f"  [warn] could not save read-out samples: {exc!r}", flush=True)

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
