"""
Aggregate the restart-efficiency study: per (target, d, arm), the COST of the
hyperparameter fit and the QUALITY of the resulting posterior.

The study only supports a "cheaper is fine" conclusion if BOTH hold: cost drops
AND quality is statistically indistinguishable from the S0 control. Quality is
therefore compared arm-vs-control per target, not just tabulated.

Usage:  python aggregate_restart.py <results_dir> [out.csv]
"""
import os
import sys
import glob
import json

import numpy as np
import pandas as pd

# Per-target headline quality metric, and whether SMALLER is better.
QUALITY = {"gauss": ("kl_nuts", True),
           "curved": ("kl_z", True),
           "multimode": ("w_relerr", True)}

# Recovery gate: the quality above which a run has NOT recovered the posterior,
# whatever GPry's convergence criterion said.
#
# Why this is needed: `converged=True, error=None` is not by itself proof that the
# posterior was recovered, and nothing downstream was checking. This gate is a
# GUARD, not a correction of any observed result.
#
# Honesty note on its origin: it was prompted by a run that reported
# converged=True with kl_nuts=143.48, but that run came from job 476785, which was
# invalidated by the srun duplicate-rank bug (4 racing ranks per task). It is
# therefore NOT evidence of a real UltraNest failure mode and must not be cited as
# such. The gate is kept because the validation below stands independently of it.
#
# Why 0.5 is not a tuning knob: across all 175 pre-existing runs the metric is
# bimodal with a ~2000x empty gap -- every recovered run is <= 0.1283 and every
# broken one is >= 276.06. ANY threshold in [0.2, 200] classifies identically.
# Validated on those 175 NUTS-only runs, where the gate changes nothing: it
# excludes exactly the 5 runs that already had error != None, and finds zero
# false convergences. It is applied identically to every sampler and arm.
RECOVERY_MAX = {"gauss": 0.5, "curved": 0.5, "multimode": 0.5}


def load(results_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if f.endswith("_iters.json"):
            continue
        try:
            rows.append(json.load(open(f)))
        except Exception as exc:
            print(f"  [warn] unreadable {os.path.basename(f)}: {exc!r}")
    return pd.DataFrame(rows)


def main():
    results_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "restart_summary.csv"
    df = load(results_dir)
    if df.empty:
        raise SystemExit(f"no results in {results_dir}")
    print(f"loaded {len(df)} runs\n")

    # A crashed run (GPAcquisitionError) stops early, so its t_fit_s is the cost of
    # a few iterations, not the cost of a fit. Including it in a median silently
    # makes the CONTROL look cheap and understates the speedup (d=30: 640 -> 609 s,
    # 5.07x -> 4.82x). Cost and quality are therefore medianed over SUCCEEDED runs
    # only; failures are reported separately as n_failed.
    if "error" not in df:
        df["error"] = None
    if "sampler" not in df:
        df["sampler"] = "nuts"          # every pre-08-19 campaign was NUTS-only
    if "timed_out" not in df:
        df["timed_out"] = False
    df["timed_out"] = df["timed_out"].fillna(False).astype(bool)
    df["failed"] = df["error"].notna()

    # A run counts as a cost data point only if it actually recovered the
    # posterior. "Converged" is GPry's stopping criterion; "recovered" is
    # measured against the truth.
    def _recovered(row):
        if row["failed"] or row["timed_out"]:
            return False
        qcol, _ = QUALITY[row["target"]]
        v = row.get(qcol)
        if v is None or not np.isfinite(v):
            return False
        return bool(v <= RECOVERY_MAX[row["target"]])

    df["recovered"] = df.apply(_recovered, axis=1)
    _fc = df[df.converged & ~df.failed & ~df.recovered]
    if len(_fc):
        print(f"  !! {len(_fc)} FALSE CONVERGENCE(S): converged=True but the "
              f"posterior is not recovered")
        for r in _fc.itertuples():
            qcol, _ = QUALITY[r.target]
            print(f"     {r.target} d={r.d} {r.sampler} seed{r.seed}: "
                  f"{qcol}={getattr(r, qcol):.4g} at n={r.n_total} "
                  f"({r.t_loop_s:.0f}s loop)")
        print()

    # Which axis is under test? The restart study varies `arm` at fixed sampler;
    # the head-to-head varies `sampler` at fixed arm. Picking the wrong one would
    # average the two arms of the comparison together into a fake null result.
    if df.sampler.nunique() > 1:
        KEY, CONTROL = "sampler", "nuts"
    else:
        KEY, CONTROL = "arm", "S0"
    print(f"comparison axis: {KEY} (control = {CONTROL}); "
          f"levels = {sorted(df[KEY].unique())}\n")

    recs = []
    for (kind, d), g in df.groupby(["target", "d"]):
        qcol, _ = QUALITY[kind]
        gok = g[g.recovered]
        ctrl = gok[gok[KEY] == CONTROL]
        for arm, ga_all in sorted(g.groupby(KEY)):
            ga = ga_all[ga_all.recovered]
            if not len(ga):
                print(f"  [warn] {kind} d={d} {arm}: 0 of {len(ga_all)} runs "
                      f"recovered the posterior")
                recs.append(dict(target=kind, d=int(d), **{KEY: arm},
                                 n_seeds=len(ga_all),
                                 n_failed=int(ga_all.failed.sum()),
                                 n_timeout=int(ga_all.timed_out.sum()),
                                 n_converged=int(ga_all.converged.sum()),
                                 n_recovered=0))
                continue
            # KEY is whichever of arm/sampler is under test; record the OTHER
            # one too, so a summary row is self-describing.
            other = "arm" if KEY == "sampler" else "sampler"
            r = dict(target=kind, d=int(d), **{KEY: arm},
                     **{other: ga[other].iloc[0]},
                     strategy=ga.strategy.iloc[0],
                     n_restarts=int(ga.n_restarts.iloc[0]),
                     n_seeds=len(ga_all),
                     n_failed=int(ga_all.failed.sum()),
                     n_timeout=int(ga_all.timed_out.sum()),
                     n_converged=int(ga_all.converged.sum()),
                     n_recovered=int(len(ga)),
                     n_false_conv=int((ga_all.converged & ~ga_all.failed
                                       & ~ga_all.recovered).sum()),
                     n_total_med=float(ga.n_total.median()),
                     t_fit_med=float(ga.t_fit_s.median()),
                     t_loop_med=float(ga.t_loop_s.median()),
                     evals_fit_med=float(ga.evals_fit_total.median()),
                     lml_med=float(ga.lml_final.median()))
            if qcol in ga:
                r["quality"] = qcol
                r["quality_med"] = float(ga[qcol].median())
                r["quality_max"] = float(ga[qcol].max())
            if kind == "multimode":
                r["modes_min"] = int(ga.n_modes_found.min())
                r["modes_med"] = float(ga.n_modes_found.median())
            # Speedups relative to the control, on the SAME target and d.
            if len(ctrl):
                r["fit_speedup_vs_ctrl"] = round(
                    float(ctrl.t_fit_s.median() / max(ga.t_fit_s.median(), 1e-9)), 2)
                r["loop_speedup_vs_ctrl"] = round(
                    float(ctrl.t_loop_s.median() / max(ga.t_loop_s.median(), 1e-9)), 2)
            recs.append(r)
    S = pd.DataFrame(recs).sort_values(["target", "d", KEY])
    S.to_csv(out, index=False)

    for (kind, d), g in S.groupby(["target", "d"]):
        print(f"=== {kind} d={d} ===")
        cols = [KEY, "n_seeds", "n_failed", "n_timeout", "n_false_conv",
                "n_converged", "n_recovered", "n_total_med", "t_fit_med",
                "t_loop_med", "fit_speedup_vs_ctrl", "loop_speedup_vs_ctrl",
                "evals_fit_med", "quality_med", "quality_max"]
        if kind == "multimode":
            cols += ["modes_min", "modes_med"]
        print(g[[c for c in cols if c in g]].to_string(index=False))
        qcol, _ = QUALITY[kind]
        base = g[g[KEY] == CONTROL]
        if len(base) and "quality_med" in g and base.quality_med.notna().any():
            b = base.quality_med.iloc[0]
            worse = g[(g[KEY] != CONTROL) & (g.quality_med > 2 * max(b, 1e-6))]
            if len(worse):
                print(f"  !! quality REGRESSION vs {CONTROL} ({qcol} med {b:.4g}): "
                      + ", ".join(f"{getattr(r, KEY)}={r.quality_med:.4g}"
                                  for r in worse.itertuples()))
            else:
                print(f"  quality OK vs {CONTROL} ({qcol} med {b:.4g}) "
                      f"for all levels")
        print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
