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

    recs = []
    for (kind, d), g in df.groupby(["target", "d"]):
        qcol, _ = QUALITY[kind]
        ctrl = g[g.arm == "S0"]
        for arm, ga in sorted(g.groupby("arm")):
            r = dict(target=kind, d=int(d), arm=arm,
                     strategy=ga.strategy.iloc[0],
                     n_restarts=int(ga.n_restarts.iloc[0]),
                     n_seeds=len(ga),
                     n_converged=int(ga.converged.sum()),
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
                r["fit_speedup_vs_S0"] = round(
                    float(ctrl.t_fit_s.median() / max(ga.t_fit_s.median(), 1e-9)), 2)
                r["loop_speedup_vs_S0"] = round(
                    float(ctrl.t_loop_s.median() / max(ga.t_loop_s.median(), 1e-9)), 2)
            recs.append(r)
    S = pd.DataFrame(recs).sort_values(["target", "d", "arm"])
    S.to_csv(out, index=False)

    for (kind, d), g in S.groupby(["target", "d"]):
        print(f"=== {kind} d={d} ===")
        cols = ["arm", "strategy", "n_restarts", "n_seeds", "n_converged",
                "t_fit_med", "fit_speedup_vs_S0", "loop_speedup_vs_S0",
                "evals_fit_med", "quality_med", "quality_max"]
        if kind == "multimode":
            cols += ["modes_min", "modes_med"]
        print(g[[c for c in cols if c in g]].to_string(index=False))
        qcol, _ = QUALITY[kind]
        base = g[g.arm == "S0"]
        if len(base) and "quality_med" in g:
            b = base.quality_med.iloc[0]
            worse = g[(g.arm != "S0") & (g.quality_med > 2 * max(b, 1e-6))]
            if len(worse):
                print(f"  !! quality REGRESSION vs S0 ({qcol} med {b:.4g}): "
                      + ", ".join(f"{r.arm}={r.quality_med:.4g}"
                                  for r in worse.itertuples()))
            else:
                print(f"  quality OK vs S0 ({qcol} med {b:.4g}) for all arms")
        print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
