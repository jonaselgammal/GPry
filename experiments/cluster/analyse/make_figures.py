"""Build the paper figures.

Two families, deliberately kept apart because they come from DIFFERENT code
versions and DIFFERENT scheduling conditions:

  settled/   from results/paper_runs/final/ (50 runs, --exclusive),
             results/paper_runs/restart/ (125 runs), results/paper_runs/h2h/
             and results/paper_runs/v3/. This is the current, defensible data.
  legacy/    from results/other_runs/legacy_csv/ (the 2026-08-10 Tier-B
             campaign, aggregated by legacy/aggregate_results.py). Predates the
             `length_scale_prior` fix (PR #4) AND the matched-final-MC fix, and
             was run WITHOUT --exclusive, so its wall-clock numbers are not
             comparable to anything current. Kept only so the old figures can
             still be rebuilt; do not quote their timings.

Usage:  python make_figures.py [settled|legacy|all]   (default: all)

Outputs go to results/figures/ (settled/ and legacy/ subdirs) and the paper-facing
ones are copied to paper/figures/.

Reads only; runs no science.
"""
import os
import csv
import sys
import glob
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths. This script only READS run products and WRITES figures; it never runs
# science. It lives in GPry/experiments/cluster/analyse/, and the run products
# live in the project-level results/ tree, so both anchors are derived from the
# project root rather than from this file's own directory.
#   results/paper_runs/  final/ restart/ h2h/ v3/   <- the defensible campaigns
#   results/other_runs/  raw/ legacy_csv/ ...       <- superseded campaigns
_HERE = os.path.dirname(os.path.abspath(__file__))
# analyse -> cluster -> experiments -> GPry -> <project root>
REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))
RESULTS = os.path.join(REPO, "results")
SETTLED = os.path.join(RESULTS, "paper_runs")
LEGACY_CSV = os.path.join(RESULTS, "other_runs", "legacy_csv")
FIG = os.path.join(RESULTS, "figures")
FIG_S = os.path.join(FIG, "settled")
FIG_L = os.path.join(FIG, "legacy")
PAPER_FIG = os.path.join(REPO, "paper", "figures")
for _d in (FIG, FIG_S, FIG_L):
    os.makedirs(_d, exist_ok=True)

C = {"nuts": "tab:red", "ultranest": "tab:blue", "blackjax": "tab:purple"}
LB = {"nuts": "NUTS", "ultranest": "nested (UltraNest)", "blackjax": "BlackJAX NUTS"}
C_ARM = {"BASE": "tab:blue", "PROP": "tab:red"}

# Per-target headline quality metric (smaller is better in all three).
QUALITY = {"gauss": "kl_nuts", "curved": "kl_z", "multimode": "w_relerr"}
# Recovery gate -- see aggregate_restart.RECOVERY_MAX for the full rationale.
# `converged=True, error=None` is not by itself proof of recovery, and nothing was
# checking. (The run that prompted this, kl_nuts=143.5, came from the invalidated
# job 476785 and is NOT evidence of a real failure mode.) Across 175 runs the
# metric is bimodal
# with a ~2000x empty gap (recovered <= 0.128, broken >= 276), so any threshold
# in [0.2, 200] classifies identically. Applied identically to every sampler.
RECOVERY_MAX = 0.5


def recovered(r):
    """Did this run actually recover the posterior (not merely stop)?"""
    if r.get("failed") or r.get("timed_out"):
        return False
    v = r.get(QUALITY.get(r.get("target"), ""))
    return v is not None and np.isfinite(v) and v <= RECOVERY_MAX
# Display order and labels for the five final/restart cases.
CASES = [("curved", 5), ("gauss", 8), ("gauss", 16), ("multimode", 5), ("gauss", 30)]
CASE_LB = {("curved", 5): "banana\n$d=5$", ("gauss", 8): "Gauss\n$d=8$",
           ("gauss", 16): "Gauss\n$d=16$", ("multimode", 5): "4-mode\n$d=5$",
           ("gauss", 30): "Gauss\n$d=30$"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def read_csv(path):
    with open(path) as fh:
        return [{k: (float(v) if v not in ("", "None") and _num(v) else v)
                 for k, v in row.items()} for row in csv.DictReader(fh)]


def load_runs(*patterns):
    """Load run JSONs. Tags each with `failed` (crashed with GPAcquisitionError)."""
    out = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            if f.endswith("_iters.json"):
                continue
            r = json.load(open(f))
            r["_file"] = f
            r["failed"] = bool(r.get("error"))
            out.append(r)
    return out


def sel(runs, target=None, d=None, arm=None, ok_only=True):
    out = [r for r in runs
           if (target is None or r.get("target") == target)
           and (d is None or r.get("d") == d)
           and (arm is None or r.get("arm") == arm)]
    return [r for r in out if not r["failed"]] if ok_only else out


def med(runs, key):
    v = [r[key] for r in runs if r.get(key) is not None]
    return float(np.median(v)) if v else np.nan


def save(fig, name, outdir, also_paper=False):
    fig.tight_layout()
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, dpi=140)
        paths.append(p)
    plt.close(fig)
    if also_paper:
        shutil.copy(os.path.join(outdir, f"{name}.pdf"),
                    os.path.join(PAPER_FIG, f"{name}.pdf"))
    return paths


# =========================================================================== #
# SETTLED figures: results/paper_runs/{final,restart,h2h,v3}
# =========================================================================== #
FINAL = None
RESTART = None
V3 = None
H2H = None


def _load_settled():
    global FINAL, RESTART, V3, H2H
    FINAL = {"BASE": load_runs(os.path.join(SETTLED, "final/BASE/*.json")),
             "PROP": load_runs(os.path.join(SETTLED, "final/PROP/*.json"))}
    RESTART = load_runs(os.path.join(SETTLED, "restart/*.json"))
    H2H = load_runs(os.path.join(SETTLED, "h2h/*.json"))
    V3 = []
    for f in sorted(glob.glob(os.path.join(SETTLED, "v3/v3_out/*.json"))):
        r = json.load(open(f))
        r["_file"] = f
        # v3 has no `error` field; a run that stopped without converging at a
        # KL of order 10-300 is a failure, not a result.
        r["failed"] = (not r["converged"]) and r["kl_nuts"] > 1.0
        V3.append(r)


# --- S1: cost of the hyperparameter fit and of the whole loop -------------- #
def fig_final_cost():
    """50-run exclusive head-to-head: fit time and loop time, BASE vs PROP."""
    x = np.arange(len(CASES))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, key, title in ((axes[0], "t_fit_s", "GP hyperparameter fit"),
                           (axes[1], "t_loop_s", "full acquisition loop")):
        for i, arm in enumerate(("BASE", "PROP")):
            v = [med(sel(FINAL[arm], t, d), key) for t, d in CASES]
            ax.bar(x + (i - 0.5) * 0.36, v, 0.34, color=C_ARM[arm],
                   label={"BASE": "BASE (current defaults)",
                          "PROP": "PROP (2 restarts, ceiling $10^3$, ftol $10^{-5}$)"}[arm])
        b = np.array([med(sel(FINAL["BASE"], t, d), key) for t, d in CASES])
        p = np.array([med(sel(FINAL["PROP"], t, d), key) for t, d in CASES])
        for xi, (bi, pi) in enumerate(zip(b, p)):
            ax.text(xi, max(bi, pi) * 1.35, f"{bi / pi:.2f}$\\times$",
                    ha="center", fontsize=8.5)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([CASE_LB[c] for c in CASES], fontsize=8)
        ax.set_ylabel("median wall time [s]")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y", which="both")
        ax.set_ylim(top=ax.get_ylim()[1] * 4)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Fit cost falls 5-11$\\times$; the loop gains only 1.1-1.5$\\times$ "
                 "(50 runs, exclusive nodes)", fontsize=10)
    return save(fig, "fig_final_cost", FIG_S, also_paper=True)


# --- S2: where the loop time actually goes --------------------------------- #
def fig_final_budget():
    """Stacked wall-time budget: the fit stops being the bottleneck, acquisition is.

    NOTE on the accounting: in the run JSONs `t_loop_s = t_fit_s + t_acquire_s`
    exactly (the final MC is NOT inside the loop), and
    `wall_s ~= t_loop_s + t_mc_s`. Shares are therefore taken of `wall_s`, so the
    three components add to ~100% instead of overflowing the bar.
    """
    x = np.arange(len(CASES))
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    keys = [("t_fit_s", "GP hyperparameter fit", "tab:orange"),
            ("t_acquire_s", "acquisition", "tab:green"),
            ("t_mc_s", "final MC", "tab:grey")]
    w = 0.34
    for i, arm in enumerate(("BASE", "PROP")):
        bot = np.zeros(len(CASES))
        tot = np.array([med(sel(FINAL[arm], t, d), "wall_s") for t, d in CASES])
        for key, lab, col in keys:
            v = np.array([med(sel(FINAL[arm], t, d), key) for t, d in CASES])
            frac = 100 * v / tot
            ax.bar(x + (i - 0.5) * (w + 0.04), frac, w, bottom=bot, color=col,
                   label=lab if i == 0 else None,
                   edgecolor="white", linewidth=0.6)
            bot += frac
        for xi in x:
            ax.text(xi + (i - 0.5) * (w + 0.04), 101.5, arm, ha="center",
                    fontsize=7, color=C_ARM[arm])
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LB[c] for c in CASES], fontsize=8)
    ax.set_ylabel("share of wall time [%]")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              frameon=False)
    ax.set_title("Where the time goes: the fit is no longer the bottleneck, "
                 "acquisition is", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    return save(fig, "fig_final_budget", FIG_S, also_paper=True)


# --- S3: quality is preserved ---------------------------------------------- #
def fig_final_quality():
    """Per-seed quality, BASE vs PROP. Equal or better in every case."""
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(len(CASES))
    for i, arm in enumerate(("BASE", "PROP")):
        for xi, (t, d) in enumerate(CASES):
            q = QUALITY[t]
            runs = sel(FINAL[arm], t, d)
            v = [r[q] for r in runs]
            xs = xi + (i - 0.5) * 0.3
            ax.scatter([xs] * len(v), v, s=22, color=C_ARM[arm], alpha=0.65,
                       zorder=3, label=arm if xi == 0 else None)
            ax.hlines(np.median(v), xs - 0.09, xs + 0.09, color=C_ARM[arm], lw=2.2,
                      zorder=4)
            nf = len(sel(FINAL[arm], t, d, ok_only=False)) - len(runs)
            if nf:
                ax.text(xs, max(v) * 1.6, f"{nf} fail", ha="center", fontsize=7,
                        color=C_ARM[arm])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LB[c] for c in CASES], fontsize=8)
    ax.set_ylabel("quality metric (lower is better)")
    ax.set_title("Quality per seed: $D_\\mathrm{KL}$ (Gauss), $D_\\mathrm{KL}^z$ "
                 "(banana), mode-weight error (4-mode)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both", axis="y")
    return save(fig, "fig_final_quality", FIG_S, also_paper=True)


# --- S4: the 125-run restart study ----------------------------------------- #
def fig_restart_arms():
    arms = ["S0", "S1", "S2", "S3", "S4"]
    arm_lb = {"S0": "S0 uniform\n$10{+}2d$", "S1": "S1 uniform 8", "S2": "S2 informed\nonly (2)",
              "S3": "S3 local 8", "S4": "S4 screen 8"}
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(arms)))
    x = np.arange(len(CASES))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for i, a in enumerate(arms):
        sp, qm = [], []
        for t, d in CASES:
            ctrl = med(sel(RESTART, t, d, "S0"), "t_fit_s")
            sp.append(ctrl / max(med(sel(RESTART, t, d, a), "t_fit_s"), 1e-9))
            qm.append(med(sel(RESTART, t, d, a), QUALITY[t]))
        off = (i - 2) * 0.16
        axes[0].bar(x + off, sp, 0.15, color=cols[i], label=arm_lb[a])
        axes[1].bar(x + off, qm, 0.15, color=cols[i])
    axes[0].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel("fit speedup vs.\\ control S0")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("quality metric (lower is better)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([CASE_LB[c] for c in CASES], fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("125-run restart study: every strategy gives the same quality; "
                 "only the cost differs", fontsize=10)
    return save(fig, "fig_restart_arms", FIG_S, also_paper=True)


# --- S5: acquisition sampler head-to-head (v3) ----------------------------- #
def fig_sampler_scaling():
    """NUTS vs BlackJAX vs UltraNest at d=16/30. NOTE: v3 is NOT --exclusive and
    predates PR #4, so this is a *tractability* figure, not a timing benchmark."""
    ds = [16, 30]
    samplers = ["nuts", "blackjax", "ultranest"]
    x = np.arange(len(ds))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for i, s in enumerate(samplers):
        off = (i - 1) * 0.26
        for xi, d in enumerate(ds):
            al = [r for r in V3 if r["d"] == d and r["sampler"] == s]
            ok = [r for r in al if not r["failed"]]
            if not al:
                continue          # cell never run (no d=30 BlackJAX in manifest_v3)
            if not ok:            # every seed failed: mark, do not plot a height
                ax.text(xi + off, 1.2, f"0/{len(al)}\nconverged", ha="center",
                        fontsize=7, color="crimson", rotation=90, va="bottom")
                continue
            wall = np.median([r["wall_s"] for r in ok]) / 60
            fit = np.median([r["t_fit_s"] for r in ok]) / 60
            ax.bar(xi + off, wall - fit, 0.24, bottom=fit, color=C[s],
                   label=LB[s] if xi == 0 else None)
            ax.bar(xi + off, fit, 0.24, color=C[s], alpha=0.45, hatch="///",
                   edgecolor="white", linewidth=0.4,
                   label="of which: GP fit" if (i == 0 and xi == 0) else None)
            ax.text(xi + off, wall * 1.15, f"{len(ok)}/{len(al)} ok", ha="center",
                    fontsize=7, color="crimson" if len(ok) < len(al) else "k")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$d={d}$" for d in ds])
    ax.set_ylabel("wall time per converged run [min]")
    ax.set_ylim(bottom=1, top=ax.get_ylim()[1] * 6)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.annotate("$d{=}30$ NUTS is 85% GP fit (length scales railed at the old\n"
                "$10^2$ ceiling, PR #4 era) -- NOT a statement about acquisition",
                xy=(1 - 0.26, 202), xytext=(0.42, 900), fontsize=7, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))
    ax.set_title("Acquisition sampler at matched final MC (results/v3, 2026-08-12)\n"
                 "SHARED nodes + pre-PR#4 fit -- tractability only, NOT a timing benchmark",
                 fontsize=8.5)
    return save(fig, "fig_sampler_scaling", FIG_S, also_paper=True)


# --- S6: all-dimension corner plots ---------------------------------------- #
CORNERS = [
    ("final/PROP/gauss_d30_S2_seed1_corner.png", "ex_unimodal_d30_corner.png"),
    ("final/PROP/multimode_d5_S2_seed1_corner.png", "ex_multimode_d5_corner.png"),
    ("final/PROP/curved_d5_S2_seed1_corner.png", "ex_curved_d5_corner.png"),
    ("final/PROP/gauss_d16_S2_seed1_corner.png", "ex_unimodal_d16_corner.png"),
]


def copy_corners():
    """Corner plots come straight from the run harness, which already plots ALL
    dimensions (run_restart_study.corner_all_dims). Never crop to a subset."""
    done = []
    for src, dst in CORNERS:
        s = os.path.join(SETTLED, src)
        if not os.path.exists(s):
            print(f"  [warn] missing corner {src}")
            continue
        shutil.copy(s, os.path.join(FIG_S, dst))
        shutil.copy(s, os.path.join(PAPER_FIG, dst))
        done.append(dst)
    return done


# --- S7: POST-PR#4 acquisition-sampler head-to-head ------------------------ #
# --- Per-point cost: ONE definition, used everywhere ----------------------- #
# `per_point_ms` = MEDIAN over runs of (t_acquire / n_total), i.e. the median of
# the per-RUN costs. NOT median(t_acquire)/median(n_total).
#
# The two differ whenever n_total is spread out, because a ratio of medians pairs
# the time from one run with the point count of a DIFFERENT run. On the banana the
# UltraNest median time (301.4 s) comes from seed2, which acquired 75 points, while
# the median n (300) comes from seeds 1/3/4; dividing them gives 1005 ms/pt, a
# value no actual run has (the five real ones are 411, 1331, 2936, 3547, 4019).
# Every quoted per-point number in this project uses the median-of-per-run-ratios.
def per_point_ms(runs):
    v = sorted(1000.0 * r["t_acquire_s"] / r["n_total"] for r in runs)
    return float(np.median(v)) if v else np.nan


# Cases where n_total is so dispersed that ANY per-point point-estimate is
# unreliable, and the number must not be quoted in either direction.
PER_POINT_UNRELIABLE = {("curved", 5)}


def fig_h2h():
    """NUTS vs UltraNest acquisition on post-fix code, PROP defaults, exclusive.

    Every earlier NUTS-vs-nested comparison this project owns is pre-PR#4, when
    the `length_scale_prior` bug made every fit optimise over 10 decades. Both
    arms here are identical except the acquisition sampler, and the final MC is
    matched to it.

    Three panels, because loop time alone is confounded: a sampler can be slower
    end-to-end either because it costs more per acquired point or because it needs
    more points. Only the 4-mode case has matched n (300 for every run), so it is
    the one confound-free comparison in the set.
    """
    if not H2H:
        print("  [skip] fig_h2h: results/h2h/ is empty")
        return []
    x = np.arange(len(CASES))
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
    cols = {"nuts": "tab:red", "ultranest": "tab:blue"}
    for i, smp in enumerate(("nuts", "ultranest")):
        off = (i - 0.5) * 0.36
        for xi, (t, d) in enumerate(CASES):
            al = [r for r in H2H if r.get("target") == t and r.get("d") == d
                  and r.get("sampler") == smp]
            ok = [r for r in al if recovered(r)]
            if not al or not ok:
                continue
            lab = LB[smp] if xi == 0 else None
            # -- panel 0: end-to-end loop cost
            loop = float(np.median([r["t_loop_s"] for r in ok]))
            axes[0].bar(xi + off, loop, 0.34, color=cols[smp], label=lab)
            nto = sum(bool(r.get("timed_out")) for r in al)
            nfc = sum(r.get("converged") and not r["failed"] and not recovered(r)
                      for r in al)
            note = (f"{len(ok)}/{len(al)}" + (f"\n{nto} t/o" if nto else "")
                    + (f"\n{nfc} false conv" if nfc else ""))
            axes[0].text(xi + off, loop * 1.18, note, ha="center", fontsize=6.5,
                         color="crimson" if len(ok) < len(al) else "k")
            # -- panel 1: cost per acquired point (normalises out points-needed)
            pp = per_point_ms(ok)
            hatch = "///" if (t, d) in PER_POINT_UNRELIABLE else None
            axes[1].bar(xi + off, pp, 0.34, color=cols[smp], label=lab,
                        hatch=hatch, edgecolor="white", linewidth=0.6)
            lo = min(1000 * r["t_acquire_s"] / r["n_total"] for r in ok)
            hi = max(1000 * r["t_acquire_s"] / r["n_total"] for r in ok)
            axes[1].vlines(xi + off, lo, hi, color="k", lw=1.0, alpha=0.65)
            # -- panel 2: quality
            q = [r[QUALITY[t]] for r in ok if QUALITY[t] in r]
            axes[2].scatter([xi + off] * len(q), q, s=22, color=cols[smp],
                            alpha=0.65, zorder=3, label=lab)
            axes[2].hlines(np.median(q), xi + off - 0.1, xi + off + 0.1,
                           color=cols[smp], lw=2.2, zorder=4)
    # Median n_total per arm, shown in the panel-0 tick labels so the confound
    # (slower end-to-end can mean "more points", not "dearer per point") is
    # impossible to miss.
    nlab = []
    for t, d in CASES:
        ns = []
        for smp in ("nuts", "ultranest"):
            ok = [r for r in H2H if r.get("target") == t and r.get("d") == d
                  and r.get("sampler") == smp and recovered(r)]
            ns.append(int(np.median([r["n_total"] for r in ok])) if ok else 0)
        nlab.append(f"{CASE_LB[(t, d)]}\n$n$={ns[0]}/{ns[1]}")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median acquisition-loop time [s]")
    axes[0].set_title("end-to-end cost (recovered runs)\n"
                      "confounded: mixes cost/point with points needed", fontsize=8.5)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 5)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("acquisition ms per acquired point")
    axes[1].set_title("cost per point (median of per-run ratios)\n"
                      "bars = min-max over seeds; hatched = too dispersed to quote",
                      fontsize=8.5)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("quality metric (lower is better)")
    axes[2].set_title("quality", fontsize=9)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([CASE_LB[c] for c in CASES], fontsize=8)
        ax.grid(alpha=0.3, axis="y", which="both")
        ax.legend(fontsize=7.5)
    axes[0].set_xticklabels(nlab, fontsize=7.5)
    fig.suptitle("Acquisition sampler head-to-head, POST-PR#4, PROP defaults, "
                 "exclusive nodes, matched final MC. Only the 4-mode case has "
                 "matched $n$ (300 both arms).", fontsize=9.5)
    return save(fig, "fig_h2h", FIG_S, also_paper=True)


def build_settled():
    _load_settled()
    out = []
    out += fig_h2h()
    out += fig_final_cost()
    out += fig_final_budget()
    out += fig_final_quality()
    out += fig_restart_arms()
    out += fig_sampler_scaling()
    out += [os.path.join(FIG_S, c) for c in copy_corners()]
    return out


# =========================================================================== #
# LEGACY figures: results/raw (2026-08-10 Tier-B campaign)
# =========================================================================== #
def build_legacy():
    mm = read_csv(os.path.join(LEGACY_CSV, "multimodal_tierB_seed1.csv"))
    uni = read_csv(os.path.join(LEGACY_CSV, "unimodal_recovery.csv"))
    out = []

    def fig_timing():
        ds = sorted(set(int(r["d"]) for r in mm))
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for s in ("nuts", "ultranest"):
            m, lo, hi = [], [], []
            for d in ds:
                w = [r["wall_s"] for r in mm if int(r["d"]) == d and r["sampler"] == s
                     and isinstance(r["wall_s"], float)]
                m.append(np.median(w)); lo.append(min(w)); hi.append(max(w))
            ax.plot(ds, np.array(m) / 60, "o-", color=C[s], label=LB[s], lw=1.8)
            ax.fill_between(ds, np.array(lo) / 60, np.array(hi) / 60, color=C[s], alpha=0.15)
        ax.set_xlabel("dimension $d$"); ax.set_ylabel("wall time per run [min]")
        ax.set_yscale("log"); ax.set_xticks(ds)
        ax.set_title("STALE (2026-08-10, shared nodes, pre-PR#4)", fontsize=9)
        ax.legend(); ax.grid(alpha=0.3, which="both")
        return save(fig, "fig_timing_vs_dim", FIG_L)

    def fig_accuracy():
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True)
        for col, d in enumerate((8, 16)):
            for s in ("nuts", "ultranest"):
                pts = sorted([r for r in mm if int(r["d"]) == d and r["sampler"] == s],
                             key=lambda r: r["sep"])
                sep = [r["sep"] for r in pts]
                axes[0, col].plot(sep, [100 * r["w_relerr"] for r in pts], "o-",
                                  color=C[s], label=LB[s])
                axes[1, col].plot(sep, [r["wass_x0"] for r in pts], "o-", color=C[s])
            axes[0, col].set_title(f"$d={d}$")
            axes[1, col].set_xlabel(r"mode separation [$\sigma$]")
        axes[0, 0].set_ylabel("mode-weight error [%]")
        axes[1, 0].set_ylabel(r"$W_1$ on bimodal axis")
        axes[0, 0].legend(fontsize=8)
        for ax in axes.ravel():
            ax.grid(alpha=0.3)
        fig.suptitle("Two-mode recovery vs. separation  [STALE: 2026-08-10 campaign]")
        return save(fig, "fig_accuracy_vs_sep", FIG_L)

    def fig_unimodal():
        ds = sorted(set(int(r["d"]) for r in uni))
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        m = [np.median([r["kl"] for r in uni if int(r["d"]) == d]) for d in ds]
        lo = [min(r["kl"] for r in uni if int(r["d"]) == d) for d in ds]
        hi = [max(r["kl"] for r in uni if int(r["d"]) == d) for d in ds]
        ax.plot(ds, m, "o-", color="tab:green", lw=1.8, label="median over 3 seeds")
        ax.fill_between(ds, lo, hi, color="tab:green", alpha=0.2)
        ax.axhline(0.05, ls="--", color="grey", lw=1, label="recovery threshold (0.05)")
        ax.set_xlabel("dimension $d$")
        ax.set_ylabel(r"$D_{\mathrm{KL}}(\mathrm{GP}\,\|\,\mathrm{truth})$")
        ax.set_title("STALE (2026-08-10 campaign)", fontsize=9)
        ax.set_xticks(ds); ax.legend(); ax.grid(alpha=0.3)
        return save(fig, "fig_unimodal_scaling", FIG_L)

    out += fig_timing() + fig_accuracy() + fig_unimodal()
    return out


# =========================================================================== #
if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    written = []
    if what in ("settled", "all"):
        print("=== SETTLED (results/paper_runs: final, restart, h2h, v3) ===")
        written += build_settled()
    if what in ("legacy", "all"):
        print("=== LEGACY (results/other_runs, 2026-08-10 -- do NOT quote timings) ===")
        written += build_legacy()
    for p in written:
        print("  ", os.path.relpath(p, REPO))
    print(f"\n{len(written)} files written")
    print(f"paper-facing copies in {os.path.relpath(PAPER_FIG, REPO)}")
