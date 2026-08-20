"""
CHEAP, re-runnable step (Tier-B multimodal): load checkpointed GPs and measure
MULTIMODAL recovery -- did acquisition find BOTH basins, with the right weights
and no spurious mass -- read out with well-tuned NUTS-many-chains and compared
against the EXACT two-mode mixture (closed-form samples; no evaluator sampler).

Usage:  python run_eval_2mode.py <run_dir>
        (run_dir = .../mm_d<d>_sep<..>_wr<..>_<sampler>_seed<..>)
"""
import os
import sys
import glob
import json

import numpy as np


def x0_marginal_plot(target, samples, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means, w = target["means"], target["weights"]
    sep = target["sep"]
    lo, hi = target["bounds"][0]
    xs = np.linspace(lo, hi, 400)
    dens = (w[0] * np.exp(-0.5 * (xs - means[0, 0]) ** 2)
            + w[1] * np.exp(-0.5 * (xs - means[1, 0]) ** 2)) / np.sqrt(2 * np.pi)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(xs, dens, "k-", lw=1.5, label="truth")
    X, ww = samples
    if X is not None and len(X) > 20:
        ax.hist(X[:, 0], bins=60, weights=ww, density=True, histtype="step",
                color="tab:red", lw=1.3, label="GP (NUTS)")
    for c in means[:, 0]:
        ax.axvline(c, color="k", ls=":", alpha=0.4)
    ax.set_title(f"x0 marginal (sep={sep:g} sigma)  truth vs GP-NUTS")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def corner_plot(target, samples, path):
    try:
        from getdist import MCSamples, plots
    except Exception:
        return
    import common as C
    d = target["d"]
    # ALL dimensions, always (see run_eval.py).
    sub = list(range(d))
    nm = [f"x{i}" for i in sub]
    ref = C.two_mode_reference_samples(target, n=40000, seed=1)[:, sub]
    mcs = [MCSamples(samples=ref, names=nm, label="truth")]
    X, w = samples
    if X is not None and len(X) > 50:
        mcs.append(MCSamples(samples=np.asarray(X)[:, sub], weights=np.asarray(w),
                             names=nm, label="GP (NUTS)"))
    g = plots.get_subplot_plotter(width_inch=min(2 + 0.7 * len(sub), 26))
    g.triangle_plot(mcs, filled=[True, False][:len(mcs)],
                    legend_labels=[m.label for m in mcs])
    g.export(path)


def main():
    run_dir = sys.argv[1]
    import common as C
    t = np.load(os.path.join(run_dir, "truth.npz"))
    target = dict(means=t["means"], cov=t["cov"], marg=t["marg"],
                  weights=t["weights"], sep=float(t["sep"]),
                  weight_ratio=float(t["weight_ratio"]), bounds=t["bounds"],
                  d=len(t["marg"]))
    results = {}
    eval_path = os.path.join(run_dir, "eval.json")
    # Include gp_final.pkl (written even when the run converged before the first
    # checkpoint), so early-converging runs are still scoreable.
    pkls = sorted(glob.glob(os.path.join(run_dir, "gp_n*.pkl")))
    pkls += sorted(glob.glob(os.path.join(run_dir, "gp_final.pkl")))
    for pkl in pkls:
        ck = os.path.basename(pkl)[3:-4].lstrip("_n") or "final"
        surrogate = C.load_gp(pkl)
        rec, samples = C.evaluate_recovery_2mode(
            surrogate, target,
            nuts_kwargs=dict(n_chains=64, n_warmup=300, n_samples=300))
        results[ck] = rec
        # Verdict: both basins found, weights within 25%, negligible spurious
        # mass in EXCESS of the chi2_d tail expected for a perfect sample.
        ok = (rec["n_modes_found"] == 2 and rec["w_relerr"] < 0.25
              and rec["spurious_excess"] < 0.05)
        rec["recovered"] = bool(ok)
        print(f"[{run_dir} :: {ck}] modes={rec['n_modes_found']}/2 "
              f"w_gp={rec['w_gp']} w_true={rec['w_true']} w_relerr={rec['w_relerr']} "
              f"spurious_excess={rec['spurious_excess']} wass_x0={rec['wass_x0']} "
              f"div={rec['n_divergent']} -> RECOVERED={ok}", flush=True)
        # Persist metrics BEFORE plotting: a matplotlib/getdist failure (e.g. the
        # usetex tex.cache race seen on the cluster) must not lose the results.
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
        for plot_fn, name in ((x0_marginal_plot, f"x0_{ck}.png"),
                              (corner_plot, f"corner_{ck}.png")):
            try:
                plot_fn(target, samples, os.path.join(run_dir, name))
            except Exception as exc:  # plots are cosmetic; metrics already saved
                print(f"  [warn] plot {name} failed: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
