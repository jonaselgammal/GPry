"""
CHEAP, re-runnable step: load checkpointed GPs from a run dir and evaluate
posterior recovery with the RELIABLE method (well-tuned NUTS-many-chains + a
sampler-free Laplace covariance cross-check) -- NOT UltraNest, which fails to
sample high-d GPs. Writes metrics + corner + all-d marginals per checkpoint.

Usage:  python run_eval.py <run_dir>
        (run_dir = .../d<d>_seed<seed> produced by run_acquisition.py)
"""
import os
import sys
import glob
import json

import numpy as np


def marginals_plot(bounds, marg, samples, path, d):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 6 if d > 6 else d
    nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.4 * nrow), squeeze=False)
    X, w = samples
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i >= d:
            ax.axis("off"); continue
        s = marg[i]
        xs = np.linspace(-4 * s, 4 * s, 200)
        ax.plot(xs, np.exp(-0.5 * (xs / s) ** 2) / (s * np.sqrt(2 * np.pi)), "k-", lw=1.3)
        if X is not None and len(X) > 20:
            ax.hist(X[:, i], bins=40, weights=w, density=True, histtype="step",
                    color="tab:orange")
        ax.axvline(0, color="k", ls=":", alpha=0.4); ax.set_yticks([])
        ax.set_title(f"x{i} (sig={s:.2f})", fontsize=7)
    fig.suptitle("marginals: truth (black) vs NUTS-GP posterior (orange)")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def corner_plot(mean, cov, samples, path, d):
    try:
        from getdist import MCSamples, plots
    except Exception:
        return
    sub = list(range(min(d, 10)))          # cap panels for readability
    nm = [f"x{i}" for i in sub]
    ref = np.random.default_rng(1).multivariate_normal(mean, cov, 20000)[:, sub]
    mcs = [MCSamples(samples=ref, names=nm, label="truth")]
    X, w = samples
    if X is not None and len(X) > 50:
        mcs.append(MCSamples(samples=np.asarray(X)[:, sub], weights=np.asarray(w),
                             names=nm, label="GP (NUTS)"))
    g = plots.get_subplot_plotter(width_inch=min(2 + 1.1 * len(sub), 12))
    g.triangle_plot(mcs, filled=[True, False][:len(mcs)],
                    legend_labels=[m.label for m in mcs])
    g.export(path)


def main():
    run_dir = sys.argv[1]
    import common as C
    t = np.load(os.path.join(run_dir, "truth.npz"))
    mean, cov, marg, bounds = t["mean"], t["cov"], t["marg"], t["bounds"]
    d = len(mean)
    results = {}
    for pkl in sorted(glob.glob(os.path.join(run_dir, "gp_n*.pkl"))):
        ck = os.path.basename(pkl)[4:-4]  # n<ckpt>
        gpr = C.load_gp(pkl)
        rec, samples = C.evaluate_recovery(
            gpr, bounds, mean, cov, marg,
            nuts_kwargs=dict(n_chains=64, n_warmup=300, n_samples=300))
        results[ck] = {k: v for k, v in rec.items() if k != "mean"}
        marginals_plot(bounds, marg, samples,
                       os.path.join(run_dir, f"marginals_{ck}.png"), d)
        corner_plot(mean, cov, samples, os.path.join(run_dir, f"corner_{ck}.png"), d)
        recovered = (rec["kl_nuts"] < 0.5 and rec["max_mean_in_sigma"] < 0.25
                     and rec["std_relerr_nuts"] < 0.25)
        print(f"[{run_dir} :: {ck}] KL={rec['kl_nuts']} max|mean|={rec['max_mean_in_sigma']}sig "
              f"stderr(nuts)={rec['std_relerr_nuts']} stderr(laplace)={rec['std_relerr_laplace']} "
              f"div={rec['n_divergent']} -> RECOVERED={recovered}", flush=True)
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
