"""
Stage 2: WHY is the GP hyperparameter refit more expensive on NUTS-acquired
training sets? (Phase 1 saw ~8x more LML evals per FULL fit at d=16.)

The refit is sampler-INDEPENDENT code, so the cause must be the training data.
This script loads the saved surrogates and runs the IDENTICAL hyperparameter fit
on each, with instrumentation that separates the two possible cost sources:

  (a) rejection spin -- the restart loop redraws theta while LML is non-finite,
      and each rejected draw is an LML eval counted in `evals_fit`;
  (b) genuinely slower L-BFGS -- more iterations per restart to converge.

and reports the training-set geometry that would explain (b): nearest-neighbour
spacing and the conditioning of K (clustered points -> ill-conditioned K ->
noisy/flat LML surface -> slow L-BFGS).

Usage: python diag_refit_analyse.py <diag_dir> [n_restarts] [n_repeats]
"""
import os
import sys
import glob

import numpy as np

# `common.py` lives in ../run (this script moved into analyse/ on 2026-08-20).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "run"))
import common as C


def geometry(X, y):
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    return dict(
        n=len(X),
        nn_min=float(nn.min()), nn_med=float(np.median(nn)),
        pairs_lt_1e2=int((D < 1e-2).sum() // 2),
        y_range=float(np.max(y) - np.min(y)),
    )


def instrumented_fit(sur, n_restarts, seed):
    """
    Run the real fit_gpr_hyperparameters, counting LML calls and splitting them
    into rejection-loop draws (gradient-free) vs L-BFGS calls (with gradient).
    """
    gpr = sur.gpr
    counts = {"nograd": 0, "grad": 0}
    orig = gpr.log_marginal_likelihood

    def wrapped(theta=None, eval_gradient=False, clone_kernel=True):
        counts["grad" if eval_gradient else "nograd"] += 1
        return orig(theta, eval_gradient=eval_gradient, clone_kernel=clone_kernel)

    gpr.log_marginal_likelihood = wrapped
    gpr.random_state = np.random.default_rng(seed)
    import time
    t0 = time.time()
    try:
        gpr._fit_hyperparameters(n_restarts=n_restarts, start_from_current=True,
                                 start_from_cov=True)
    finally:
        gpr.log_marginal_likelihood = orig
    dt = time.time() - t0
    return dt, counts


def main():
    ddir = sys.argv[1]
    n_restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n_rep = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    rows = []
    for pkl in sorted(glob.glob(os.path.join(ddir, "*_surrogate.pkl"))):
        tag = os.path.basename(pkl).replace("_surrogate.pkl", "")
        npz = os.path.join(ddir, f"{tag}_train.npz")
        z = np.load(npz)
        X, y = z["X_train_"], z["y_train_"]
        d = X.shape[1]
        nr = n_restarts or (10 + 2 * d)
        g = geometry(X, y)
        # conditioning of K at the fitted hyperparameters
        sur0 = C.load_gp(pkl)
        K = sur0.gpr.kernel_(X)
        K[np.diag_indices_from(K)] += getattr(sur0.gpr, "alpha", 1e-4)
        g["cond_K"] = float(np.linalg.cond(K))
        g["theta_ls_med"] = float(np.median(np.exp(sur0.gpr.kernel_.theta[1:])))

        dts, ngs, gs = [], [], []
        for rep in range(n_rep):
            sur = C.load_gp(pkl)  # fresh copy each repeat
            dt, cnt = instrumented_fit(sur, nr, seed=100 + rep)
            dts.append(dt); ngs.append(cnt["nograd"]); gs.append(cnt["grad"])
        g.update(tag=tag, n_restarts=nr,
                 fit_s=float(np.median(dts)),
                 rejection_evals=float(np.median(ngs)),
                 lbfgs_evals=float(np.median(gs)),
                 lbfgs_per_restart=float(np.median(gs)) / nr)
        rows.append(g)

    keys = ["tag", "n", "nn_min", "nn_med", "pairs_lt_1e2", "y_range", "cond_K",
            "theta_ls_med", "fit_s", "rejection_evals", "lbfgs_evals",
            "lbfgs_per_restart"]
    hdr = " ".join(f"{k:>17}" if k == "tag" else f"{k:>13}" for k in keys)
    print(hdr); print("-" * len(hdr))
    for r in rows:
        cells = []
        for k in keys:
            v = r[k]
            if isinstance(v, str):
                cells.append(f"{v:>17}")
            elif isinstance(v, float):
                cells.append(f"{v:>13.4g}")
            else:
                cells.append(f"{v:>13}")
        print(" ".join(cells))


if __name__ == "__main__":
    main()
