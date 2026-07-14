"""
Decisive test for Lever 1: does training-seeded HMC beat NORA's NS on
GP-evaluation COUNT in the regime the handoff claims -- a stiff, UNIMODAL,
high-d target -- where NS eval-count is supposed to grow steeply with d?

For each dimension we build an anisotropic (stiff) correlated Gaussian, fit a
GP to samples from it, then run BOTH samplers on the same GP and report:
  - GP-evaluation count (the metric Lever 1 targets),
  - wall-time,
  - posterior recovery quality (weighted mean / std error vs truth).

HMC uses a FIXED, sensible config (a modest, d-independent number of chains,
since the target is unimodal), so its eval-count is ~d-independent by design;
NS eval-count grows with d. The question is where (if anywhere) they cross.
"""
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from gpry.gpr import GaussianProcessRegressor as GPR
from gpry.gp_acquisition import NORA
from gpry.mc_interfaces import hmc_sample_gp_mean

RNG = np.random.default_rng(7)


def stiff_gaussian(d, cond=100.0):
    """Anisotropic zero-mean Gaussian; condition number ~ cond."""
    # log-spaced eigenvalues (stddevs) from 1 down to 1/sqrt(cond)
    stds = np.geomspace(1.0, 1.0 / np.sqrt(cond), d)
    Q, _ = np.linalg.qr(RNG.standard_normal((d, d)))  # random rotation
    cov = Q @ np.diag(stds ** 2) @ Q.T
    cov = 0.5 * (cov + cov.T)
    prec = np.linalg.inv(cov)
    logdet = np.linalg.slogdet(cov)[1]
    const = -0.5 * (d * np.log(2 * np.pi) + logdet)

    def logp(X):
        X = np.atleast_2d(X)
        q = np.einsum("ni,ij,nj->n", X, prec, X, optimize=True)
        return const - 0.5 * q

    return logp, cov


def weighted_stats(X, w):
    if X is None or len(X) == 0:
        return np.nan, np.nan
    if w is None:
        w = np.ones(len(X))
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    mean = np.average(X, axis=0, weights=w)
    var = np.average((X - mean) ** 2, axis=0, weights=w)
    return mean, np.sqrt(var)


def run_dim(d, cond=100.0):
    logp, cov = stiff_gaussian(d, cond)
    true_std = np.sqrt(np.diag(cov))
    span = 5.0  # prior box in units where the largest std is 1
    bounds = np.array([[-span, span]] * d)

    n_train = 12 * d
    Xtr = RNG.multivariate_normal(np.zeros(d), cov, size=n_train)
    Xtr = np.clip(Xtr, bounds[:, 0], bounds[:, 1])
    ytr = logp(Xtr)
    gpr = GPR(kernel="RBF", bounds=bounds, account_for_inf=None,
              n_restarts_optimizer=2 * d)
    gpr.append_to_data(Xtr, ytr, fit_gpr=True, fit_classifier=False)

    acq = NORA(bounds=bounds, preprocessing_X=gpr.preprocessing_X,
               acq_func="LogExp", sampler="ultranest", verbose=0,
               max_ncalls=400000)

    out = {}
    # --- UltraNest ---
    gpr.n_eval = 0
    t0 = time.time()
    Xn, yn, syn, wn = acq.do_MC_sample(
        gpr, bounds=bounds, rng=np.random.default_rng(11), sampler="ultranest")
    out["ns"] = dict(nev=int(gpr.n_eval), dt=time.time() - t0,
                     n=(0 if Xn is None else len(Xn)),
                     stats=weighted_stats(Xn, wn))
    # --- HMC (fixed, d-independent chain budget: unimodal) ---
    gpr.n_eval = 0
    seeds = np.asarray(gpr.X_train_)
    t0 = time.time()
    res = hmc_sample_gp_mean(
        gpr, bounds[:, 0], bounds[:, 1], seeds=seeds,
        rng=np.random.default_rng(12),
        max_chains=16, n_warmup=30, n_samples=40, thin=2, n_leapfrog=10)
    # HMC samples live in transformed space == original (identity preprocessing)
    out["hmc"] = dict(nev=int(gpr.n_eval), dt=time.time() - t0,
                      n=len(res["X"]), acc=res["accept_rate"],
                      stats=weighted_stats(res["X"], None))

    def std_err(stats):
        _, s = stats
        if np.any(np.isnan(s)):
            return np.nan
        return float(np.median(np.abs(s - true_std) / true_std))

    print(f"\n===== d = {d}  (cond {cond:.0f}, n_train {n_train}) =====")
    print(f"  {'sampler':10s} {'GP-evals':>10s} {'wall(s)':>8s} {'pool':>6s} "
          f"{'med|Δstd|/std':>14s}")
    for k, label in [("ns", "ultranest"), ("hmc", "hmc")]:
        r = out[k]
        print(f"  {label:10s} {r['nev']:>10d} {r['dt']:>8.2f} {r['n']:>6d} "
              f"{std_err(r['stats']):>14.3f}")
    ratio = out["ns"]["nev"] / max(1, out["hmc"]["nev"])
    print(f"  --> GP-eval count  NS / HMC = {ratio:.2f}x   "
          f"({'HMC cheaper' if ratio > 1 else 'NS cheaper'})")
    return out


if __name__ == "__main__":
    for d in [4, 8, 16]:
        run_dim(d)
