"""
Why did HMC use ~21x more GP-evals than UltraNest in the 2D bimodal test?

The raw eval COUNT is not a fair efficiency metric on its own, because the two
samplers were stopped by different rules:
  - UltraNest ran to its OWN convergence (logZ accuracy / ESS target);
  - HMC ran a FIXED, hand-chosen budget (n_chains x n_iters x n_leapfrog), and I
    picked that budget profligately (65 chains, 100 iters incl. 40 warmup, 15
    leapfrog steps).

The regime-independent metric is GP-evaluations per EFFECTIVE sample (ESS).
This script reports it for: UltraNest, HMC "as-run", and a lean HMC config, so
we can separate (a) config waste from (b) the fundamental low-d disadvantage.
"""
import time
import warnings

import numpy as np
import arviz as az

warnings.filterwarnings("ignore")

from gpry.gpr import GaussianProcessRegressor as GPR
from gpry.gp_acquisition import NORA
from gpry.mc_interfaces import hmc_sample_gp_mean

RNG = np.random.default_rng(0)
M1, M2, S = np.array([-2.0, -2.0]), np.array([2.0, 2.0]), 0.6
BOUNDS = np.array([[-4.0, 4.0]] * 2)


def bimodal_logp(X):
    X = np.atleast_2d(X)
    c = -0.5 * 2 * np.log(2 * np.pi * S ** 2)
    l1 = c - 0.5 * np.sum((X - M1) ** 2, axis=1) / S ** 2
    l2 = c - 0.5 * np.sum((X - M2) ** 2, axis=1) / S ** 2
    return np.logaddexp(l1 + np.log(0.5), l2 + np.log(0.5))


def build_gpr():
    Xtr = np.concatenate([
        M1 + 0.6 * RNG.standard_normal((25, 2)),
        M2 + 0.6 * RNG.standard_normal((25, 2)),
        RNG.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(15, 2)),
    ])
    Xtr = np.clip(Xtr, BOUNDS[:, 0], BOUNDS[:, 1])
    gpr = GPR(kernel="RBF", bounds=BOUNDS, account_for_inf=None, n_restarts_optimizer=6)
    gpr.append_to_data(Xtr, bimodal_logp(Xtr), fit_gpr=True, fit_classifier=False)
    return gpr


def ess_weighted(w):
    """Kish effective sample size of an importance-weighted set."""
    if w is None or len(w) == 0:
        return np.nan
    w = np.asarray(w, dtype=float)
    return (w.sum() ** 2) / np.sum(w ** 2)


def ess_hmc(trace):
    """Min-over-dimensions ESS from a chain-structured HMC trace.

    trace: (n_draws, n_chains, d)  ->  arviz wants (chain, draw) per variable.
    """
    if trace.shape[0] < 4:
        return np.nan
    d = trace.shape[2]
    per_dim = []
    for k in range(d):
        arr = np.ascontiguousarray(trace[:, :, k].T)  # (chain, draw)
        per_dim.append(float(az.ess(arr)))
    return min(per_dim)


def run_ultranest(gpr):
    acq = NORA(bounds=BOUNDS, preprocessing_X=gpr.preprocessing_X,
               acq_func="LogExp", sampler="ultranest", verbose=0)
    gpr.n_eval = 0
    t0 = time.time()
    X, y, sy, w = acq.do_MC_sample(gpr, bounds=BOUNDS,
                                   rng=np.random.default_rng(3), sampler="ultranest")
    dt = time.time() - t0
    ess = ess_weighted(w)
    nev = int(gpr.n_eval)
    return dict(name="ultranest", nev=nev, dt=dt, pool=len(X), ess=ess,
                eff=nev / max(1e-9, ess))


def run_hmc(gpr, label, **kw):
    gpr.n_eval = 0
    t0 = time.time()
    res = hmc_sample_gp_mean(gpr, BOUNDS[:, 0], BOUNDS[:, 1],
                             rng=np.random.default_rng(4), **kw)
    dt = time.time() - t0
    nev = int(gpr.n_eval)
    ess = ess_hmc(res["trace"])
    # exact accounting check
    nchains = res["n_chains"]
    formula = nchains * (kw["n_warmup"] + kw["n_samples"]) * kw["n_leapfrog"] + nchains
    return dict(name=label, nev=nev, dt=dt, pool=len(res["X"]), ess=ess,
                eff=nev / max(1e-9, ess), acc=res["accept_rate"],
                nchains=nchains, formula=formula,
                warmup_frac=kw["n_warmup"] / (kw["n_warmup"] + kw["n_samples"]))


def main():
    gpr = build_gpr()
    print(f"GPR: {gpr.kernel_}   n_train={len(gpr.X_train_)}")

    rows = []
    rows.append(run_ultranest(gpr))
    # HMC "as-run" in Part C (profligate): all training pts as chains, 15 leapfrog
    rows.append(run_hmc(gpr, "hmc as-run", max_chains=256, n_warmup=40,
                        n_samples=60, thin=1, n_leapfrog=15))
    # HMC lean: few chains (unimodal-per-mode is enough), short traj, little warmup
    rows.append(run_hmc(gpr, "hmc lean", seeds=None, max_chains=8, n_warmup=15,
                        n_samples=40, thin=1, n_leapfrog=8))

    print("\n" + "-" * 78)
    print(f"{'sampler':12s} {'GP-evals':>9s} {'wall(s)':>8s} {'pool':>6s} "
          f"{'ESS':>7s} {'evals/ESS':>10s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['name']:12s} {r['nev']:>9d} {r['dt']:>8.2f} {r['pool']:>6d} "
              f"{r['ess']:>7.0f} {r['eff']:>10.1f}")
    print("-" * 78)

    for r in rows[1:]:
        print(f"\n[{r['name']}] eval accounting: "
              f"n_chains({r['nchains']}) x (warmup+sample) x n_leapfrog + init "
              f"= {r['formula']}  (measured {r['nev']})   "
              f"warmup_frac={r['warmup_frac']:.0%}  accept={r['acc']:.2f}")

    ns, asrun, lean = rows
    print("\n=== DECOMPOSITION of the eval gap (per-ESS is the fair metric) ===")
    print(f"  UltraNest : {ns['eff']:8.1f} GP-evals / effective sample")
    print(f"  HMC as-run: {asrun['eff']:8.1f} GP-evals / effective sample "
          f"({asrun['eff']/ns['eff']:.1f}x NS)")
    print(f"  HMC lean  : {lean['eff']:8.1f} GP-evals / effective sample "
          f"({lean['eff']/ns['eff']:.1f}x NS)")
    print(f"\n  Config waste recovered by leaning out: "
          f"{asrun['eff']/lean['eff']:.1f}x")
    print(f"  Residual (lean HMC vs NS, per ESS): {lean['eff']/ns['eff']:.1f}x  "
          f"<- this is the fundamental low-d part (~ n_leapfrog overhead)")


if __name__ == "__main__":
    main()
