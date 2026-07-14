"""
Disambiguate the low HMC ESS: is it (A) an ESS-estimator artifact from
multimodality (chains trapped in different modes inflate between-chain variance,
which rank-normalized cross-chain ESS reads as terrible mixing), or (B) genuine
poor mixing of the fixed-config HMC (step size too small -> high acceptance but
high autocorrelation), or both?

Test A: on the bimodal trace, compare cross-chain ESS vs summed within-chain ESS.
Test B: on a UNIMODAL Gaussian (where cross-chain ESS is valid), sweep HMC step
        size / trajectory length and see whether evals/ESS approaches UltraNest.
"""
import warnings
import numpy as np
import arviz as az

warnings.filterwarnings("ignore")

from gpry.gpr import GaussianProcessRegressor as GPR
from gpry.gp_acquisition import NORA
from gpry.mc_interfaces import hmc_sample_gp_mean

RNG = np.random.default_rng(0)


def cross_chain_ess(trace):
    d = trace.shape[2]
    return min(float(az.ess(np.ascontiguousarray(trace[:, :, k].T))) for k in range(d))


def within_chain_ess(trace):
    # sum over chains of per-chain ESS, min over dims
    n_chains, d = trace.shape[1], trace.shape[2]
    per_dim = []
    for k in range(d):
        s = 0.0
        for c in range(n_chains):
            s += float(az.ess(np.ascontiguousarray(trace[:, c, k])))
        per_dim.append(s)
    return min(per_dim)


# ---------------- Test A: multimodal ESS artifact ---------------- #
def test_A():
    m1, m2, s = np.array([-2., -2.]), np.array([2., 2.]), 0.6
    bounds = np.array([[-4., 4.]] * 2)

    def logp(X):
        X = np.atleast_2d(X)
        c = -np.log(2 * np.pi * s ** 2)
        l1 = c - 0.5 * np.sum((X - m1) ** 2, axis=1) / s ** 2
        l2 = c - 0.5 * np.sum((X - m2) ** 2, axis=1) / s ** 2
        return np.logaddexp(l1 + np.log(0.5), l2 + np.log(0.5))

    Xtr = np.concatenate([m1 + 0.6 * RNG.standard_normal((25, 2)),
                          m2 + 0.6 * RNG.standard_normal((25, 2)),
                          RNG.uniform(-4, 4, size=(15, 2))])
    gpr = GPR(kernel="RBF", bounds=bounds, account_for_inf=None, n_restarts_optimizer=6)
    gpr.append_to_data(Xtr, logp(Xtr), fit_gpr=True, fit_classifier=False)

    res = hmc_sample_gp_mean(gpr, bounds[:, 0], bounds[:, 1],
                             rng=np.random.default_rng(4),
                             max_chains=256, n_warmup=40, n_samples=60,
                             thin=1, n_leapfrog=15)
    cc = cross_chain_ess(res["trace"])
    wc = within_chain_ess(res["trace"])
    print("=== TEST A: bimodal target, ESS estimator ===")
    print(f"  n_chains={res['trace'].shape[1]}  draws/chain={res['trace'].shape[0]}")
    print(f"  cross-chain ESS (arviz default) = {cc:8.1f}   <- assumes all chains "
          f"sample the same unimodal law")
    print(f"  summed within-chain ESS         = {wc:8.1f}   <- valid when chains "
          f"are trapped in separate modes (our seeding)")
    print(f"  --> multimodality deflates cross-chain ESS by {wc/max(1e-9,cc):.1f}x "
          f"(artifact, not real inefficiency)\n")


# ---------------- Test B: unimodal efficiency sweep ---------------- #
def test_B(d=2):
    bounds = np.array([[-5., 5.]] * d)
    cov = np.diag(np.linspace(1.0, 0.4, d) ** 2)
    prec = np.linalg.inv(cov)
    const = -0.5 * (d * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1])

    def logp(X):
        X = np.atleast_2d(X)
        return const - 0.5 * np.einsum("ni,ij,nj->n", X, prec, X, optimize=True)

    Xtr = RNG.multivariate_normal(np.zeros(d), cov, size=12 * d)
    Xtr = np.clip(Xtr, bounds[:, 0], bounds[:, 1])
    gpr = GPR(kernel="RBF", bounds=bounds, account_for_inf=None, n_restarts_optimizer=2 * d)
    gpr.append_to_data(Xtr, logp(Xtr), fit_gpr=True, fit_classifier=False)

    # UltraNest reference
    acq = NORA(bounds=bounds, preprocessing_X=gpr.preprocessing_X,
               acq_func="LogExp", sampler="ultranest", verbose=0)
    gpr.n_eval = 0
    Xn, yn, syn, wn = acq.do_MC_sample(gpr, bounds=bounds,
                                       rng=np.random.default_rng(3), sampler="ultranest")
    ns_ess = (wn.sum() ** 2) / np.sum(wn ** 2)
    ns_nev = int(gpr.n_eval)
    print(f"=== TEST B: UNIMODAL Gaussian, d={d} (cross-chain ESS is valid) ===")
    print(f"  {'config':28s} {'evals':>7s} {'ESS':>6s} {'evals/ESS':>10s} {'accept':>7s}")
    print(f"  {'ultranest':28s} {ns_nev:>7d} {ns_ess:>6.0f} "
          f"{ns_nev/ns_ess:>10.1f} {'-':>7s}")

    # HMC configs: vary trajectory length and fixed step size (bypass weak adapt)
    for L, eps in [(8, None), (20, None), (20, 0.6), (40, 0.6)]:
        gpr.n_eval = 0
        res = hmc_sample_gp_mean(gpr, bounds[:, 0], bounds[:, 1],
                                 rng=np.random.default_rng(5),
                                 max_chains=8, n_warmup=30, n_samples=60,
                                 thin=1, n_leapfrog=L, step_size=eps)
        ess = cross_chain_ess(res["trace"])
        nev = int(gpr.n_eval)
        tag = f"hmc L={L} eps={'adapt' if eps is None else eps}"
        print(f"  {tag:28s} {nev:>7d} {ess:>6.0f} {nev/max(1e-9,ess):>10.1f} "
              f"{res['accept_rate']:>7.2f}")


if __name__ == "__main__":
    test_A()
    test_B(d=2)
    test_B(d=5)
