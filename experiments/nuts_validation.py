"""
Validate the BlackJAX NUTS backend and compare its GP-eval efficiency against
numpy-HMC and UltraNest.

  Part 1: the JAX log-density matches an independent numpy reference built from
          the TRUE fitted kernel (gpr.kernel_) -> confirms the JAX GP-mean and
          the logit-bijector Jacobian are correct.
  Part 2: UNIMODAL Gaussian (valid cross-chain ESS): GP-evals per effective
          sample for UltraNest vs numpy-HMC vs BlackJAX NUTS.
  Part 3: bimodal -> NUTS seeded from training points covers both modes.
"""
import warnings
import numpy as np
import arviz as az

warnings.filterwarnings("ignore")

from gpry.gpr import GaussianProcessRegressor as GPR
from gpry.gp_acquisition import NORA
from gpry import mc_interfaces as mci

RNG = np.random.default_rng(0)


def fit(kernel, bounds, X, y, nr=6):
    g = GPR(kernel=kernel, bounds=bounds, account_for_inf=None, n_restarts_optimizer=nr)
    g.append_to_data(X, y, fit_gpr=True, fit_classifier=False)
    return g


def cross_chain_ess(trace):
    d = trace.shape[2]
    return min(float(az.ess(np.ascontiguousarray(trace[:, :, k].T))) for k in range(d))


# ------------------------------------------------------------------ Part 1 --- #
def part1():
    print("=== PART 1: JAX log-density vs numpy reference (true kernel_) ===")
    import jax
    jax.config.update("jax_enable_x64", True)
    d = 3
    bounds = np.array([[-3.0, 3.0]] * d)
    X = RNG.uniform(-3, 3, size=(40, d))
    y = -0.5 * np.sum(X ** 2, axis=1) + np.sin(X[:, 0]) * X[:, 1]
    for kernel in ["RBF", "Matern"]:
        gpr = fit(kernel, bounds, X, y)
        lo, hi = bounds[:, 0], bounds[:, 1]
        logdensity_fn, x_of_u, u_of_x = mci._build_jax_logdensity(gpr, lo, hi, beta=1.0)
        std_y = float(np.ravel(gpr.preprocessing_y.inverse_transform_scale(np.ones(1)))[0])
        alpha = np.ravel(np.asarray(gpr.alpha_))
        errs = []
        for _ in range(20):
            u = RNG.uniform(-3, 3, size=d)
            xn = lo + (hi - lo) / (1 + np.exp(-u))                 # sigmoid
            mean_norm = float((gpr.kernel_(xn[None], gpr.X_train_) @ alpha).ravel()[0])
            s = 1 / (1 + np.exp(-u))
            jac = np.sum(np.log(hi - lo) + np.log(s) + np.log1p(-s))
            ref = std_y * mean_norm + jac
            got = float(logdensity_fn(np.asarray(u)))
            errs.append(abs(ref - got))
        print(f"  [{kernel:6s}] max|jax_logdensity - numpy_ref| = {max(errs):.2e}  "
              f"--> {'PASS' if max(errs) < 1e-6 else 'FAIL'}")
        assert max(errs) < 1e-6


# ------------------------------------------------------------------ Part 2 --- #
def part2(d=5):
    print(f"\n=== PART 2: UNIMODAL Gaussian d={d}, GP-evals per effective sample ===")
    bounds = np.array([[-5.0, 5.0]] * d)
    cov = np.diag(np.linspace(1.0, 0.4, d) ** 2)
    prec = np.linalg.inv(cov)
    const = -0.5 * (d * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1])

    def logp(P):
        P = np.atleast_2d(P)
        return const - 0.5 * np.einsum("ni,ij,nj->n", P, prec, P, optimize=True)

    Xtr = np.clip(RNG.multivariate_normal(np.zeros(d), cov, size=12 * d),
                  bounds[:, 0], bounds[:, 1])
    gpr = fit("RBF", bounds, Xtr, logp(Xtr), nr=2 * d)

    print(f"  {'sampler':16s} {'evals':>8s} {'ESS':>6s} {'evals/ESS':>10s} "
          f"{'accept':>7s} {'extra':>16s}")

    # UltraNest
    acq = NORA(bounds=bounds, preprocessing_X=gpr.preprocessing_X,
               acq_func="LogExp", sampler="ultranest", verbose=0)
    gpr.n_eval = 0
    Xn, yn, syn, wn = acq.do_MC_sample(gpr, bounds=bounds,
                                       rng=np.random.default_rng(3), sampler="ultranest")
    ns_ess = (wn.sum() ** 2) / np.sum(wn ** 2)
    print(f"  {'ultranest':16s} {int(gpr.n_eval):>8d} {ns_ess:>6.0f} "
          f"{gpr.n_eval/ns_ess:>10.1f} {'-':>7s} {'-':>16s}")

    # numpy HMC (best-effort)
    gpr.n_eval = 0
    rh = mci.hmc_sample_gp_mean(gpr, bounds[:, 0], bounds[:, 1],
                                rng=np.random.default_rng(5), max_chains=8,
                                n_warmup=30, n_samples=60, thin=1, n_leapfrog=20)
    he = cross_chain_ess(rh["trace"])
    print(f"  {'numpy-HMC':16s} {int(gpr.n_eval):>8d} {he:>6.0f} "
          f"{gpr.n_eval/max(1e-9,he):>10.1f} {rh['accept_rate']:>7.2f} {'-':>16s}")

    # BlackJAX NUTS
    gpr.n_eval = 0
    rn = mci.nuts_sample_gp_mean(gpr, bounds[:, 0], bounds[:, 1],
                                 rng=np.random.default_rng(6), max_chains=8,
                                 n_warmup=200, n_samples=200)
    ne = cross_chain_ess(rn["trace"])
    tot = rn["n_eval"]
    print(f"  {'blackjax-NUTS':16s} {tot:>8d} {ne:>6.0f} "
          f"{tot/max(1e-9,ne):>10.1f} {rn['accept_rate']:>7.2f} "
          f"{'tree=%.1f div=%d' % (rn['mean_tree_size'], rn['divergences']):>16s}")
    print(f"    (NUTS sampling-only evals/ESS = "
          f"{rn['n_eval_sampling']/max(1e-9,ne):.1f}; warmup est. "
          f"{rn['n_eval_warmup']} evals)")


# ------------------------------------------------------------------ Part 3 --- #
def part3():
    print("\n=== PART 3: bimodal -> NUTS covers both seeded modes ===")
    m1, m2, s = np.array([-2., -2.]), np.array([2., 2.]), 0.6
    bounds = np.array([[-4., 4.]] * 2)

    def logp(P):
        P = np.atleast_2d(P)
        c = -np.log(2 * np.pi * s ** 2)
        l1 = c - 0.5 * np.sum((P - m1) ** 2, axis=1) / s ** 2
        l2 = c - 0.5 * np.sum((P - m2) ** 2, axis=1) / s ** 2
        return np.logaddexp(l1 + np.log(0.5), l2 + np.log(0.5))

    Xtr = np.clip(np.concatenate([m1 + 0.6 * RNG.standard_normal((25, 2)),
                                  m2 + 0.6 * RNG.standard_normal((25, 2)),
                                  RNG.uniform(-4, 4, size=(15, 2))]),
                  bounds[:, 0], bounds[:, 1])
    gpr = fit("RBF", bounds, Xtr, logp(Xtr))
    rn = mci.nuts_sample_gp_mean(gpr, bounds[:, 0], bounds[:, 1],
                                 rng=np.random.default_rng(7), max_chains=50,
                                 n_warmup=150, n_samples=150)
    S = rn["X"]
    f1 = np.mean(np.linalg.norm(S - m1, axis=1) < 1.2)
    f2 = np.mean(np.linalg.norm(S - m2, axis=1) < 1.2)
    print(f"  occupancy: mode1={f1:.2f}  mode2={f2:.2f}  accept={rn['accept_rate']:.2f}  "
          f"div={rn['divergences']}  --> both covered: {f1 > 0.15 and f2 > 0.15}")


if __name__ == "__main__":
    part1()
    part2(d=5)
    part3()
