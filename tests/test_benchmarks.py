"""
Comprehensive benchmark test suite for GPry's surrogate model.

Contains standalone test likelihood functions spanning a range of difficulties:
multimodal, ill-conditioned, funnel-shaped, shell-like, and cosmology-inspired.

Each benchmark provides: likelihood function, bounds, true_mean (if known),
and a description string.
"""

import sys
import os
import time
import tempfile
import shutil
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===========================================================================
# 1. Multimodal Gaussian Mixture (2D and 5D)
# ===========================================================================

def make_gaussian_mixture_2d():
    """3-component Gaussian mixture in 2D with well-separated modes."""
    weights = np.array([0.5, 0.3, 0.2])
    log_weights = np.log(weights)
    means = [
        np.array([-3.0, -3.0]),
        np.array([3.0, 3.0]),
        np.array([-2.0, 4.0]),
    ]
    covs = [
        np.array([[0.5, 0.1], [0.1, 0.5]]),
        np.array([[0.8, -0.3], [-0.3, 0.8]]),
        np.array([[0.3, 0.0], [0.0, 0.6]]),
    ]
    rvs = [multivariate_normal(m, c) for m, c in zip(means, covs)]

    def logLkl(x_0, x_1):
        pt = np.array([x_0, x_1])
        log_components = np.array([
            lw + rv.logpdf(pt) for lw, rv in zip(log_weights, rvs)
        ])
        return logsumexp(log_components)

    bounds = [[-8, 8], [-8, 8]]
    # True mean is the weighted average of component means
    true_mean = sum(w * m for w, m in zip(weights, means))
    return logLkl, bounds, true_mean, "Gaussian Mixture 2D"


def make_gaussian_mixture_5d():
    """3-component Gaussian mixture in 5D with well-separated modes."""
    rng = np.random.RandomState(123)
    ndim = 5
    weights = np.array([0.5, 0.3, 0.2])
    log_weights = np.log(weights)
    means = [
        np.array([-3.0, -3.0, 0.0, 1.0, -1.0]),
        np.array([3.0, 3.0, 0.0, -1.0, 1.0]),
        np.array([0.0, 0.0, 3.0, 2.0, -2.0]),
    ]
    covs = []
    for _ in range(3):
        A = rng.randn(ndim, ndim) * 0.3
        covs.append(A @ A.T + 0.3 * np.eye(ndim))
    rvs = [multivariate_normal(m, c) for m, c in zip(means, covs)]

    def logLkl(x_0, x_1, x_2, x_3, x_4):
        pt = np.array([x_0, x_1, x_2, x_3, x_4])
        log_components = np.array([
            lw + rv.logpdf(pt) for lw, rv in zip(log_weights, rvs)
        ])
        return logsumexp(log_components)

    bounds = [[-8, 8]] * ndim
    true_mean = sum(w * m for w, m in zip(weights, means))
    return logLkl, bounds, true_mean, "Gaussian Mixture 5D"


# ===========================================================================
# 2. Eggbox (2D)
# ===========================================================================

def make_eggbox_2d():
    """Eggbox likelihood: highly multimodal periodic structure."""

    def logLkl(x_0, x_1):
        return (2.0 + np.cos(x_0 / 2.0) * np.cos(x_1 / 2.0)) ** 5

    bounds = [[0.0, 10.0 * np.pi], [0.0, 10.0 * np.pi]]
    # The function is periodic; no single true mean
    return logLkl, bounds, None, "Eggbox 2D"


# ===========================================================================
# 3. Neal's Funnel (10D)
# ===========================================================================

def make_neals_funnel_10d():
    """Neal's funnel distribution: extreme scale variation across dimensions.
    log p = -v^2/18 - (d-1)/2 * v - sum(x_i^2) / (2*exp(v))
    where v = x_0 and x_i for i=1..9 are the remaining dimensions.
    """

    def logLkl(x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9):
        v = x_0
        xs = np.array([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9])
        d_minus_1 = 9
        log_p = -v ** 2 / 18.0 - d_minus_1 / 2.0 * v - np.sum(xs ** 2) / (2.0 * np.exp(v))
        return log_p

    bounds = [[-20, 20]] * 10
    # True mean is zero for all dimensions
    true_mean = np.zeros(10)
    return logLkl, bounds, true_mean, "Neal's Funnel 10D"


# ===========================================================================
# 4. Gaussian Shells (2D and 5D)
# ===========================================================================

def _shell_logpdf(x, center, radius, width):
    """Log-pdf of a thin spherical shell."""
    dist = np.sqrt(np.sum((x - center) ** 2))
    return -0.5 * ((dist - radius) / width) ** 2


def make_gaussian_shells_2d():
    """Two thin spherical shells in 2D."""
    c1 = np.array([-1.5, 0.0])
    c2 = np.array([1.5, 0.0])
    r1, r2 = 2.0, 2.0
    w1, w2 = 0.1, 0.1

    def logLkl(x_0, x_1):
        pt = np.array([x_0, x_1])
        lp1 = _shell_logpdf(pt, c1, r1, w1)
        lp2 = _shell_logpdf(pt, c2, r2, w2)
        return logsumexp([lp1, lp2])

    bounds = [[-6, 6], [-6, 6]]
    # By symmetry the mean is at the origin
    true_mean = np.array([0.0, 0.0])
    return logLkl, bounds, true_mean, "Gaussian Shells 2D"


def make_gaussian_shells_5d():
    """Two thin spherical shells in 5D."""
    ndim = 5
    c1 = np.zeros(ndim)
    c1[0] = -1.5
    c2 = np.zeros(ndim)
    c2[0] = 1.5
    r1, r2 = 2.0, 2.0
    w1, w2 = 0.1, 0.1

    def logLkl(x_0, x_1, x_2, x_3, x_4):
        pt = np.array([x_0, x_1, x_2, x_3, x_4])
        lp1 = _shell_logpdf(pt, c1, r1, w1)
        lp2 = _shell_logpdf(pt, c2, r2, w2)
        return logsumexp([lp1, lp2])

    bounds = [[-6, 6]] * ndim
    true_mean = np.zeros(ndim)
    return logLkl, bounds, true_mean, "Gaussian Shells 5D"


# ===========================================================================
# 5. Cigar / Ill-conditioned Gaussian (10D)
# ===========================================================================

def make_cigar_10d():
    """Ill-conditioned Gaussian: one long axis (var=100), rest short (var=0.01)."""
    ndim = 10
    mean = np.zeros(ndim)
    # Eigenvalues: first is 100, rest are 0.01
    eigenvalues = np.full(ndim, 0.01)
    eigenvalues[0] = 100.0
    # Random rotation for the covariance
    rng = np.random.RandomState(99)
    Q, _ = np.linalg.qr(rng.randn(ndim, ndim))
    cov = Q @ np.diag(eigenvalues) @ Q.T
    # Ensure symmetry
    cov = (cov + cov.T) / 2.0
    rv = multivariate_normal(mean, cov)

    def logLkl(x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9):
        pt = np.array([x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9])
        return rv.logpdf(pt)

    # Bounds wide enough to contain the long axis
    bounds = [[-30, 30]] * ndim
    return logLkl, bounds, mean.copy(), "Cigar 10D"


# ===========================================================================
# 6. Twisted/Warped Gaussian — Haario's Banana (2D and 5D)
# ===========================================================================

def make_banana_2d():
    """Haario's banana distribution in 2D.
    x2_twisted = x2 - b*(x1^2 - 100), then Gaussian in twisted coords.
    """
    b = 0.1
    sigma1_sq = 100.0  # wide in x1
    sigma2_sq = 1.0    # narrow in twisted x2

    def logLkl(x_0, x_1):
        x1_tw = x_0
        x2_tw = x_1 - b * (x_0 ** 2 - 100.0)
        return -0.5 * (x1_tw ** 2 / sigma1_sq + x2_tw ** 2 / sigma2_sq)

    bounds = [[-30, 30], [-30, 30]]
    # True mean: x1=0, x2 = b*(E[x1^2] - 100) = b*(sigma1_sq - 100) = 0
    true_mean = np.array([0.0, 0.0])
    return logLkl, bounds, true_mean, "Banana 2D"


def make_banana_5d():
    """Haario's banana in 5D: first two dims are banana-shaped, rest Gaussian."""
    b = 0.1
    sigma1_sq = 100.0
    sigma2_sq = 1.0
    sigma_rest_sq = 1.0

    def logLkl(x_0, x_1, x_2, x_3, x_4):
        x1_tw = x_0
        x2_tw = x_1 - b * (x_0 ** 2 - 100.0)
        logp = -0.5 * (x1_tw ** 2 / sigma1_sq + x2_tw ** 2 / sigma2_sq)
        # Remaining dimensions are independent Gaussians
        rest = np.array([x_2, x_3, x_4])
        logp += -0.5 * np.sum(rest ** 2 / sigma_rest_sq)
        return logp

    bounds = [[-30, 30], [-30, 30], [-5, 5], [-5, 5], [-5, 5]]
    true_mean = np.zeros(5)
    return logLkl, bounds, true_mean, "Banana 5D"


# ===========================================================================
# 7. Rastrigin (2D and 5D)
# ===========================================================================

def make_rastrigin_2d():
    """Rastrigin function as negative log-likelihood (2D).
    Massively multimodal with a global optimum at the origin.
    """
    A = 10.0

    def logLkl(x_0, x_1):
        xs = np.array([x_0, x_1])
        n = len(xs)
        val = A * n + np.sum(xs ** 2 - A * np.cos(2.0 * np.pi * xs))
        return -val

    bounds = [[-5.12, 5.12], [-5.12, 5.12]]
    true_mean = np.array([0.0, 0.0])
    return logLkl, bounds, true_mean, "Rastrigin 2D"


def make_rastrigin_5d():
    """Rastrigin function as negative log-likelihood (5D)."""
    A = 10.0

    def logLkl(x_0, x_1, x_2, x_3, x_4):
        xs = np.array([x_0, x_1, x_2, x_3, x_4])
        n = len(xs)
        val = A * n + np.sum(xs ** 2 - A * np.cos(2.0 * np.pi * xs))
        return -val

    bounds = [[-5.12, 5.12]] * 5
    true_mean = np.array([0.0] * 5)
    return logLkl, bounds, true_mean, "Rastrigin 5D"


# ===========================================================================
# 8. SN Cosmology (2D)
# ===========================================================================

def make_sn_cosmology_2d():
    """Simple Type Ia supernova cosmology: fit Omega_m and w.

    Generates 50 mock SNe from a fiducial cosmology (Omega_m=0.3, w=-1.0),
    computes luminosity distances via numerical integration, and returns
    a chi-squared log-likelihood.
    """
    from scipy.integrate import quad

    # Fiducial cosmology
    Om_fid = 0.3
    w_fid = -1.0
    n_sn = 50
    sigma_mu = 0.15  # magnitude scatter

    def _luminosity_distance(z, Om, w):
        """Comoving distance in flat wCDM."""
        def integrand(zp):
            Ode = 1.0 - Om
            Ez2 = Om * (1.0 + zp) ** 3 + Ode * (1.0 + zp) ** (3.0 * (1.0 + w))
            if Ez2 <= 0:
                return 1e10  # unphysical
            return 1.0 / np.sqrt(Ez2)
        dc, _ = quad(integrand, 0, z)
        return (1.0 + z) * dc

    # Generate mock data with fixed seed
    rng = np.random.RandomState(2024)
    z_sn = np.linspace(0.01, 1.5, n_sn)
    mu_true = np.array([
        5.0 * np.log10(_luminosity_distance(z, Om_fid, w_fid)) + 25.0
        for z in z_sn
    ])
    # Add Gaussian noise
    mu_obs = mu_true + rng.normal(0, sigma_mu, n_sn)

    def logLkl(x_0, x_1):
        """x_0 = Omega_m, x_1 = w"""
        Om = x_0
        w = x_1
        if Om <= 0 or Om >= 1:
            return -1e30
        mu_model = np.empty(n_sn)
        for i, z in enumerate(z_sn):
            dl = _luminosity_distance(z, Om, w)
            if dl <= 0:
                return -1e30
            mu_model[i] = 5.0 * np.log10(dl) + 25.0
        chi2 = np.sum(((mu_obs - mu_model) / sigma_mu) ** 2)
        return -0.5 * chi2

    bounds = [[0.05, 0.8], [-2.0, 0.0]]
    true_mean = np.array([Om_fid, w_fid])
    return logLkl, bounds, true_mean, "SN Cosmology 2D"


# ===========================================================================
# Registry of all benchmarks
# ===========================================================================

ALL_BENCHMARKS = {
    "gmm_2d": make_gaussian_mixture_2d,
    "gmm_5d": make_gaussian_mixture_5d,
    "eggbox_2d": make_eggbox_2d,
    "funnel_10d": make_neals_funnel_10d,
    "shells_2d": make_gaussian_shells_2d,
    "shells_5d": make_gaussian_shells_5d,
    "cigar_10d": make_cigar_10d,
    "banana_2d": make_banana_2d,
    "banana_5d": make_banana_5d,
    "rastrigin_2d": make_rastrigin_2d,
    "rastrigin_5d": make_rastrigin_5d,
    "sn_cosmo_2d": make_sn_cosmology_2d,
}


# ===========================================================================
# Runner helper
# ===========================================================================

def run_benchmark(logLkl, bounds, label, surrogate_cfg=None, options=None,
                  seed=42, verbose=0, gp_acquisition=None,
                  convergence_criterion=None, initial_proposer=None):
    """Run GPry on a benchmark and return a results dict.

    Parameters
    ----------
    logLkl : callable
        Log-likelihood function with scalar arguments.
    bounds : list of [lo, hi]
        Parameter bounds.
    label : str
        Human-readable name for the benchmark.
    surrogate_cfg : dict, optional
        Configuration passed to Runner(surrogate=...).
    options : dict, optional
        Options passed to Runner(options=...). Defaults to
        max_finite=200, max_total=400.
    seed : int
        Random seed.
    verbose : int
        Verbosity level.
    gp_acquisition : dict or object, optional
        Acquisition function config passed to Runner(gp_acquisition=...).
    convergence_criterion : optional
        Convergence criterion passed to Runner.
    initial_proposer : optional
        Initial proposer passed to Runner.

    Returns
    -------
    dict with keys: label, converged, n_total, time, mean_mc, error
    """
    from gpry.run import Runner
    from gpry.tools import mean_covmat_from_samples

    if surrogate_cfg is None:
        surrogate_cfg = {
            "regressor": {"kernel": "RBF", "use_jax": False},
        }
    if options is None:
        options = {"max_finite": 200, "max_total": 400}

    tmpdir = tempfile.mkdtemp(prefix=f"gpry_bench_{label.replace(' ', '_')}_")
    checkpoint = os.path.join(tmpdir, "run")

    result = {
        "label": label,
        "converged": False,
        "n_total": None,
        "time": None,
        "mean_mc": None,
        "error": None,
    }

    runner_kwargs = {
        "surrogate": surrogate_cfg,
        "checkpoint": checkpoint,
        "load_checkpoint": "overwrite",
        "verbose": verbose,
        "seed": seed,
        "options": options,
    }
    if gp_acquisition is not None:
        runner_kwargs["gp_acquisition"] = gp_acquisition
    if convergence_criterion is not None:
        runner_kwargs["convergence_criterion"] = convergence_criterion
    if initial_proposer is not None:
        runner_kwargs["initial_proposer"] = initial_proposer

    try:
        t0 = time.time()
        runner = Runner(logLkl, bounds, **runner_kwargs)
        runner.run()
        t_total = time.time() - t0

        result["converged"] = runner.has_converged
        result["n_total"] = runner.surrogate.n_total
        result["time"] = t_total

        mc = runner.last_mc_samples()
        if mc is not None:
            mean_mc, cov_mc = mean_covmat_from_samples(mc["X"], mc["w"])
            result["mean_mc"] = mean_mc

    except Exception as e:
        import traceback
        result["time"] = time.time() - t0
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result


# ===========================================================================
# Main: quick validation on easy benchmarks
# ===========================================================================

if __name__ == "__main__":
    # Run a subset of easy benchmarks with a modest budget
    easy_benchmarks = [
        ("gmm_2d", make_gaussian_mixture_2d),
        ("banana_2d", make_banana_2d),
        ("cigar_10d", make_cigar_10d),
    ]

    max_finite = 200

    print("=" * 78)
    print("  GPry Benchmark Suite — Quick Validation")
    print("=" * 78)
    print(f"  Budget: max_finite={max_finite}")
    print()

    # Table header
    header = f"  {'Benchmark':<22} {'Status':<8} {'Evals':<7} {'Time (s)':<10} {'Mean Err':<12}"
    print(header)
    print("  " + "-" * 72)

    all_passed = True

    for key, make_fn in easy_benchmarks:
        logLkl, bounds, true_mean, name = make_fn()

        ndim = len(bounds)
        mf = max(max_finite, int(50 * ndim**1.5))  # scale budget with dimensionality
        options = {"max_finite": mf, "max_total": mf * 3}
        r = run_benchmark(logLkl, bounds, name, options=options, verbose=0)

        status = "CONV" if r["converged"] else "NCONV"
        n = str(r["n_total"]) if r["n_total"] is not None else "ERR"
        t = f"{r['time']:.1f}" if r["time"] is not None else "N/A"

        if r.get("mean_mc") is not None and true_mean is not None:
            mean_err = np.max(np.abs(r["mean_mc"] - true_mean))
            err_str = f"{mean_err:.4f}"
        elif r.get("mean_mc") is not None:
            err_str = f"mean={np.array2string(r['mean_mc'], precision=3)}"
        else:
            err_str = "N/A"

        row = f"  {name:<22} {status:<8} {n:<7} {t:<10} {err_str:<12}"
        print(row)

        if r.get("error"):
            print(f"    ERROR: {r['error'][:70]}")
            if "traceback" in r:
                # Print last few lines of traceback
                tb_lines = r["traceback"].strip().split("\n")
                for line in tb_lines[-3:]:
                    print(f"    {line}")
            all_passed = False

    print("  " + "-" * 72)
    if all_passed:
        print("  All quick-validation benchmarks completed.")
    else:
        print("  Some benchmarks had errors — see above.")
    print("=" * 78)
