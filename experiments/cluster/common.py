"""
Shared helpers for the cluster NUTS high-d tests.

Design: the EXPENSIVE acquisition run is done once and the GP is checkpointed
(pickled) at target point-counts. EVALUATION (cheap, re-runnable) loads a saved
GP and measures posterior recovery WITHOUT relying on UltraNest (which fails to
sample high-d GPs): a sampler-free Laplace check (GP mode + Hessian) plus a
well-tuned NUTS posterior for the corner.
"""
import dill as pickle  # GPry's own serializer; handles Normalize_y's lambdas

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

# Quantile of the chi2_d radial distribution used as the "far from any mode"
# cut in the spurious-mass metric. 0.99 -> a perfect sample contributes 1%.
SPURIOUS_QUANTILE = 0.99


# --------------------------------------------------------------------------- #
# Target: anisotropic Gaussian, deterministic per dimension (target_seed=d)
# --------------------------------------------------------------------------- #
def make_gaussian(d, cond=10.0):
    rng = np.random.default_rng(d)  # fixed target per dimension
    stds = np.geomspace(1.0, 1.0 / np.sqrt(cond), d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    cov = Q @ np.diag(stds ** 2) @ Q.T
    cov = 0.5 * (cov + cov.T)
    prec = np.linalg.inv(cov)
    const = -0.5 * (d * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1])
    marg = np.sqrt(np.diag(cov))

    def logLkl(X):
        X = np.atleast_2d(np.asarray(X, float))
        out = const - 0.5 * np.einsum("ni,ij,nj->n", X, prec, X, optimize=True)
        return float(out[0]) if X.shape[0] == 1 else out

    bounds = np.array([[-6 * m, 6 * m] for m in marg])
    ref_bounds = np.array([[-2 * m, 2 * m] for m in marg])
    return dict(logLkl=logLkl, bounds=bounds, ref_bounds=ref_bounds,
                mean=np.zeros(d), cov=cov, marg=marg, d=d)


def gaussian_kl(m0, c0, m1, c1):
    c1i = np.linalg.inv(c1)
    dm = m1 - m0
    return float(0.5 * (np.trace(c1i @ c0) + dm @ c1i @ dm - len(m0)
                        + np.log(np.linalg.det(c1) / np.linalg.det(c0))))


# --------------------------------------------------------------------------- #
# Multimodal target: two isotropic unit Gaussians separated by `sep` sigma
# along axis 0.  weight_ratio = w[mode0]/w[mode1] (1.0 -> equal weights).
# This is the Tier-B separation-sweep target: the ONE knob is `sep`, so the
# mode-recall-vs-separation curve is a clean phase transition.  Unlike the
# unimodal make_gaussian(), the truth here has closed-form EXACT samples (draw
# a component, draw a Gaussian) -- so evaluation never needs a sampler for the
# reference side.
# --------------------------------------------------------------------------- #
def make_two_mode(d, sep, weight_ratio=1.0):
    cov = np.eye(d)                    # isotropic unit covariance per mode
    prec = np.eye(d)
    marg = np.ones(d)
    c0 = np.zeros(d); c0[0] = -0.5 * sep
    c1 = np.zeros(d); c1[0] = +0.5 * sep
    means = np.array([c0, c1])
    w = np.array([float(weight_ratio), 1.0]); w = w / w.sum()
    logw = np.log(w)
    lognorm = -0.5 * d * np.log(2 * np.pi)   # slogdet(I) = 0

    def logLkl(X):
        X = np.atleast_2d(np.asarray(X, float))
        comps = np.empty((2, X.shape[0]))
        for k in range(2):
            dx = X - means[k]
            comps[k] = logw[k] + lognorm - 0.5 * np.einsum(
                "ni,ij,nj->n", dx, prec, dx, optimize=True)
        M = comps.max(axis=0)
        out = M + np.log(np.exp(comps - M).sum(axis=0))
        return float(out[0]) if X.shape[0] == 1 else out

    half0 = 0.5 * sep + 6.0            # axis 0 must bracket both modes + 6 sigma
    bounds = np.array([[-half0, half0]] + [[-6.0, 6.0]] * (d - 1))
    # ref_bounds spans BOTH basins on axis 0 so the initial design seeds both.
    ref_bounds = np.array([[-0.5 * sep - 2.0, 0.5 * sep + 2.0]]
                          + [[-2.0, 2.0]] * (d - 1))
    return dict(logLkl=logLkl, bounds=bounds, ref_bounds=ref_bounds,
                means=means, cov=cov, marg=marg, weights=w, sep=float(sep),
                weight_ratio=float(weight_ratio), d=d)


def two_mode_reference_samples(target, n=40000, seed=123):
    """EXACT samples from the two-mode mixture (closed form -- no sampler)."""
    rng = np.random.default_rng(seed)
    w, means, cov, d = (target["weights"], target["means"],
                        target["cov"], target["d"])
    comp = rng.choice(2, size=n, p=w)
    L = np.linalg.cholesky(cov)
    return means[comp] + rng.standard_normal((n, d)) @ L.T


def evaluate_recovery_2mode(gpr, target, nuts_kwargs=None):
    """
    Multimodal recovery of the GP posterior, read out with well-tuned
    NUTS-many-chains and compared against the EXACT mixture.  Metrics:
      * mode_recall / n_modes_found -- did acquisition discover BOTH basins?
      * w_gp vs w_true, w_relerr     -- relative mode WEIGHTS (the GP Achilles heel)
      * spurious_frac                -- GP mass >5 sigma from either true mode
      * wass_x0 / wass_max           -- 1-Wasserstein vs exact samples (x0 is bimodal)
    """
    from scipy.stats import wasserstein_distance
    nuts_kwargs = nuts_kwargs or {}
    X, w, info = nuts_corner_samples(gpr, target["bounds"], **nuts_kwargs)
    wn = np.asarray(w, float); wn = wn / wn.sum()
    means, wtrue = target["means"], target["weights"]

    d0 = np.linalg.norm(X - means[0], axis=1)   # unit cov -> Euclidean = Mahalanobis
    d1 = np.linalg.norm(X - means[1], axis=1)
    assign = (d1 < d0).astype(int)
    dmin = np.minimum(d0, d1)
    w_gp = np.array([wn[assign == 0].sum(), wn[assign == 1].sum()])
    recall = [bool(w_gp[k] > 0.02) for k in range(2)]
    # Spurious mass: fraction sitting far from BOTH modes. The threshold MUST be
    # dimension-aware: ||x - m||^2 ~ chi2_d for a perfect unit-covariance sample,
    # so a fixed Euclidean cut flags a growing fraction of *correct* samples with
    # d (a naive 5-sigma cut flags ~7% of perfect d=16 draws). Use a fixed
    # chi2_d quantile instead, and additionally report the excess over the
    # analytically expected tail so the number is comparable across dimensions.
    d_dim = target["d"]
    thresh = np.sqrt(chi2.ppf(SPURIOUS_QUANTILE, d_dim))
    spurious = float(wn[dmin > thresh].sum())
    spurious_excess = max(0.0, spurious - (1.0 - SPURIOUS_QUANTILE))

    ref = two_mode_reference_samples(target, n=min(len(X), 40000))
    wass = [float(wasserstein_distance(X[:, i], ref[:, i], u_weights=wn))
            for i in range(target["d"])]
    rec = dict(
        w_gp=[round(float(x), 4) for x in w_gp],
        w_true=[round(float(x), 4) for x in wtrue],
        w_relerr=round(float(np.max(np.abs(w_gp - wtrue) / wtrue)), 4),
        mode_recall=recall,
        n_modes_found=int(sum(recall)),
        spurious_frac=round(spurious, 4),
        spurious_excess=round(spurious_excess, 4),
        spurious_threshold=round(float(thresh), 3),
        wass_x0=round(wass[0], 4),
        wass_max=round(float(np.max(wass)), 4),
        ess=round(float(1.0 / np.sum(wn ** 2)), 0),
        pool=int(len(X)),
        n_divergent=int(info.get("divergences", -1)),
        accept=round(float(info.get("accept_rate", 0.0)), 3),
    )
    return rec, (X, w)


# --------------------------------------------------------------------------- #
# Sampler-free evaluator: Laplace approximation of the GP posterior
# --------------------------------------------------------------------------- #
def laplace_cov_at(gpr, bounds, x0):
    """
    Laplace covariance (-Hessian^-1 of the GP mean) at a GIVEN point x0 (use the
    NUTS posterior mean -- robust, unlike the fragile argmax of a bumpy GP mean).
    Sampler-free cross-check of the covariance. Uses gpr.predict (mean only).
    """
    bounds = np.asarray(bounds, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    d = len(lo)
    x0 = np.clip(np.asarray(x0, float), lo, hi)

    def mu(x):
        return float(np.ravel(gpr.predict(np.atleast_2d(np.clip(x, lo, hi))))[0])

    step = 1e-3 * (hi - lo)
    H = np.zeros((d, d))
    for i in range(d):
        ei = np.zeros(d); ei[i] = step[i]
        for j in range(i, d):
            ej = np.zeros(d); ej[j] = step[j]
            H[i, j] = H[j, i] = (mu(x0 + ei + ej) - mu(x0 + ei - ej)
                                 - mu(x0 - ei + ej) + mu(x0 - ei - ej)) \
                / (4 * step[i] * step[j])
    negH = -0.5 * (H + H.T)
    w, V = np.linalg.eigh(negH)
    # Only valid if the GP mean is concave at x0 (all eigenvalues > 0). On real
    # GPs the SVM/clip_factor can flatten/kink the surface -> not PD -> invalid.
    if np.min(w) <= 1e-6 * np.max(np.abs(w)):
        return None
    cov = (V * (1.0 / w)) @ V.T
    return 0.5 * (cov + cov.T)


def evaluate_recovery(gpr, bounds, true_mean, true_cov, marg, nuts_kwargs=None):
    """
    Primary reliable evaluator: well-tuned NUTS-many-chains posterior of the GP
    (NOT UltraNest). Reports weighted mean/cov recovery vs truth, plus a
    sampler-free Laplace-covariance cross-check anchored at the NUTS mean.
    Returns the metrics dict AND the (samples, weights) for the corner plot.
    """
    nuts_kwargs = nuts_kwargs or {}
    X, w, info = nuts_corner_samples(gpr, bounds, **nuts_kwargs)
    wn = np.asarray(w, float); wn = wn / wn.sum()
    m = np.average(X, axis=0, weights=wn)
    c = np.cov(X.T, aweights=wn)
    cov_lap = laplace_cov_at(gpr, bounds, m)  # sampler-free cross-check (best-effort)
    std_lap = (round(float(np.median(np.abs(np.sqrt(np.diag(cov_lap)) - marg) / marg)), 4)
               if cov_lap is not None else float("nan"))
    rec = dict(
        kl_nuts=round(gaussian_kl(m, np.atleast_2d(c), true_mean, true_cov), 4),
        max_mean_in_sigma=round(float(np.max(np.abs(m - true_mean) / marg)), 4),
        std_relerr_nuts=round(float(np.median(
            np.abs(np.sqrt(np.diag(c)) - marg) / marg)), 4),
        std_relerr_laplace=std_lap,
        ess=round(float(1.0 / np.sum(wn ** 2)), 0),
        pool=int(len(X)),
        n_divergent=int(info.get("divergences", -1)),
        accept=round(float(info.get("accept_rate", 0.0)), 3),
        mean=m.tolist(),
    )
    return rec, (X, w)


# --------------------------------------------------------------------------- #
# Reliable posterior SAMPLES for the corner (well-tuned NUTS, NOT UltraNest)
# --------------------------------------------------------------------------- #
def nuts_corner_samples(gpr, bounds, n_chains=64, n_warmup=300, n_samples=300):
    from gpry.mc_interfaces import nuts_acquire
    X, _, _, w, info = nuts_acquire(
        gpr, np.asarray(bounds, float), rng=np.random.default_rng(123),
        return_info=True, max_chains=n_chains, n_warmup=n_warmup,
        n_samples=n_samples, pad_multiple=512, max_num_doublings=10)
    return X, w, info


# --------------------------------------------------------------------------- #
# pickling
# --------------------------------------------------------------------------- #
def save_gp(surrogate, path):
    with open(path, "wb") as f:
        pickle.dump(surrogate, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_gp(path):
    with open(path, "rb") as f:
        return pickle.load(f)
