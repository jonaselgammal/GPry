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
def save_gp(gpr, path):
    with open(path, "wb") as f:
        pickle.dump(gpr, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_gp(path):
    with open(path, "rb") as f:
        return pickle.load(f)
