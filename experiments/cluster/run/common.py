"""
Shared helpers for the cluster NUTS high-d tests.

Design: the EXPENSIVE acquisition run is done once and the GP is checkpointed
(pickled) at target point-counts. EVALUATION (cheap, re-runnable) loads a saved
GP and measures posterior recovery WITHOUT relying on UltraNest (which fails to
sample high-d GPs): a sampler-free Laplace check (GP mode + Hessian) plus a
well-tuned NUTS posterior for the corner.
"""
import json

import dill as pickle  # GPry's own serializer; handles Normalize_y's lambdas

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

# Quantile of the chi2_d radial distribution used as the "far from any mode"
# cut in the spurious-mass metric. 0.99 -> a perfect sample contributes 1%.
SPURIOUS_QUANTILE = 0.99

# Trim threshold (in base-sigma units) for the banana's inverse twist; see
# `evaluate_recovery_curved`. Exact samples stay below ~5.
Z_TRIM = 8.0


def _wcorr(a, b, w=None):
    """Weighted Pearson correlation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = np.ones(len(a)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    am, bm = a - np.average(a, weights=w), b - np.average(b, weights=w)
    va, vb = np.average(am ** 2, weights=w), np.average(bm ** 2, weights=w)
    if va <= 0 or vb <= 0:
        return 0.0
    return float(np.average(am * bm, weights=w) / np.sqrt(va * vb))


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


# --------------------------------------------------------------------------- #
# Non-Gaussian targets for the restart-efficiency study (d=5)
#
# The restart study must show that a cheaper hyperparameter-fit budget does not
# just go faster but stays ROBUST, so the low-d arm uses targets that a wrong
# length scale actually breaks: a curved (banana) degeneracy and a 4-mode
# mixture.  GPry's own test suite only ships 2d versions of these
# (`Curved_degeneracy`, `Ring`) and its n-dim `Himmelblau` silently leaves
# dimensions unconstrained, so both targets are built from scratch here.
#
# Both have EXACT closed-form samples, so evaluation never needs a sampler on
# the truth side (same design as `two_mode_reference_samples`).
# --------------------------------------------------------------------------- #
def make_curved(d, b=0.5, cond=10.0, n_twist=None, seed=None):
    """
    Chained twisted Gaussian ("banana") in d dimensions.

    Base z ~ N(0, diag(s^2)); the target is the pushforward under the chained
    twist::

        x_0 = z_0,   x_i = z_i + b * (z_{i-1}^2 - s_{i-1}^2)   (i >= 1)

    which is unit lower-triangular with unit Jacobian, so
    ``logp(x) = log N(z(x); 0, diag(s^2))`` with ``z(x)`` the (equally
    triangular, hence exact and cheap) inverse.

    Why this target: the twist is centred (``E[z^2] = s^2``), so the pushforward
    has mean 0 and a *diagonal* covariance ``diag(s_i^2 + 2 b^2 s_{i-1}^4)`` --
    i.e. the coordinates are linearly UNCORRELATED but strongly dependent.  A
    surrogate that collapses to a Gaussian blob therefore scores perfectly on
    every first- and second-moment metric while being completely wrong, which
    is exactly the failure a moment-based score would hide.  `evaluate_recovery
    _curved` catches it by mapping back through ``z(x)``.
    """
    s = np.geomspace(1.0, 1.0 / np.sqrt(cond), d)
    s2 = s ** 2
    # Number of twisted links. n_twist=1 is the classic single-bend banana;
    # chaining every link (n_twist=d-1) compounds the curvature and proved
    # unsolvable for GPry at any twist strength that was not already nearly
    # Gaussian -- see notes/RESTART_STUDY.md.
    n_twist = (d - 1) if n_twist is None else int(n_twist)
    n_twist = max(0, min(n_twist, d - 1))
    links = range(1, n_twist + 1)

    def forward(Z):
        """base -> target (vectorised over rows)"""
        Z = np.atleast_2d(np.asarray(Z, float))
        X = Z.copy()
        for i in links:
            X[:, i] = Z[:, i] + b * (Z[:, i - 1] ** 2 - s2[i - 1])
        return X

    def inverse(X):
        """target -> base (vectorised over rows)"""
        X = np.atleast_2d(np.asarray(X, float))
        Z = X.copy()
        for i in links:
            Z[:, i] = X[:, i] - b * (Z[:, i - 1] ** 2 - s2[i - 1])
        return Z

    const = -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(s2)))

    def logLkl(X):
        X = np.atleast_2d(np.asarray(X, float))
        Z = inverse(X)
        out = const - 0.5 * np.sum(Z ** 2 / s2, axis=1)
        return float(out[0]) if X.shape[0] == 1 else out

    # Analytic moments of the pushforward (see docstring): mean 0, diagonal cov.
    extra = np.zeros(d)
    for i in links:
        extra[i] = 2 * b ** 2 * s2[i - 1] ** 2
    marg = np.sqrt(s2 + extra)
    cov = np.diag(marg ** 2)
    # Bounds from a large EXACT sample: the banana is skewed, so a symmetric
    # k-sigma box would clip the arms.
    big = forward(np.random.default_rng(1234).standard_normal((200000, d)) * s)
    lo, hi = big.min(axis=0), big.max(axis=0)
    pad = 0.05 * (hi - lo)
    bounds = np.stack([lo - pad, hi + pad], axis=1)
    q = np.quantile(big, [0.1, 0.9], axis=0)
    ref_bounds = np.stack([q[0], q[1]], axis=1)
    return dict(logLkl=logLkl, bounds=bounds, ref_bounds=ref_bounds,
                mean=np.zeros(d), cov=cov, marg=marg, d=d, b=float(b),
                n_twist=n_twist, base_std=s, forward=forward, inverse=inverse,
                kind="curved")


def curved_reference_samples(target, n=40000, seed=123):
    """EXACT samples from the banana (push exact Gaussian draws forward)."""
    rng = np.random.default_rng(seed)
    d, s = target["d"], target["base_std"]
    return target["forward"](rng.standard_normal((n, d)) * s)


def make_multimode(d, n_modes=4, sep=6.0, weight_ratio=1.0, seed=7):
    """
    ``n_modes`` isotropic unit Gaussians whose centres are ``sep`` apart, placed
    along mutually orthogonal directions so that EVERY dimension separates the
    modes (unlike `make_two_mode`, which separates along axis 0 only).

    Centres are ``R * q_k`` for orthonormal ``q_k``, with ``R = sep / sqrt(2)``
    so that ``||c_i - c_j|| = R * sqrt(2) = sep`` for every pair.  Requires
    ``n_modes <= d``.  Weights are geometric with ratio ``weight_ratio``
    (1.0 -> equal).  Exact samples in closed form.
    """
    if n_modes > d:
        raise ValueError(f"n_modes={n_modes} > d={d}: need orthogonal centres.")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    means = (sep / np.sqrt(2.0)) * Q[:n_modes]
    cov = np.eye(d)
    marg = np.ones(d)
    w = np.geomspace(1.0, 1.0 / weight_ratio, n_modes)
    w = w / w.sum()
    logw = np.log(w)
    lognorm = -0.5 * d * np.log(2 * np.pi)   # slogdet(I) = 0

    def logLkl(X):
        X = np.atleast_2d(np.asarray(X, float))
        comps = np.empty((n_modes, X.shape[0]))
        for k in range(n_modes):
            dx = X - means[k]
            comps[k] = logw[k] + lognorm - 0.5 * np.sum(dx ** 2, axis=1)
        M = comps.max(axis=0)
        out = M + np.log(np.exp(comps - M).sum(axis=0))
        return float(out[0]) if X.shape[0] == 1 else out

    lo, hi = means.min(axis=0), means.max(axis=0)
    bounds = np.stack([lo - 6.0, hi + 6.0], axis=1)
    # ref_bounds must span every basin, else the initial design cannot seed them.
    ref_bounds = np.stack([lo - 2.0, hi + 2.0], axis=1)
    return dict(logLkl=logLkl, bounds=bounds, ref_bounds=ref_bounds,
                means=means, cov=cov, marg=marg, weights=w, sep=float(sep),
                weight_ratio=float(weight_ratio), n_modes=int(n_modes), d=d,
                kind="multimode")


def mixture_reference_samples(target, n=40000, seed=123):
    """EXACT samples from a K-mode unit-covariance mixture (closed form)."""
    rng = np.random.default_rng(seed)
    w, means, d = target["weights"], target["means"], target["d"]
    comp = rng.choice(len(w), size=n, p=w)
    return means[comp] + rng.standard_normal((n, d))


def evaluate_recovery_mixture(gpr, target, nuts_kwargs=None):
    """
    K-mode generalisation of `evaluate_recovery_2mode`: nearest-centre
    assignment (unit covariance -> Euclidean == Mahalanobis), mode recall,
    per-mode weight error, chi2_d-corrected spurious mass, and per-dimension
    1-Wasserstein against exact mixture samples.
    """
    from scipy.stats import wasserstein_distance
    nuts_kwargs = nuts_kwargs or {}
    X, w, info = nuts_corner_samples(gpr, target["bounds"], **nuts_kwargs)
    wn = np.asarray(w, float); wn = wn / wn.sum()
    means, wtrue, K = target["means"], target["weights"], len(target["weights"])

    dists = np.stack([np.linalg.norm(X - means[k], axis=1) for k in range(K)])
    assign = dists.argmin(axis=0)
    dmin = dists.min(axis=0)
    w_gp = np.array([wn[assign == k].sum() for k in range(K)])
    # "Found" = holds at least 10% of the weight it should have. Scaling by the
    # TRUE weight keeps the test equally strict for light and heavy modes.
    recall = [bool(w_gp[k] > 0.1 * wtrue[k]) for k in range(K)]

    d_dim = target["d"]
    thresh = np.sqrt(chi2.ppf(SPURIOUS_QUANTILE, d_dim))
    spurious = float(wn[dmin > thresh].sum())
    spurious_excess = max(0.0, spurious - (1.0 - SPURIOUS_QUANTILE))

    ref = mixture_reference_samples(target, n=min(len(X), 40000))
    wass = [float(wasserstein_distance(X[:, i], ref[:, i], u_weights=wn))
            for i in range(d_dim)]
    return dict(
        w_gp=[round(float(x), 4) for x in w_gp],
        w_true=[round(float(x), 4) for x in wtrue],
        w_relerr=round(float(np.max(np.abs(w_gp - wtrue) / wtrue)), 4),
        mode_recall=recall,
        n_modes_found=int(sum(recall)),
        n_modes_true=int(K),
        spurious_frac=round(spurious, 4),
        spurious_excess=round(spurious_excess, 4),
        wass_max=round(float(np.max(wass)), 4),
        ess=round(float(1.0 / np.sum(wn ** 2)), 0),
        pool=int(len(X)),
        n_divergent=int(info.get("divergences", -1)),
        accept=round(float(info.get("accept_rate", 0.0)), 3),
    ), (X, w)


def evaluate_recovery_curved(gpr, target, nuts_kwargs=None):
    """
    Recovery of the banana.  Moment metrics ALONE are useless here (the target
    has diagonal covariance, so a Gaussian blob scores perfectly), so the
    headline metric maps the GP samples back through the known inverse twist:
    under the truth ``z(x) ~ N(0, diag(s^2))`` exactly, so a Gaussian KL in
    z-space is zero iff the curvature was captured.  The x-space moments are
    reported alongside as an interpretable, but NOT sufficient, cross-check.
    """
    from scipy.stats import wasserstein_distance
    nuts_kwargs = nuts_kwargs or {}
    X, w, info = nuts_corner_samples(gpr, target["bounds"], **nuts_kwargs)
    wn = np.asarray(w, float); wn = wn / wn.sum()
    marg, d = target["marg"], target["d"]

    m = np.average(X, axis=0, weights=wn)
    c = np.cov(X.T, aweights=wn)
    # Curvature-sensitive metric: push back to the base coordinates.  The
    # inverse twist is a chained quadratic recursion, so it AMPLIFIES tail
    # error: on exact samples |z/s| stays below ~5, but on a wrong (blobby)
    # posterior it reaches ~1e7, which would let a handful of outliers dominate
    # the moments.  Trim at |z/s| <= Z_TRIM (discards ~0 of a correct sample)
    # and report the trimmed fraction, which is itself a failure signal.
    s = target["base_std"]
    Z = target["inverse"](X)
    keep = np.all(np.abs(Z / s) <= Z_TRIM, axis=1)
    z_outlier_frac = max(0.0, float(1.0 - wn[keep].sum()))
    if keep.sum() < 10:      # essentially nothing survived -> maximally wrong
        kl_z = float("inf")
    else:
        wk = wn[keep] / wn[keep].sum()
        mz = np.average(Z[keep], axis=0, weights=wk)
        cz = np.atleast_2d(np.cov(Z[keep].T, aweights=wk))
        kl_z = gaussian_kl(mz, cz, np.zeros(d), np.diag(s ** 2))
    # Bounded companion: the banana's signature is corr(x_{i-1}^2, x_i), which
    # is ~0.69 for this target and ~0 for any elliptical fit.  Unlike KL_z it
    # cannot blow up, so it stays readable when the fit is bad.
    def _curv(A, aw=None):
        return float(np.mean([_wcorr(A[:, i - 1] ** 2, A[:, i], aw)
                              for i in range(1, d)]))

    ref = curved_reference_samples(target, n=min(len(X), 40000))
    wass = [float(wasserstein_distance(X[:, i], ref[:, i], u_weights=wn))
            for i in range(d)]
    return dict(
        kl_z=round(float(kl_z), 4),                       # <- headline
        z_outlier_frac=round(z_outlier_frac, 4),
        curv_gp=round(_curv(X, wn), 4),                   # <- bounded companion
        curv_true=round(_curv(ref), 4),
        kl_gauss_x=round(gaussian_kl(m, np.atleast_2d(c), target["mean"],
                                     target["cov"]), 4),  # <- NOT sufficient
        max_mean_in_sigma=round(float(np.max(np.abs(m - target["mean"]) / marg)), 4),
        std_relerr=round(float(np.median(
            np.abs(np.sqrt(np.diag(c)) - marg) / marg)), 4),
        wass_max=round(float(np.max(wass)), 4),
        ess=round(float(1.0 / np.sum(wn ** 2)), 0),
        pool=int(len(X)),
        n_divergent=int(info.get("divergences", -1)),
        accept=round(float(info.get("accept_rate", 0.0)), 3),
    ), (X, w)


# --------------------------------------------------------------------------- #
# Per-iteration convergence checkpointing
#
# Motivation: a run with `DontConverge` (used to match arms on evaluation budget
# rather than on each sampler's own stopping rule) throws away all convergence
# information -- so answering "where would this run have stopped?" or drawing a
# convergence curve means re-running the whole thing. Saving the surrogate state
# and the acquisition chain at every point where the criterion WOULD have been
# evaluated makes both answerable after the fact, for ANY criterion, not just
# the one we happened to configure.
#
# Hooked in through GPry's own `callback=`, which fires once per acquisition
# iteration at exactly that point, so this needs no change to gpry core.
# --------------------------------------------------------------------------- #
class ConvergenceCheckpointer:
    """
    Save enough state per iteration to reconstruct the run a posteriori.

    Stores a compact NPZ per iteration rather than pickling the whole surrogate:
    the training set plus the kernel hyperparameters are sufficient to rebuild
    an equivalent GP, and are ~10x smaller (at d=30, n=690: ~200 kB against
    ~12 MB). Full surrogate pickles are written only every ``pickle_every``
    iterations, since they are the expensive part and are rarely all needed.

    Usage::

        ckpt = ConvergenceCheckpointer(outdir, tag)
        Runner(..., callback=ckpt, convergence_criterion="DontConverge")

    Parameters
    ----------
    outdir, tag : str
        Files are written to ``<outdir>/<tag>_conv/iter_<NNN>.npz``.
    pickle_every : int (default: 0)
        Also pickle the full surrogate every N iterations. 0 disables. The NPZ
        alone is enough to refit; use this only when the exact fitted object is
        needed.
    save_chain : bool (default: True)
        Store the acquisition MC chain. This is the bulky part; disable if only
        the GP trajectory matters.
    max_chain : int (default: 20000)
        Subsample the stored chain above this many points, so a long NORA
        sample does not dominate the checkpoint size. Subsampling is
        weight-aware (it keeps weights alongside).
    """

    def __init__(self, outdir, tag, pickle_every=0, save_chain=True,
                 max_chain=20000):
        import os

        self.dir = os.path.join(outdir, f"{tag}_conv")
        os.makedirs(self.dir, exist_ok=True)
        self.pickle_every = int(pickle_every)
        self.save_chain = bool(save_chain)
        self.max_chain = int(max_chain)
        self.n_calls = 0
        self.manifest = []

    def __call__(self, runner):
        import os

        i = self.n_calls
        self.n_calls += 1
        sur = runner.surrogate
        rec = {"iteration": i, "n_total": int(sur.n_total)}
        for attr in ("n_finite", "n_regress", "n_last_appended"):
            v = getattr(sur, attr, None)
            if v is not None:
                rec[attr] = int(v)

        payload = {
            "iteration": i,
            "n_total": int(sur.n_total),
            # Training set in the ORIGINAL parameter space, so the checkpoint does
            # not depend on the preprocessor state at read time.
            "X_train": np.asarray(sur.X, dtype=float),
            "y_train": np.asarray(sur.y, dtype=float),
            # Kernel hyperparameters (log-space theta) + a human-readable form.
            "kernel_theta": np.asarray(sur.gpr.kernel_.theta, dtype=float),
            "kernel_repr": str(sur.gpr.kernel_),
        }
        lml = getattr(sur.gpr, "log_marginal_likelihood_value_", None)
        if lml is not None:
            payload["lml"] = float(lml)

        # The acquisition chain for this iteration: what the criterion would have
        # been evaluated on.
        if self.save_chain:
            try:
                X, y, _sig, w = runner.acquisition.last_mc_sample(
                    copy=False, warn_reweight=False)
                X = np.asarray(X, dtype=float)
                w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
                if len(X) > self.max_chain:
                    sel = np.linspace(0, len(X) - 1, self.max_chain).astype(int)
                    X, w = X[sel], w[sel]
                    payload["chain_subsampled"] = True
                payload["chain_X"] = X
                payload["chain_w"] = w
                if y is not None:
                    yy = np.asarray(y, dtype=float)
                    payload["chain_y"] = yy[sel] if "chain_subsampled" in payload else yy
            except Exception as exc:  # chain is a bonus; never lose the GP state
                payload["chain_error"] = repr(exc)

        path = os.path.join(self.dir, f"iter_{i:04d}.npz")
        np.savez_compressed(path, **payload)
        rec["path"] = os.path.basename(path)

        if self.pickle_every and (i % self.pickle_every == 0):
            from copy import deepcopy

            ppath = os.path.join(self.dir, f"iter_{i:04d}_surrogate.pkl")
            try:
                save_gp(deepcopy(sur), ppath)
                rec["pickle"] = os.path.basename(ppath)
            except Exception as exc:
                rec["pickle_error"] = repr(exc)

        self.manifest.append(rec)
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump(self.manifest, f, indent=2)
        return None


def load_convergence_checkpoints(conv_dir):
    """
    Read back a `ConvergenceCheckpointer` directory.

    Returns a list of dicts, one per iteration, ordered by iteration. Each holds
    the arrays saved at that step, so a convergence criterion can be replayed
    over the trajectory without re-running anything.
    """
    import glob
    import os

    out = []
    for p in sorted(glob.glob(os.path.join(conv_dir, "iter_*.npz"))):
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["_path"] = p
        out.append(d)
    return sorted(out, key=lambda d: int(d["iteration"]))
