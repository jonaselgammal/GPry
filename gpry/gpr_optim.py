"""Shared hyperparameter-restart logic for the GPry GP backends.

Both ``gpr_sklearn.py`` and ``gpr_jax.py`` used to carry copy-pasted helper
closures (kernel-structure extraction, top-y training subset, theta-guess
construction, local-covariance / local-quadratic deterministic seeds, the
Latin-hypercube restart pool, and the post-scoring boundary penalties /
pathological-optimum filter). This module owns that logic so that the
backends differ only in (a) the actual L-BFGS-B / jaxopt invocation and
(b) the ``lml_callable`` they hand in.

The functions here are backend-agnostic and operate on plain numpy arrays;
they take the (possibly sklearn-wrapped) kernel object plus training arrays
as explicit arguments rather than reaching into a GPR instance.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc  # type: ignore


__all__ = [
    "kernel_structure",
    "top_training_subset",
    "build_theta_guess",
    "estimate_output_scale",
    "theta_guess_from_local_cov",
    "theta_guess_from_local_quadratic",
    "build_restart_candidates",
    "score_and_filter_candidates",
    "boundary_penalty",
    "is_pathological_optimum",
]


# ---------------------------------------------------------------------------
# Kernel-structure inspection and training-data slicing
# ---------------------------------------------------------------------------


def kernel_structure(kernel):
    """Decompose a (possibly compound) sklearn-style kernel into the
    ``(constant_kernel, length_kernel, noise_kernel)`` triple GPry expects.

    Returns ``(None, None, noise_kernel_or_None)`` if the kernel is not a
    product of (constant * length) (+ noise).
    """
    if kernel is None:
        return None, None, None
    noise_kernel = None
    product_kernel = kernel
    if hasattr(kernel, "k1") and hasattr(kernel, "k2"):
        if hasattr(kernel.k2, "noise_level"):
            noise_kernel = kernel.k2
            product_kernel = kernel.k1
    if not (hasattr(product_kernel, "k1") and hasattr(product_kernel, "k2")):
        return None, None, noise_kernel
    return product_kernel.k1, product_kernel.k2, noise_kernel


def top_training_subset(X_train, y_train):
    """Return the highest-y subset of training points used to seed
    local-covariance / local-quadratic hyperparameter starts.

    Returns ``(None, None)`` if there aren't enough points.
    """
    if X_train is None:
        return None, None
    X_train = np.asarray(X_train)
    if X_train.ndim != 2:
        return None, None
    n_train = len(X_train)
    d_train = X_train.shape[1]
    if n_train < max(6, d_train + 1):
        return None, None
    y_train = np.asarray(y_train).reshape(-1)
    n_keep = min(
        n_train,
        max(3 * d_train, 12),
        max(6, int(np.ceil(0.35 * n_train))),
    )
    idx = np.argsort(y_train)[-n_keep:]
    return np.asarray(X_train[idx], dtype=float), y_train[idx]


# ---------------------------------------------------------------------------
# theta-guess construction
# ---------------------------------------------------------------------------


def build_theta_guess(
    kernel,
    *,
    hyperparameter_bounds,
    length_scales=None,
    output_scale=None,
):
    """Assemble a theta vector (log space) from explicit length-scale and
    output-scale guesses, clipped to ``hyperparameter_bounds``.

    Returns ``None`` if the kernel structure doesn't expose
    ``(constant, length[, noise])`` or the produced theta has the wrong
    dimensionality.
    """
    constant_kernel, length_kernel, noise_kernel = kernel_structure(kernel)
    if constant_kernel is None or length_kernel is None:
        return None
    theta_actual = []
    if output_scale is None:
        output_scale_sq = float(constant_kernel.constant_value)
    else:
        output_scale_sq = float(output_scale) ** 2
    output_bounds = np.exp(np.asarray(constant_kernel.bounds, dtype=float))
    output_scale_sq = float(
        np.clip(output_scale_sq, output_bounds[0, 0], output_bounds[0, 1])
    )
    theta_actual.append(output_scale_sq)

    ls_bounds = np.exp(np.asarray(length_kernel.bounds, dtype=float))
    ls_current = np.atleast_1d(length_kernel.length_scale).astype(float)
    ls_guess = (
        ls_current if length_scales is None
        else np.atleast_1d(length_scales).astype(float)
    )
    if ls_guess.shape[0] != ls_current.shape[0]:
        return None
    ls_guess = np.clip(ls_guess, ls_bounds[:, 0], ls_bounds[:, 1])
    theta_actual.extend(ls_guess.tolist())
    if noise_kernel is not None:
        noise_bounds = np.exp(np.asarray(noise_kernel.bounds, dtype=float))
        noise_level = float(
            np.clip(
                float(noise_kernel.noise_level),
                noise_bounds[0, 0],
                noise_bounds[0, 1],
            )
        )
        theta_actual.append(noise_level)
    theta = np.log(np.asarray(theta_actual, dtype=float))
    if theta.shape[0] != hyperparameter_bounds.shape[0]:
        return None
    return np.clip(theta, hyperparameter_bounds[:, 0], hyperparameter_bounds[:, 1])


def estimate_output_scale(y_subset, y_full=None):
    """Estimate an output-scale starting value from a y-subset.

    Falls back to the full y vector's std if the subset is degenerate, and
    finally to 1.0. The minimum returned value is 1e-3.
    """
    if y_subset is None or len(y_subset) == 0:
        return None
    y_scale = float(np.std(y_subset))
    if not np.isfinite(y_scale) or y_scale <= 0:
        if y_full is not None:
            y_scale = float(np.std(np.asarray(y_full).reshape(-1)))
        else:
            y_scale = 0.0
    if not np.isfinite(y_scale) or y_scale <= 0:
        y_scale = 1.0
    return max(y_scale, 1e-3)


def theta_guess_from_local_cov(
    kernel, X_train, y_train, *, hyperparameter_bounds
):
    """Length-scale seed from per-coordinate std of the top-y training subset."""
    X_subset, y_subset = top_training_subset(X_train, y_train)
    if X_subset is None or len(X_subset) < 3:
        return None
    if len(X_subset) == 1:
        diag_scales = np.full(X_subset.shape[1], 0.1)
    else:
        cov = np.cov(X_subset.T, ddof=0)
        cov = np.atleast_2d(cov)
        diag_scales = np.sqrt(np.maximum(np.diag(cov), 1e-6))
    diag_scales = np.maximum(diag_scales, 1e-3)
    return build_theta_guess(
        kernel,
        hyperparameter_bounds=hyperparameter_bounds,
        length_scales=diag_scales,
        output_scale=estimate_output_scale(y_subset, y_full=y_train),
    )


def theta_guess_from_local_quadratic(
    kernel, X_train, y_train, *, hyperparameter_bounds
):
    """Length-scale seed from a local quadratic fit around the best y."""
    X_subset, y_subset = top_training_subset(X_train, y_train)
    if X_subset is None:
        return None
    d = X_subset.shape[1]
    n_subset = len(X_subset)
    min_points = max(2 * d + 3, d + 4)
    if n_subset < min_points:
        return None
    x0 = X_subset[np.argmax(y_subset)]
    dx = X_subset - x0
    columns = [np.ones(n_subset)]
    columns.extend(dx[:, i] for i in range(d))
    quad_terms = []
    for i in range(d):
        for j in range(i, d):
            quad_terms.append((i, j))
            columns.append(dx[:, i] * dx[:, j])
    design = np.column_stack(columns)
    ridge = 1e-6 * np.eye(design.shape[1])
    ridge[0, 0] = 0.0
    try:
        beta = np.linalg.solve(design.T @ design + ridge, design.T @ y_subset)
    except np.linalg.LinAlgError:
        return None
    hessian = np.zeros((d, d), dtype=float)
    coeffs = beta[1 + d:]
    for coeff, (i, j) in zip(coeffs, quad_terms):
        if i == j:
            hessian[i, i] = -2.0 * coeff
        else:
            value = -coeff
            hessian[i, j] = value
            hessian[j, i] = value
    curvature = np.maximum(np.diag(hessian), 1e-6)
    output_scale = estimate_output_scale(y_subset, y_full=y_train)
    length_scales = np.sqrt(max(output_scale, 1e-6) / curvature)
    return build_theta_guess(
        kernel,
        hyperparameter_bounds=hyperparameter_bounds,
        length_scales=length_scales,
        output_scale=output_scale,
    )


# ---------------------------------------------------------------------------
# Restart-candidate generation
# ---------------------------------------------------------------------------


def _rng_uint32(rng):
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(0, 2**32 - 1))
    return int(rng.randint(0, 2**32 - 1))


def _sample_restart_pool(n_samples, *, hyperparameter_bounds, rng):
    if n_samples <= 0:
        return []
    sampler = qmc.LatinHypercube(
        d=hyperparameter_bounds.shape[0],
        seed=_rng_uint32(rng),
    )
    unit = sampler.random(n=n_samples)
    return qmc.scale(
        unit, hyperparameter_bounds[:, 0], hyperparameter_bounds[:, 1]
    )


def build_restart_candidates(
    kernel,
    X_train,
    y_train,
    *,
    hyperparameter_bounds,
    n_random,
    rng,
    prev_theta=None,
    start_from_current=False,
):
    """Build the unordered list of theta start points handed to the optimizer.

    Order (matching the legacy in-place generation):

    1. ``prev_theta`` if ``start_from_current`` and ``prev_theta`` is not None.
    2. Local-covariance seed (if computable).
    3. Local-quadratic seed (if computable).
    4. ``n_random`` Latin-hypercube samples in ``hyperparameter_bounds``.

    Returns a list of 1-D numpy arrays (each clipped to bounds where
    applicable).
    """
    hyperparameter_bounds = np.asarray(hyperparameter_bounds, dtype=float)
    candidates = []
    if start_from_current and prev_theta is not None:
        candidates.append(np.array(prev_theta, copy=True))
    for deterministic_guess in (
        theta_guess_from_local_cov(
            kernel, X_train, y_train,
            hyperparameter_bounds=hyperparameter_bounds,
        ),
        theta_guess_from_local_quadratic(
            kernel, X_train, y_train,
            hyperparameter_bounds=hyperparameter_bounds,
        ),
    ):
        if deterministic_guess is not None:
            candidates.append(np.asarray(deterministic_guess, dtype=float))
    for theta_initial in _sample_restart_pool(
        n_random,
        hyperparameter_bounds=hyperparameter_bounds,
        rng=rng,
    ):
        candidates.append(np.asarray(theta_initial, dtype=float))
    return candidates


# ---------------------------------------------------------------------------
# Boundary penalty / pathological-optimum detection
# ---------------------------------------------------------------------------


def boundary_penalty(theta, hyperparameter_bounds):
    """Return a non-negative penalty rewarding theta away from bounds."""
    theta = np.asarray(theta, dtype=float)
    hyperparameter_bounds = np.asarray(hyperparameter_bounds, dtype=float)
    bounds_span = np.maximum(
        hyperparameter_bounds[:, 1] - hyperparameter_bounds[:, 0], 1e-12
    )
    lower_margin = (theta - hyperparameter_bounds[:, 0]) / bounds_span
    upper_margin = (hyperparameter_bounds[:, 1] - theta) / bounds_span
    margin = np.minimum(lower_margin, upper_margin)
    clipped = np.clip(0.02 - margin, 0.0, None)
    return float(np.sum(clipped / 0.02))


def is_pathological_optimum(theta, hyperparameter_bounds):
    """Detect optima with a majority of components pinned to the bounds."""
    theta = np.asarray(theta, dtype=float)
    hyperparameter_bounds = np.asarray(hyperparameter_bounds, dtype=float)
    bounds_span = np.maximum(
        hyperparameter_bounds[:, 1] - hyperparameter_bounds[:, 0], 1e-12
    )
    lower_margin = (theta - hyperparameter_bounds[:, 0]) / bounds_span
    upper_margin = (hyperparameter_bounds[:, 1] - theta) / bounds_span
    margin = np.minimum(lower_margin, upper_margin)
    close_to_bounds = margin < 0.01
    return int(np.sum(close_to_bounds)) >= max(2, int(np.ceil(len(theta) / 2)))


# ---------------------------------------------------------------------------
# Pre-optimizer scoring / filtering
# ---------------------------------------------------------------------------


def score_and_filter_candidates(
    thetas,
    neg_lml_callable,
    *,
    hyperparameter_bounds,
    n_select,
):
    """Evaluate ``neg_lml_callable(theta) -> float`` for each candidate (no
    gradient, no optimization), sort by ascending (more negative LML is
    better, so this corresponds to descending LML), break ties by lower
    boundary penalty, deduplicate, and return up to ``n_select`` theta
    vectors.

    The callable must return the **negative** log-marginal likelihood (the
    objective the optimizer will minimize). Returning ``inf`` / ``nan`` /
    raising are all treated as "skip this candidate".
    """
    scored = []
    for theta_initial in thetas:
        try:
            value = float(neg_lml_callable(np.asarray(theta_initial)))
        except Exception:
            continue
        if np.isfinite(value):
            scored.append((value, np.asarray(theta_initial)))
    if not scored:
        raise RuntimeError("Failed to produce any finite hyperparameter start.")
    scored.sort(
        key=lambda item: (item[0], boundary_penalty(item[1], hyperparameter_bounds))
    )
    selected = []
    seen = set()
    for _, theta_initial in scored:
        key = tuple(np.round(theta_initial, decimals=12))
        if key in seen:
            continue
        seen.add(key)
        selected.append(theta_initial)
        if len(selected) >= n_select:
            break
    return selected
