"""GP-JAX linear algebra and runtime state for the JAX GP backend.

This module owns the JAX-side linear-algebra primitives used by
``gpr_jax.py``:

- Cholesky-based fit precomputation (``_gp_fit_precompute``),
- predictive mean / variance (``_gp_predict_mean``, ``_gp_predict_var``,
  ``_gp_predict_mean_and_var``),
- compiled log-marginal likelihood factory (``_make_lml_fn``),
- compiled single-point predict / acquisition factories,
- the ``JaxGaussianProcessMixin`` that stores the JAX-side fitted state and
  drives hyperparameter / acquisition optimization via jaxopt.

Kernel-family math no longer lives here — it has moved to
``kernels_jax.py``. The JAX backend dispatches to the right kernel function
by calling ``kernel.evaluate_jax_fn()`` on the (numpy) kernel object, removing
the previous ``type(kernel)`` switch.

This module will be renamed to ``gpr_jax_linalg.py`` to match its post-Phase-4
role; the rename happens together with the kernel split.
"""

import warnings
from copy import deepcopy

import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from jax.scipy.linalg import cho_solve, solve_triangular
import numpy as np

# Ensure JAX uses 64-bit floats for numerical accuracy
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# GP operations
# ---------------------------------------------------------------------------

@jit
def _gp_fit_precompute(K_train, y_train, alpha_noise):
    """Precompute L (Cholesky) and alpha_ (dual coefficients) from kernel matrix.

    Parameters
    ----------
    K_train : array (n, n)
        Kernel matrix K(X_train, X_train) with output_scale already applied.
    y_train : array (n,)
        Training targets.
    alpha_noise : float or array (n,)
        Noise to add to diagonal.

    Returns
    -------
    L : array (n, n)
        Lower Cholesky factor of (K + noise * I).
    V : array (n, n)
        L^{-1} (for variance computation).
    alpha_ : array (n,)
        (K + noise * I)^{-1} @ y_train.
    """
    n = K_train.shape[0]
    K_noisy = K_train + alpha_noise * jnp.eye(n)
    L = jnp.linalg.cholesky(K_noisy)
    V = solve_triangular(L, jnp.eye(n), lower=True)
    alpha_ = cho_solve((L, True), y_train)
    return L, V, alpha_


@jit
def _gp_fit_precompute_padded(K_padded, y_padded, mask, alpha_noise, pad_alpha):
    """Padded counterpart of ``_gp_fit_precompute``.

    The padded kernel matrix is *block-diagonalized*: cross terms between
    valid and invalid rows/cols are zeroed and the invalid diagonal is set
    to ``pad_alpha``. After adding ``alpha_noise * I`` the matrix has the
    structure

        [ K_valid + α·I        0          ]
        [    0          (α + pad_alpha)·I ]

    whose Cholesky factor inherits the block-diagonal structure. As a
    consequence:

      * ``L_padded[:n_valid, :n_valid] = L_unpadded`` (machine precision),
      * ``alpha_padded[:n_valid]`` matches the unpadded solve because
        ``y_padded[n_valid:] == 0`` (kills the invalid block's contribution),
      * ``V_padded[:n_valid, :n_valid] = V_unpadded`` and the invalid block
        of ``V_padded`` is ``1/sqrt(α + pad_alpha) · I`` -- it only ever
        multiplies the already-zeroed invalid columns of ``K_trans`` in
        downstream predict ops, so it contributes nothing.

    Math parity vs. the unpadded reference verified to ~1e-15 in
    ``/tmp/gpry_pad_microtest{,2}.py``.

    Parameters
    ----------
    K_padded : array (N, N)
        Kernel matrix on the padded training set (output-scaled).
    y_padded : array (N,)
        Training targets, padded with zeros past ``n_valid``.
    mask : array (N,) bool
        First ``n_valid`` entries True.
    alpha_noise : float or array (N,)
        Diagonal noise term.
    pad_alpha : float (static)
        Positive constant placed on the invalid diagonal for conditioning.
    """
    N = K_padded.shape[0]
    mask_f = mask.astype(K_padded.dtype)
    mask_outer = mask_f[:, None] * mask_f[None, :]
    inv_diag = (1.0 - mask_f) * pad_alpha
    K_masked = K_padded * mask_outer + jnp.diag(inv_diag)
    K_noisy = K_masked + alpha_noise * jnp.eye(N)
    L = jnp.linalg.cholesky(K_noisy)
    V = solve_triangular(L, jnp.eye(N), lower=True)
    alpha_ = cho_solve((L, True), y_padded)
    return L, V, alpha_


@jit
def _gp_append_block_update(L_valid, V_valid, A, B, y_old, y_new):
    """Block-append Cholesky / V / alpha update (Phase R-B).

    Given a fitted GP with Cholesky factor ``L_valid`` of size ``(n, n)``,
    extend the training set by ``m`` new points whose cross-covariance with
    the old data is ``A`` (shape ``(n, m)``) and self-covariance plus noise
    is ``B`` (shape ``(m, m)``).  Returns the new ``L``, ``V``, and dual
    coefficients ``alpha`` -- all in O((n+m)^2) work, vs O((n+m)^3) for a
    full re-Cholesky.

    Math
    ----
    Given  K_n = L_n L_n^T  and the augmented matrix
        K_{n+m} = [ K_n  A ; A^T  B ],
    the extended Cholesky factor is
        L_{n+m} = [ L_n  0 ; U^T  D ]
    with  U = L_n^{-1} A   (triangular solve)
          D D^T = B - U^T U   (small m-by-m Cholesky)
    The matching update for  V = L^{-1}  is
        V_{n+m} = [ V_n  0 ; -D^{-1} U^T V_n   D^{-1} ].
    Dual coefficients  alpha = K^{-1} y  are then a cho_solve on the new L.

    Numerical safety
    ----------------
    The diagonal of ``S = B - U^T U`` is floored at ``1e-12`` to absorb
    roundoff that would otherwise make the m-by-m Cholesky fail on a
    nearly-rank-deficient append (e.g., a near-duplicate training point).
    Microtest ``/tmp/gpry_chol_append_microtest.py`` verified drift of
    ~7.8e-15 over 100 sequential single-point appends — no periodic
    refresh required at typical GP-acquisition scales.
    """
    n = L_valid.shape[0]
    m = B.shape[0]
    # U = L_n^{-1} A (triangular solve, O(n^2 m))
    U = solve_triangular(L_valid, A, lower=True)
    # S = B - U^T U,  symmetrize + floor diagonal
    S = B - U.T @ U
    S = 0.5 * (S + S.T)
    diag = jnp.diag(S)
    safe_diag = jnp.maximum(diag, 1e-12)
    S = S - jnp.diag(diag) + jnp.diag(safe_diag)
    D = jnp.linalg.cholesky(S)
    # Assemble L_{n+m}
    L_new = jnp.concatenate([
        jnp.concatenate([L_valid, jnp.zeros((n, m))], axis=1),
        jnp.concatenate([U.T, D], axis=1),
    ], axis=0)
    # Assemble V_{n+m}
    D_inv = solve_triangular(D, jnp.eye(m), lower=True)
    bot_left = -D_inv @ (U.T @ V_valid)
    V_new = jnp.concatenate([
        jnp.concatenate([V_valid, jnp.zeros((n, m))], axis=1),
        jnp.concatenate([bot_left, D_inv], axis=1),
    ], axis=0)
    # alpha = K^{-1} y via cho_solve on the new L (O((n+m)^2))
    y_new_full = jnp.concatenate([y_old, y_new], axis=0)
    alpha_new = cho_solve((L_new, True), y_new_full)
    return L_new, V_new, alpha_new


@jit
def _gp_predict_mean(K_trans, alpha_):
    """Predict mean: K(X_new, X_train) @ alpha_."""
    return K_trans @ alpha_


@jit
def _gp_predict_var(K_trans, V, k_diag):
    """Predict variance: k(x,x) - K_trans @ K_inv @ K_trans.T (diagonal only).

    Parameters
    ----------
    K_trans : array (n_new, n_train)
        Cross-covariance K(X_new, X_train).
    V : array (n_train, n_train)
        L^{-1} from Cholesky.
    k_diag : array (n_new,)
        Prior variance diagonal k(x,x).

    Returns
    -------
    var : array (n_new,)
        Predictive variance (clipped to >= 0).
    """
    M = V @ K_trans.T  # (n_train, n_new)
    var = k_diag - jnp.sum(M ** 2, axis=0)
    return jnp.maximum(var, 0.0)


@jit
def _gp_predict_mean_and_var(K_trans, alpha_, V, k_diag):
    """Predict both mean and variance in one call."""
    y_mean = K_trans @ alpha_
    M = V @ K_trans.T
    y_var = k_diag - jnp.sum(M ** 2, axis=0)
    y_var = jnp.maximum(y_var, 0.0)
    return y_mean, y_var


# ---------------------------------------------------------------------------
# Differentiable LML for automatic gradient computation
# ---------------------------------------------------------------------------

def _make_lml_fn(kernel_fn, has_white_noise, n_dims):
    """Create a differentiable LML function for a specific kernel type.

    Parameters
    ----------
    kernel_fn : callable
        Kernel function (X1, X2, length_scale) -> K.
    has_white_noise : bool
        Whether theta includes a white noise parameter at the end.
    n_dims : int
        Number of input dimensions (static, baked into the function).

    Returns
    -------
    lml_fn : callable
        Function(theta, X_train, y_train, alpha_noise) -> scalar LML.
        Differentiable w.r.t. theta via jax.grad.
    """
    @jit
    def lml_fn(theta, X_train, y_train, alpha_noise):
        # Unpack theta (log-space hyperparameters):
        # theta[0] = log(constant_value) where constant_value = output_scale^2
        # theta[1:1+n_dims] = log(length_scale_i)
        # theta[1+n_dims] = log(white_noise_level) (if has_white_noise)
        output_scale_sq = jnp.exp(theta[0])
        length_scale = jnp.exp(theta[1:1 + n_dims])

        # Build kernel matrix
        K = kernel_fn(X_train, X_train, length_scale)
        K = output_scale_sq * K

        # Add white noise if present
        n = X_train.shape[0]
        if has_white_noise:
            white_noise = jnp.exp(theta[1 + n_dims])
            K = K + white_noise * jnp.eye(n)

        # Add alpha noise (fixed noise term)
        K = K + alpha_noise * jnp.eye(n)

        # Cholesky and solve
        L = jnp.linalg.cholesky(K)
        alpha_vec = cho_solve((L, True), y_train)

        # LML = -0.5 * y^T K^{-1} y - 0.5 * log|K| - n/2 * log(2*pi)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        lml = (-0.5 * jnp.dot(y_train, alpha_vec)
               - 0.5 * log_det
               - 0.5 * n * jnp.log(2.0 * jnp.pi))
        return lml

    return lml_fn


def _make_lml_padded_fn(kernel_fn, has_white_noise, n_dims, pad_alpha):
    """Create a differentiable LML function operating on *padded* buffers.

    Same contract as ``_make_lml_fn`` but the input arrays may be padded to
    a fixed capacity ``N >= n_valid``. The ``mask`` argument is a bool
    array of length ``N`` whose first ``n_valid`` entries are True. Invalid
    rows/cols of ``K`` are zeroed and their diagonals are set to
    ``pad_alpha`` so the resulting padded matrix is block-diagonal:
    ``[K_valid + αI, 0; 0, (pad_alpha + α + (wn?))·I]``. The Cholesky
    factor inherits this block structure, so:

      * the dual coefficients on the valid block are identical to the
        unpadded solve (the invalid block of ``y_padded`` is zero and
        contributes nothing),
      * the log-determinant of the *valid* block is recovered by
        ``2 * sum(mask * log(diag(L)))`` — the invalid-block diagonal
        contributions are masked out.

    Math parity vs. the unpadded reference is machine-precision; see
    ``/tmp/gpry_pad_microtest{,2}.py`` for the validation suite.

    Returns
    -------
    lml_fn : callable
        Function(theta, X_padded, y_padded, mask, alpha_noise) -> scalar LML.
        Differentiable w.r.t. theta via jax.grad.
    """
    @jit
    def lml_fn(theta, X_padded, y_padded, mask, alpha_noise):
        output_scale_sq = jnp.exp(theta[0])
        length_scale = jnp.exp(theta[1:1 + n_dims])

        K = kernel_fn(X_padded, X_padded, length_scale)
        K = output_scale_sq * K

        # Block-diagonalize K via the mask; invalid diag gets pad_alpha.
        mask_f = mask.astype(K.dtype)
        mask_outer = mask_f[:, None] * mask_f[None, :]
        inv_diag = (1.0 - mask_f) * pad_alpha
        K = K * mask_outer + jnp.diag(inv_diag)

        N = X_padded.shape[0]
        if has_white_noise:
            white_noise = jnp.exp(theta[1 + n_dims])
            K = K + white_noise * jnp.eye(N)
        K = K + alpha_noise * jnp.eye(N)

        L = jnp.linalg.cholesky(K)
        alpha_vec = cho_solve((L, True), y_padded)

        # Valid-block log-det only. Without the mask we'd add
        # (N - n_valid) * log(diag_invalid) and corrupt the LML.
        log_det_valid = 2.0 * jnp.sum(mask_f * jnp.log(jnp.diag(L)))
        n_valid = jnp.sum(mask_f)
        lml = (-0.5 * jnp.dot(y_padded, alpha_vec)
               - 0.5 * log_det_valid
               - 0.5 * n_valid * jnp.log(2.0 * jnp.pi))
        return lml

    return lml_fn


# ---------------------------------------------------------------------------
# Differentiable predict functions for automatic gradient w.r.t. input x
# ---------------------------------------------------------------------------

def _make_predict_mean_single_fn(kernel_fn):
    """Create a differentiable predict_mean for a single point x.

    Returns a function f(x, X_train, alpha_, length_scale, output_scale_sq) -> scalar.
    """
    @jit
    def predict_mean_single(x, X_train, alpha_, length_scale, output_scale_sq):
        # x shape: (n_dims,), reshape to (1, n_dims)
        x_2d = x[None, :]
        K_trans = kernel_fn(x_2d, X_train, length_scale)  # (1, n_train)
        K_trans = output_scale_sq * K_trans
        return jnp.dot(K_trans[0], alpha_)

    return predict_mean_single


def _build_neg_acq_fn(predict_mean_fn, predict_std_fn):
    """Build a static JIT-compiled acquisition function.

    The returned function takes all parameters as explicit arguments (not
    closure captures), so JAX traces it once and reuses the compiled code
    for any float values of zeta, noise_var, baseline.
    """
    @jit
    def neg_acq(x, X_train, alpha_, V_, length_scale,
                output_scale_sq, white_noise_level,
                zeta, noise_var, baseline):
        mu = predict_mean_fn(x, X_train, alpha_, length_scale,
                             output_scale_sq)
        std = predict_std_fn(x, X_train, V_, length_scale,
                             output_scale_sq, white_noise_level)
        var_excess = std ** 2 - noise_var ** 2
        safe_var = jnp.maximum(var_excess, 1e-30)
        acq = 2.0 * zeta * (mu - baseline) + 0.5 * jnp.log(safe_var)
        return -acq
    return neg_acq


def _make_predict_std_single_fn(kernel_fn):
    """Create a differentiable predict_std for a single point x.

    Returns a function f(x, X_train, V_, length_scale, output_scale_sq,
                          white_noise_level) -> scalar.
    """
    @jit
    def predict_std_single(x, X_train, V_, length_scale, output_scale_sq,
                           white_noise_level):
        x_2d = x[None, :]
        K_trans = kernel_fn(x_2d, X_train, length_scale)  # (1, n_train)
        K_trans = output_scale_sq * K_trans
        k_diag = output_scale_sq + white_noise_level
        M = V_ @ K_trans[0]  # (n_train,)
        y_var = k_diag - jnp.dot(M, M)
        y_var = jnp.maximum(y_var, 1e-30)  # small epsilon for differentiability
        return jnp.sqrt(y_var)

    return predict_std_single


# ---------------------------------------------------------------------------
# Padded predict / acq factories (Phase B-3)
# ---------------------------------------------------------------------------

def _make_predict_mean_single_padded_fn(kernel_fn):
    """Single-point predict mean operating on padded ``X_padded``.

    The ``mask`` zeros out the contribution from invalid training rows: the
    cross-covariance ``K(x, X_padded[i])`` for ``i >= n_valid`` is multiplied
    by ``mask_f[i] = 0``, and the ``alpha_padded[i]`` slots for those rows
    are zero by construction (see ``_refresh_padded_state``). The valid block
    of the dot product is exactly the unpadded answer.
    """
    @jit
    def predict_mean_single_pad(x, X_padded, alpha_padded, mask,
                                 length_scale, output_scale_sq):
        x_2d = x[None, :]
        K_trans = kernel_fn(x_2d, X_padded, length_scale)  # (1, capacity)
        K_trans = output_scale_sq * K_trans
        mask_f = mask.astype(K_trans.dtype)
        K_trans = K_trans * mask_f[None, :]
        return jnp.dot(K_trans[0], alpha_padded)

    return predict_mean_single_pad


def _make_predict_std_single_padded_fn(kernel_fn):
    """Single-point predict std operating on padded buffers.

    ``V_padded`` is the full Cholesky inverse of the padded ``K``; on the
    valid block it equals the unpadded ``V``, and on the invalid block it is
    a fixed identity-scaled block (see ``_refresh_padded_state``). The mask
    zeros out invalid columns of ``K_trans``, which kills the
    invalid-block's contribution to ``M @ M``.
    """
    @jit
    def predict_std_single_pad(x, X_padded, V_padded, mask,
                                length_scale, output_scale_sq,
                                white_noise_level):
        x_2d = x[None, :]
        K_trans = kernel_fn(x_2d, X_padded, length_scale)
        K_trans = output_scale_sq * K_trans
        mask_f = mask.astype(K_trans.dtype)
        K_trans = K_trans * mask_f[None, :]
        k_diag = output_scale_sq + white_noise_level
        M = V_padded @ K_trans[0]
        y_var = k_diag - jnp.dot(M, M)
        y_var = jnp.maximum(y_var, 1e-30)
        return jnp.sqrt(y_var)

    return predict_std_single_pad


def _build_neg_acq_padded_fn(predict_mean_fn, predict_std_fn):
    """Build a static JIT'd negative-acquisition function over padded
    buffers. Same LogExp form as ``_build_neg_acq_fn`` but threads
    ``mask`` through the inner predict calls.
    """
    @jit
    def neg_acq(x, X_padded, alpha_padded, V_padded, mask,
                length_scale, output_scale_sq, white_noise_level,
                zeta, noise_var, baseline):
        mu = predict_mean_fn(x, X_padded, alpha_padded, mask,
                              length_scale, output_scale_sq)
        std = predict_std_fn(x, X_padded, V_padded, mask,
                              length_scale, output_scale_sq, white_noise_level)
        var_excess = std ** 2 - noise_var ** 2
        safe_var = jnp.maximum(var_excess, 1e-30)
        acq = 2.0 * zeta * (mu - baseline) + 0.5 * jnp.log(safe_var)
        return -acq
    return neg_acq


# ---------------------------------------------------------------------------
# Internal JAX GP implementation mixin
# ---------------------------------------------------------------------------

class JaxGaussianProcessMixin:
    """Internal JAX state and compiled helpers for the concrete JAX GP backend.

    This mixin holds the JAX-native GP state derived from a fitted model and the
    compiled helper functions that operate on that state. It is part of the JAX
    GP implementation itself, not an optional acceleration object attached to a
    separate sklearn-backed regressor.

    Padded-buffer state (Phase B)
    -----------------------------
    To eliminate ``X_train.shape[0]``-driven JIT recompiles between acquisition
    iterations, the mixin keeps a *padded* copy of the training state in
    addition to the live (unpadded) arrays:

    * ``_X_train_padded``  shape ``(_capacity, n_dims)``
    * ``_y_train_padded``  shape ``(_capacity,)``
    * ``_alpha_padded``    shape ``(_capacity,)``
    * ``_V_padded``        shape ``(_capacity, _capacity)``
    * ``_L_padded``        shape ``(_capacity, _capacity)``
    * ``_mask``            bool array shape ``(_capacity,)`` — first
      ``_n_valid`` entries are ``True``, rest are ``False``.

    Padded compute paths use the block-diagonal Cholesky trick:
    ``K_padded = K * mask_outer + diag((1 - mask) * _PAD_ALPHA)``, so the
    invalid sub-block of the Cholesky factor is a fixed identity-scaled block
    that decouples from ``theta``, ``y``, and the test point. Predict / LML
    parity vs. the unpadded reference was verified to machine precision in
    ``/tmp/gpry_pad_microtest{,2}.py`` before this state was wired in.

    The capacity grows by doubling (``_ensure_capacity``), so the total number
    of compiles over the whole acquisition sweep is ``O(log n_total)`` rather
    than ``O(n_iter)``. The initial capacity is dim-aware (see
    ``_choose_initial_capacity``) and overridable per instance via
    ``_initial_capacity_override``.
    """

    # Positive scalar placed on the unused diagonal of K so its sub-block is
    # well-conditioned. Any positive value gives identical predict/LML/grad
    # outputs on the valid block — see microtest #2 — so we pick 1.0.
    _PAD_ALPHA = 1.0

    def __init__(self):
        self._kernel_fn = None
        self._X_train = None
        self._y_train = None
        self._length_scale = None
        self._output_scale_sq = None
        self._white_noise_level = 0.0  # from WhiteKernel if present
        self._alpha_ = None
        self._V_ = None
        self._L_ = None
        self._alpha_noise = None
        self._n_dims = None
        self._has_white_noise = False
        self._lml_value_and_grad_fn = None
        self._predict_mean_grad_fn = None
        self._predict_std_grad_fn = None
        self._predict_mean_single_fn = None
        self._predict_std_single_fn = None
        self._neg_acq_fn = None  # static JIT-compiled acq function
        self._cached_acq_solver = None  # reusable jaxopt.LBFGSB
        self._cached_acq_solver_fn_id = None  # id of _neg_acq_fn for cache invalidation
        self._ready = False
        # Phase B padded-buffer state (parallel to the unpadded arrays;
        # populated by _refresh_padded_state in update_from_gpr /
        # prime_fit_inputs). Compute paths are still unpadded in B-1; B-2 / B-3
        # swap them in.
        self._X_train_padded = None
        self._y_train_padded = None
        self._alpha_padded = None
        self._V_padded = None
        self._L_padded = None
        self._mask = None
        self._capacity = 0
        self._n_valid = 0
        self._alpha_noise_padded = None  # scalar or shape-(capacity,) array
        # Optional per-instance override of the initial capacity. If ``None``
        # the dim-aware heuristic in ``_choose_initial_capacity`` is used.
        self._initial_capacity_override = None
        # Phase R-B: counts online appends since last full Cholesky. Drives
        # the periodic-refresh safety net (forces a full re-fit every
        # ``_REFRESH_EVERY`` appends to bound any roundoff drift).
        self._appends_since_full_refit = 0

    # ------------------------------------------------------------------
    # Padded-buffer capacity policy (Phase B)
    # ------------------------------------------------------------------

    def _choose_initial_capacity(self, n_dims):
        """Pick the starting buffer capacity given input dimensionality.

        Heuristic (overridable via ``_initial_capacity_override``):

        * ``n_dims < 4``  →  32   (low-d problems converge in 30-100 evals)
        * ``4 ≤ n_dims ≤ 7`` → 64
        * ``n_dims ≥ 8``  → 128   (higher-d typically needs more samples)

        The exact thresholds are heuristic; the doubling policy means the
        first few acquisition iterations may pay one extra recompile if the
        initial capacity is too small, but capacity changes are O(log
        n_total) regardless. Picking a starting size that *matches* the
        common steady-state ``n_train`` saves those early recompiles.
        """
        if self._initial_capacity_override is not None:
            return int(self._initial_capacity_override)
        if n_dims is None or n_dims < 4:
            return 32
        if n_dims <= 7:
            return 64
        return 128

    def _ensure_capacity(self, n_needed):
        """Grow ``_capacity`` (by doubling) until it is at least ``n_needed``.

        Returns ``True`` if the capacity changed (caller must invalidate any
        cached JIT'd code that closed over ``_capacity``), else ``False``.
        """
        if self._capacity == 0:
            self._capacity = self._choose_initial_capacity(self._n_dims)
            changed = True
        else:
            changed = False
        while self._capacity < n_needed:
            self._capacity *= 2
            changed = True
        return changed

    def _refresh_padded_state(self, X_train, y_train,
                              alpha_=None, V_=None, L_=None):
        """Rebuild ``_X_train_padded`` / ``_y_train_padded`` / ``_mask`` from
        the live arrays. ``alpha_`` / ``V_`` / ``L_`` may be ``None`` if the
        caller hasn't run a fit yet (``prime_fit_inputs`` case).

        Capacity is grown as needed by doubling; the helper is the *only*
        place the padded buffers are populated, so the invariants
        (``_mask[:n_valid] == True``, padded arrays zero outside the valid
        block except for the placeholder on L/V) live here.
        """
        n_valid = X_train.shape[0]
        self._ensure_capacity(n_valid)
        capacity = self._capacity
        n_dims = X_train.shape[1]

        # X / y padding
        X_padded = np.zeros((capacity, n_dims), dtype=np.float64)
        X_padded[:n_valid] = np.asarray(X_train)
        y_padded = np.zeros(capacity, dtype=np.float64)
        y_padded[:n_valid] = np.asarray(y_train)
        mask = np.zeros(capacity, dtype=bool)
        mask[:n_valid] = True

        self._X_train_padded = jnp.array(X_padded)
        self._y_train_padded = jnp.array(y_padded)
        self._mask = jnp.array(mask)
        self._n_valid = int(n_valid)

        # Pad per-point alpha_noise if it is an array. Scalar noise stays
        # as-is; jnp broadcasting handles it. Invalid-block entries are
        # filled with 0.0 -- the invalid diagonal is already pinned to
        # _PAD_ALPHA by the mask logic in the LML / fit functions, so any
        # value here is safe.
        if (self._alpha_noise is not None
                and not isinstance(self._alpha_noise, (float, int))
                and hasattr(self._alpha_noise, "shape")
                and len(self._alpha_noise.shape) >= 1):
            an_np = np.asarray(self._alpha_noise)
            if an_np.shape[0] == n_valid:
                padded = np.zeros(capacity, dtype=np.float64)
                padded[:n_valid] = an_np
                self._alpha_noise_padded = jnp.array(padded)
            else:
                # Array length already matches capacity, or some other shape
                # we don't pad. Pass through.
                self._alpha_noise_padded = self._alpha_noise
        else:
            self._alpha_noise_padded = self._alpha_noise

        # alpha / V / L padding (post-fit caches). The unused block of these
        # arrays is set to match what ``fit_precompute`` will produce on the
        # padded K with the ``_PAD_ALPHA`` placeholder, so external callers
        # that want to read the padded caches directly get consistent data.
        if alpha_ is not None:
            alpha_padded = np.zeros(capacity, dtype=np.float64)
            alpha_padded[:n_valid] = np.asarray(alpha_)
            self._alpha_padded = jnp.array(alpha_padded)
        else:
            self._alpha_padded = None

        if L_ is not None:
            # Place the unpadded L_ in the valid block; the invalid block on
            # the diagonal gets sqrt(_PAD_ALPHA + alpha_noise) so the full
            # Cholesky factor exactly matches a fresh padded fit.
            alpha_noise_scalar = (
                float(self._alpha_noise)
                if isinstance(self._alpha_noise, (float, int))
                else 0.0  # array-valued noise -- placeholder is approximate
            )
            inv_diag_val = np.sqrt(self._PAD_ALPHA + alpha_noise_scalar)
            L_padded = np.zeros((capacity, capacity), dtype=np.float64)
            L_padded[:n_valid, :n_valid] = np.asarray(L_)
            for i in range(n_valid, capacity):
                L_padded[i, i] = inv_diag_val
            self._L_padded = jnp.array(L_padded)
        else:
            self._L_padded = None

        if V_ is not None:
            # V = L^{-1}. The invalid block of L is diag(inv_diag_val), so
            # its inverse is diag(1/inv_diag_val). Off-diagonal blocks are 0
            # for a block-diagonal L.
            alpha_noise_scalar = (
                float(self._alpha_noise)
                if isinstance(self._alpha_noise, (float, int))
                else 0.0
            )
            inv_diag_val = np.sqrt(self._PAD_ALPHA + alpha_noise_scalar)
            V_padded = np.zeros((capacity, capacity), dtype=np.float64)
            V_padded[:n_valid, :n_valid] = np.asarray(V_)
            inv_v = 1.0 / inv_diag_val if inv_diag_val != 0 else 0.0
            for i in range(n_valid, capacity):
                V_padded[i, i] = inv_v
            self._V_padded = jnp.array(V_padded)
        else:
            self._V_padded = None

    def __deepcopy__(self, memo):
        """Deep copy that shares JIT-compiled functions but copies mutable data.

        JIT-compiled functions are pure functions of kernel type, dimensionality,
        and noise configuration -- they don't depend on training data. Sharing them
        avoids expensive JIT recompilation when the surrogate model is deep-copied
        (e.g. in RankedPool.cache_model).
        """
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        # JIT-compiled functions to share (immutable, keyed on kernel type/dims/noise)
        jit_compiled_attrs = {
            "_kernel_fn",
            "_lml_value_and_grad_fn",
            "_predict_mean_grad_fn",
            "_predict_std_grad_fn",
            "_predict_mean_single_fn",
            "_predict_std_single_fn",
            "_neg_acq_fn",
        }
        # Scalar/config attributes that can be shallow-copied
        scalar_attrs = {
            "_output_scale_sq",
            "_white_noise_level",
            "_n_dims",
            "_has_white_noise",
            "_ready",
            "_cached_acq_solver",  # None on copy; rebuilt lazily
            "_cached_acq_solver_fn_id",
            # Phase B padded-buffer scalar state
            "_capacity",
            "_n_valid",
            "_initial_capacity_override",
            "_appends_since_full_refit",
        }
        # Mutable JAX/numpy data arrays that must be copied
        data_attrs = {
            "_X_train",
            "_y_train",
            "_alpha_noise",
            "_length_scale",
            "_alpha_",
            "_V_",
            "_L_",
            # Phase B padded-buffer arrays (JAX immutable; share-reference is safe)
            "_X_train_padded",
            "_y_train_padded",
            "_alpha_padded",
            "_V_padded",
            "_L_padded",
            "_mask",
            "_alpha_noise_padded",
        }
        for attr in jit_compiled_attrs:
            setattr(new, attr, getattr(self, attr))
        for attr in scalar_attrs:
            setattr(new, attr, getattr(self, attr))
        for attr in data_attrs:
            val = getattr(self, attr)
            if val is not None:
                # JAX arrays are immutable, so a numpy round-trip copy is
                # unnecessary -- just reference the same array. The bundle
                # replaces these wholesale via update_from_gpr(), never mutates
                # them in-place, so sharing is safe.
                setattr(new, attr, val)
            else:
                setattr(new, attr, None)
        handled_attrs = jit_compiled_attrs | scalar_attrs | data_attrs
        for attr, val in self.__dict__.items():
            if attr in handled_attrs:
                continue
            setattr(new, attr, deepcopy(val, memo))
        return new

    @property
    def ready(self):
        return self._ready

    @property
    def ready_for_acquisition_optimization(self):
        """Whether the bundle can run the JAX acquisition optimizer path."""
        return self.ready and self._neg_acq_fn is not None

    def prime_fit_inputs(self, X_train, y_train, alpha_noise):
        """Refresh cached training inputs before hyperparameter optimization."""
        self._X_train = jnp.array(X_train, dtype=jnp.float64)
        self._y_train = jnp.array(y_train, dtype=jnp.float64)
        if isinstance(alpha_noise, (float, int)):
            self._alpha_noise = float(alpha_noise)
        else:
            self._alpha_noise = jnp.array(alpha_noise, dtype=jnp.float64)
        # Padded buffers track the live X_train/y_train. Caches (alpha/V/L)
        # are not available pre-fit so the corresponding padded arrays are
        # left at None until fit_precompute / update_from_gpr populates them.
        if self._n_dims is None and hasattr(X_train, "shape") and len(X_train.shape) == 2:
            self._n_dims = X_train.shape[1]
        self._refresh_padded_state(self._X_train, self._y_train)

    def update_from_gpr(self, gpr):
        """Extract kernel parameters and cached arrays from a fitted GPR.

        Parameters
        ----------
        gpr : GaussianProcessRegressor
            A fitted GPR instance (must have X_train_, y_train_, L_, V_, alpha_).
        """
        if not hasattr(gpr, "X_train_") or gpr.L_ is None:
            self._ready = False
            return

        # Extract kernel parameters
        kernel = gpr.fitted_kernel
        # Handle possible WhiteKernel sum: (C * length_kernel) + WhiteKernel
        self._white_noise_level = 0.0
        if hasattr(kernel, 'k1') and hasattr(kernel.k1, 'k1'):
            # Sum kernel: k1 = C * length-kernel, k2 = WhiteKernel
            product_kernel = kernel.k1
            # k2 is WhiteKernel - its diag contribution is noise_level
            if hasattr(kernel.k2, 'noise_level'):
                self._white_noise_level = float(kernel.k2.noise_level)
        else:
            product_kernel = kernel

        # product_kernel should be ConstantKernel * length-kernel
        output_scale_sq = product_kernel.k1.constant_value
        length_kernel = product_kernel.k2

        # Determine kernel function — only rebuild JIT functions if kernel type
        # or dimensionality changes (avoids expensive JIT recompilation)
        new_has_white_noise = self._white_noise_level > 0.0
        new_n_dims = gpr.X_train_.shape[1]
        need_rebuild = (self._lml_value_and_grad_fn is None
                        or new_has_white_noise != self._has_white_noise
                        or new_n_dims != self._n_dims)

        # Dispatch to the kernel's JAX implementation via the kernel object,
        # not by inspecting its concrete Python type. Kernels without a JAX
        # implementation raise NotImplementedError, which we treat as "fall
        # back to numpy".
        try:
            new_kernel_fn = length_kernel.evaluate_jax_fn()
        except NotImplementedError:
            warnings.warn(
                f"JAX acceleration not supported for kernel {type(length_kernel)}. "
                "Falling back to numpy."
            )
            self._ready = False
            return

        if new_kernel_fn is not self._kernel_fn:
            need_rebuild = True
        self._kernel_fn = new_kernel_fn

        length_scale = np.atleast_1d(length_kernel.length_scale)
        self._has_white_noise = new_has_white_noise
        self._n_dims = new_n_dims

        # Cache as JAX arrays
        self._X_train = jnp.array(gpr.X_train_, dtype=jnp.float64)
        self._y_train = jnp.array(gpr.y_train_, dtype=jnp.float64)
        self._length_scale = jnp.array(length_scale, dtype=jnp.float64)
        self._output_scale_sq = float(output_scale_sq)
        self._alpha_noise = float(np.atleast_1d(gpr.alpha).item()) if np.ndim(gpr.alpha) == 0 else jnp.array(gpr.alpha, dtype=jnp.float64)

        # Cache precomputed arrays from the fitted GPR (use the same L_, V_, alpha_)
        self._L_ = jnp.array(gpr.L_, dtype=jnp.float64)
        self._V_ = jnp.array(gpr.V_, dtype=jnp.float64)
        self._alpha_ = jnp.array(gpr.alpha_, dtype=jnp.float64)

        # Phase B: refresh padded buffers so downstream JIT'd code that consumes
        # them sees a fixed-shape view. Compute paths are still unpadded in B-1.
        self._refresh_padded_state(
            self._X_train, self._y_train,
            alpha_=self._alpha_, V_=self._V_, L_=self._L_,
        )
        # Phase R-B: this is a fresh fit (sklearn or otherwise), so the
        # online-update drift accumulator restarts from zero.
        self._appends_since_full_refit = 0

        # Build gradient functions only when kernel type/dims/noise config changes
        if need_rebuild:
            # Phase B-2: switch the LML JIT to the padded factory. The signature
            # changes from (theta, X, y, alpha_noise) -> (theta, X_padded,
            # y_padded, mask, alpha_noise); the public log_marginal_likelihood
            # / hyperopt callers below feed the padded arrays. JIT compile is
            # keyed on the padded capacity, not on n_valid -- so as long as
            # _capacity is stable, repeated fits reuse the trace.
            lml_fn = _make_lml_padded_fn(
                self._kernel_fn, self._has_white_noise,
                self._n_dims, self._PAD_ALPHA,
            )
            self._lml_value_and_grad_fn = value_and_grad(lml_fn, argnums=0)

            # Phase B-3: switch single-point predict + acq to padded variants.
            # New signatures take (x, X_padded, alpha_padded/V_padded, mask,
            # length_scale, output_scale_sq[, white_noise_level]). Callers
            # below feed the padded arrays. JIT traces are keyed on the
            # padded capacity, not on n_valid -- stable across acq iters.
            predict_mean_single = _make_predict_mean_single_padded_fn(self._kernel_fn)
            predict_std_single = _make_predict_std_single_padded_fn(self._kernel_fn)
            self._predict_mean_single_fn = predict_mean_single
            self._predict_std_single_fn = predict_std_single
            self._predict_mean_grad_fn = jit(grad(predict_mean_single, argnums=0))
            self._predict_std_grad_fn = jit(grad(predict_std_single, argnums=0))
            # Static acq function: compiled once, reused for all zeta/baseline values
            self._neg_acq_fn = _build_neg_acq_padded_fn(predict_mean_single,
                                                         predict_std_single)
            # Invalidate cached solver since the function changed
            self._cached_acq_solver = None
            self._cached_acq_solver_fn_id = None

        self._ready = True

    def _compute_K_trans(self, X_new):
        """Compute cross-covariance K(X_new, X_train) with output scale.

        Phase B-3: uses padded ``_X_train_padded`` and zeros out invalid
        columns (those past ``_n_valid``). The resulting ``K_trans`` has
        shape ``(n_new, capacity)``; downstream code (predict mean / var)
        consumes the padded ``alpha_`` / ``V_`` caches whose invalid
        entries are also zero (mean) or the fixed identity-scaled block
        (V). The dot products on the valid block reproduce the unpadded
        result; the invalid block contributes zero.
        """
        X_new_jax = jnp.array(X_new, dtype=jnp.float64) if not isinstance(X_new, jnp.ndarray) else X_new
        K_trans = self._kernel_fn(X_new_jax, self._X_train_padded,
                                  self._length_scale)
        K_trans = self._output_scale_sq * K_trans
        mask_f = self._mask.astype(K_trans.dtype)
        K_trans = K_trans * mask_f[None, :]
        return K_trans

    def _compute_k_diag(self, X_new):
        """Compute prior variance diagonal (output_scale^2 + white_noise for stationary kernels)."""
        n = X_new.shape[0] if hasattr(X_new, 'shape') else len(X_new)
        return (self._output_scale_sq + self._white_noise_level) * jnp.ones(n)

    def predict_mean_std_jax(self, X_new):
        """Predict GP mean and std at X_new. Returns JAX arrays (no copy).

        Phase B-3: feeds padded caches. ``K_trans`` is (n_new, capacity)
        with invalid columns zeroed; ``alpha_padded`` and ``V_padded`` have
        the corresponding invalid entries / block that contribute zero to
        the dot products on the valid block. JIT trace is keyed on
        (n_new, capacity), not on (n_new, n_valid) -- so capacity-stable
        means no recompile across acq iters even as n_valid grows.
        """
        X_new_jax = (X_new if isinstance(X_new, jnp.ndarray)
                     else jnp.array(X_new, dtype=jnp.float64))
        K_trans = self._compute_K_trans(X_new_jax)
        k_diag = self._compute_k_diag(X_new_jax)
        y_mean, y_var = _gp_predict_mean_and_var(
            K_trans, self._alpha_padded, self._V_padded, k_diag
        )
        return y_mean, jnp.sqrt(y_var)

    def predict_mean_jax(self, X_new):
        """Predict GP mean at X_new. Returns JAX array (no copy)."""
        X_new_jax = (X_new if isinstance(X_new, jnp.ndarray)
                     else jnp.array(X_new, dtype=jnp.float64))
        K_trans = self._compute_K_trans(X_new_jax)
        return _gp_predict_mean(K_trans, self._alpha_padded)

    def predict_std_jax(self, X_new):
        """Predict GP std at X_new. Returns JAX array (no copy)."""
        X_new_jax = (X_new if isinstance(X_new, jnp.ndarray)
                     else jnp.array(X_new, dtype=jnp.float64))
        K_trans = self._compute_K_trans(X_new_jax)
        k_diag = self._compute_k_diag(X_new_jax)
        y_var = _gp_predict_var(K_trans, self._V_padded, k_diag)
        return jnp.sqrt(y_var)

    def predict_mean(self, X_new):
        """Predict GP mean at X_new. Returns numpy array."""
        return np.asarray(self.predict_mean_jax(X_new))

    def predict_std(self, X_new):
        """Predict GP std at X_new. Returns numpy array."""
        return np.asarray(self.predict_std_jax(X_new))

    def predict_mean_std(self, X_new):
        """Predict GP mean and std at X_new. Returns numpy arrays."""
        y_mean, y_std = self.predict_mean_std_jax(X_new)
        return np.asarray(y_mean), np.asarray(y_std)

    def log_marginal_likelihood(self, length_scale, output_scale_sq, alpha_noise):
        """Compute LML for given hyperparameters. For benchmarking/testing."""
        K = self._kernel_fn(self._X_train, self._X_train,
                            jnp.array(length_scale, dtype=jnp.float64))
        K = output_scale_sq * K
        n = self._X_train.shape[0]
        K = K + alpha_noise * jnp.eye(n)
        L = jnp.linalg.cholesky(K)
        alpha_vec = cho_solve((L, True), self._y_train)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        lml = (-0.5 * jnp.dot(self._y_train, alpha_vec)
               - 0.5 * log_det
               - 0.5 * n * jnp.log(2.0 * jnp.pi))
        return float(lml)

    def log_marginal_likelihood_with_grad(self, theta):
        """Compute LML and its gradient w.r.t. theta using JAX auto-diff.

        Parameters
        ----------
        theta : array-like, shape (n_kernel_params,)
            Log-space kernel hyperparameters:
            [log(constant_value), log(length_scale_1), ..., log(length_scale_d)]
            and optionally log(white_noise_level) at the end.

        Returns
        -------
        lml : float
            Log marginal likelihood.
        grad : numpy array, shape (n_kernel_params,)
            Gradient of LML w.r.t. theta.
        """
        theta_jax = jnp.array(theta, dtype=jnp.float64)
        # Phase B-2: feed the padded arrays + mask so the JIT'd LML kernel is
        # shape-stable across acquisition iterations. Scalar noise stays as
        # a scalar (broadcasts onto the (capacity, capacity) diag); per-point
        # noise was padded to length `capacity` in _refresh_padded_state.
        alpha_noise = (
            jnp.float64(self._alpha_noise_padded)
            if isinstance(self._alpha_noise_padded, (float, int))
            else self._alpha_noise_padded
        )

        lml, grad_theta = self._lml_value_and_grad_fn(
            theta_jax,
            self._X_train_padded, self._y_train_padded, self._mask,
            alpha_noise,
        )
        return float(lml), np.asarray(grad_theta)

    def predict_mean_grad(self, x):
        """Compute gradient of GP mean prediction w.r.t. input x.

        Parameters
        ----------
        x : array-like, shape (n_features,)
            Single query point.

        Returns
        -------
        grad_mean : numpy array, shape (n_features,)
            Gradient of mean prediction w.r.t. x.
        """
        x_jax = jnp.array(x, dtype=jnp.float64)
        grad_mean = self._predict_mean_grad_fn(
            x_jax, self._X_train_padded, self._alpha_padded, self._mask,
            self._length_scale, jnp.float64(self._output_scale_sq)
        )
        return np.asarray(grad_mean)

    def predict_std_grad(self, x):
        """Compute gradient of GP std prediction w.r.t. input x.

        Parameters
        ----------
        x : array-like, shape (n_features,)
            Single query point.

        Returns
        -------
        grad_std : numpy array, shape (n_features,)
            Gradient of std prediction w.r.t. x.
        """
        x_jax = jnp.array(x, dtype=jnp.float64)
        grad_std = self._predict_std_grad_fn(
            x_jax, self._X_train_padded, self._V_padded, self._mask,
            self._length_scale, jnp.float64(self._output_scale_sq),
            jnp.float64(self._white_noise_level)
        )
        return np.asarray(grad_std)

    def predict_mean_single_jax(self, x):
        """Predict GP mean for a single point. Returns JAX scalar.

        Parameters
        ----------
        x : array-like, shape (n_features,)

        Returns
        -------
        y_mean : jax.Array, scalar
        """
        x_jax = jnp.array(x, dtype=jnp.float64) if not isinstance(x, jnp.ndarray) else x
        return self._predict_mean_single_fn(
            x_jax, self._X_train_padded, self._alpha_padded, self._mask,
            self._length_scale, jnp.float64(self._output_scale_sq)
        )

    def predict_with_grads(self, x):
        """Compute mean, std, mean_grad, std_grad for a single point.

        Parameters
        ----------
        x : array-like, shape (n_features,)
            Single query point.

        Returns
        -------
        mean : float
        std : float
        mean_grad : numpy array, shape (n_features,)
        std_grad : numpy array, shape (n_features,)
        """
        x_jax = jnp.array(x, dtype=jnp.float64)
        osc = jnp.float64(self._output_scale_sq)
        wnl = jnp.float64(self._white_noise_level)

        mean = float(self._predict_mean_single_fn(
            x_jax, self._X_train_padded, self._alpha_padded, self._mask,
            self._length_scale, osc))
        std = float(self._predict_std_single_fn(
            x_jax, self._X_train_padded, self._V_padded, self._mask,
            self._length_scale, osc, wnl))
        mean_grad = np.asarray(self._predict_mean_grad_fn(
            x_jax, self._X_train_padded, self._alpha_padded, self._mask,
            self._length_scale, osc))
        std_grad = np.asarray(self._predict_std_grad_fn(
            x_jax, self._X_train_padded, self._V_padded, self._mask,
            self._length_scale, osc, wnl))
        return mean, std, mean_grad, std_grad

    # ------------------------------------------------------------------
    # Pure JAX GP fit (Item 4)
    # ------------------------------------------------------------------

    def fit_precompute(self):
        """Compute L, V, alpha_ from cached training data and kernel params.

        Uses JAX JIT-compiled Cholesky decomposition instead of scipy.
        Must be called after kernel parameters are updated (via update_params).

        Returns
        -------
        L : jax.Array, shape (n, n)
        V : jax.Array, shape (n, n)
        alpha_ : jax.Array, shape (n,)
        """
        # Phase B-5: compute the Cholesky on the *padded* kernel matrix so the
        # JIT trace is shape-keyed on ``_capacity`` (stable across acq iters)
        # rather than on ``n_valid`` (which changes every KB step). This is the
        # last shape-keyed function in the hot loop; padding it unlocks
        # re-enabling JAX in the KB-conditioning loop (the original Option 1
        # workaround was a recompile-tax mitigation that's now obsolete).
        #
        # Ensure padded buffers exist (they may not yet if this is the very
        # first fit_precompute on a freshly primed mixin).
        if (self._mask is None
                or self._X_train_padded is None
                or self._X_train_padded.shape[0] < self._n_valid):
            self._refresh_padded_state(self._X_train, self._y_train)

        K_padded = self._kernel_fn(
            self._X_train_padded, self._X_train_padded, self._length_scale)
        K_padded = self._output_scale_sq * K_padded
        if self._white_noise_level > 0:
            K_padded = K_padded + self._white_noise_level * jnp.eye(
                self._capacity)
        alpha_noise = (jnp.float64(self._alpha_noise_padded) if isinstance(
            self._alpha_noise_padded, (float, int)) else self._alpha_noise_padded)
        L_padded, V_padded, alpha_padded = _gp_fit_precompute_padded(
            K_padded, self._y_train_padded, self._mask,
            alpha_noise, self._PAD_ALPHA,
        )
        # Cache padded versions directly (these are what the padded predict /
        # LML / acq paths consume) and slice out the valid blocks for the
        # legacy unpadded attributes (still read by some non-hot paths and by
        # external code that grabs gpr.L_ / gpr.V_ / gpr.alpha_).
        n = self._n_valid
        self._L_padded = L_padded
        self._V_padded = V_padded
        self._alpha_padded = alpha_padded
        self._L_ = L_padded[:n, :n]
        self._V_ = V_padded[:n, :n]
        self._alpha_ = alpha_padded[:n]
        # Reset refresh counter on a full fit -- the L/V/alpha are exact and
        # subsequent online appends can drift relative to this anchor.
        self._appends_since_full_refit = 0
        return self._L_, self._V_, self._alpha_

    # ------------------------------------------------------------------
    # Online block-append update (Phase R-B)
    #
    # When the KB-conditioning loop adds m new points to a deepcopied
    # surrogate (fit_hyperparameters=False), recomputing the full Cholesky
    # is O((n+m)^3).  The block-append update below is O((n+m)^2) and exact
    # in floating-point arithmetic (drift bounded at ~1e-14 over 100
    # sequential appends in the microtest).  A periodic full-refit
    # (governed by ``_REFRESH_EVERY``) is run as a safety net.
    # ------------------------------------------------------------------
    _REFRESH_EVERY = 50  # full re-Cholesky every N online appends (defensive)

    def append_points_jax(self, X_new, y_new, force_full_refit=False):
        """Online block-append update for L, V, alpha.

        Updates the padded buffers in place so subsequent predict / acq
        calls see the augmented training set.  Equivalent to (but
        ``O(n^2)`` per append point instead of ``O(n^3)`` for):

            self._X_train = vstack(self._X_train, X_new)
            self._y_train = concat (self._y_train, y_new)
            self.fit_precompute()

        Parameters
        ----------
        X_new : array-like, shape (m, d)
            New input points (in the same space as ``self._X_train``).
        y_new : array-like, shape (m,)
            Targets for the new points.
        force_full_refit : bool
            If True, run a full ``fit_precompute`` instead of an online
            update.  Used by the refresh-every-N safety net and by callers
            who know drift may have accumulated.

        Returns
        -------
        L_valid, V_valid, alpha_valid : jax.Array
            The new ``(n+m, n+m)`` blocks (sliced from the padded buffers).
        """
        X_new_np = np.atleast_2d(np.asarray(X_new))
        y_new_np = np.atleast_1d(np.asarray(y_new, dtype=np.float64))
        m = X_new_np.shape[0]
        if m == 0:
            return self._L_, self._V_, self._alpha_
        if y_new_np.shape[0] != m:
            raise ValueError(
                f"X_new shape {X_new_np.shape} and y_new shape "
                f"{y_new_np.shape} disagree on the leading dimension."
            )

        n_before = self._n_valid
        n_after = n_before + m

        # Capacity check: if appending overflows the padded buffer (or we
        # don't have padded state at all yet), do a full refit on the
        # combined training set.  This is the same fallback the periodic
        # refresh uses.
        need_full = (
            force_full_refit
            or self._L_padded is None
            or n_after > self._capacity
            or self._appends_since_full_refit + 1 > self._REFRESH_EVERY
        )

        # Build the augmented training arrays in numpy first.  Always do
        # this so the unpadded ``_X_train`` / ``_y_train`` stay in sync.
        X_old_np = np.asarray(self._X_train)
        y_old_np = np.asarray(self._y_train)
        X_full_np = np.vstack([X_old_np, X_new_np])
        y_full_np = np.concatenate([y_old_np, y_new_np])
        self._X_train = jnp.array(X_full_np, dtype=jnp.float64)
        self._y_train = jnp.array(y_full_np, dtype=jnp.float64)

        if need_full:
            # Refresh padded state + full Cholesky on the new combined data.
            self._refresh_padded_state(self._X_train, self._y_train)
            self.fit_precompute()  # resets _appends_since_full_refit
            self._sync_gpr_attrs()
            return self._L_, self._V_, self._alpha_

        # ---- Online block-append path ----
        # Compute the kernel blocks A (n_old, m) and B (m, m) using the
        # *unpadded* valid arrays.  These shapes vary with n_before and m,
        # so the underlying JIT may retrace; that's tolerated because the
        # work per append is dominated by O(n^2) algebra, not trace
        # overhead.
        ls = self._length_scale
        osc = jnp.float64(self._output_scale_sq)
        X_old_j = jnp.array(X_old_np, dtype=jnp.float64)
        X_new_j = jnp.array(X_new_np, dtype=jnp.float64)
        A = osc * self._kernel_fn(X_old_j, X_new_j, ls)            # (n, m)
        B = osc * self._kernel_fn(X_new_j, X_new_j, ls)            # (m, m)
        # Noise on the new diagonal.  Always scalar in the current GPry
        # config; the array case is handled by falling through to
        # need_full above.
        if isinstance(self._alpha_noise, (float, int)):
            noise = float(self._alpha_noise)
        else:
            # Per-point noise: only safe with full refit (we'd need to pad
            # the noise vector too).  Force the safe path.
            self._refresh_padded_state(self._X_train, self._y_train)
            self.fit_precompute()
            self._sync_gpr_attrs()
            return self._L_, self._V_, self._alpha_
        if self._white_noise_level > 0:
            noise = noise + float(self._white_noise_level)
        B = B + noise * jnp.eye(m)
        # Pull the current valid blocks out of the padded buffers
        L_valid = self._L_padded[:n_before, :n_before]
        V_valid = self._V_padded[:n_before, :n_before]
        y_old_j = jnp.array(y_old_np, dtype=jnp.float64)
        y_new_j = jnp.array(y_new_np, dtype=jnp.float64)

        L_new, V_new, alpha_new = _gp_append_block_update(
            L_valid, V_valid, A, B, y_old_j, y_new_j,
        )
        # Write back into the padded buffers (rebuild to keep the
        # block-diagonal "inv_diag on invalid" invariant; cheap relative
        # to a full Cholesky).
        self._refresh_padded_state(
            self._X_train, self._y_train,
            alpha_=alpha_new, V_=V_new, L_=L_new,
        )
        # _refresh_padded_state populates the _padded attributes; mirror the
        # unpadded slices to match fit_precompute's contract.
        self._L_ = L_new
        self._V_ = V_new
        self._alpha_ = alpha_new
        self._appends_since_full_refit += 1
        self._sync_gpr_attrs()
        return L_new, V_new, alpha_new

    def _sync_gpr_attrs(self):
        """Mirror the JAX-side training state into the GPR's sklearn-style
        numpy attributes (``X_train_`` / ``y_train_`` / ``L_`` / ``V_`` /
        ``alpha_``) so that ``predict`` on the *numpy* path (when
        ``_native_acceleration_enabled`` is False, the KB case) sees the
        same data the online append just produced.

        No-op on the mixin alone; meaningful only when this mixin is
        composed into a concrete GPR class.
        """
        if not hasattr(self, "X_train_"):
            return
        try:
            self.X_train_ = np.asarray(self._X_train)
            self.y_train_ = np.asarray(self._y_train)
            if self._L_ is not None:
                self.L_ = np.asarray(self._L_)
            if self._V_ is not None:
                self.V_ = np.asarray(self._V_)
            if self._alpha_ is not None:
                self.alpha_ = np.asarray(self._alpha_)
        except AttributeError:
            # No GPR attrs to sync (e.g. used as a pure mixin in a test).
            pass

    def update_params(self, length_scale, output_scale_sq, white_noise_level,
                      alpha_noise, X_train=None, y_train=None):
        """Update kernel parameters (and optionally training data) from numpy.

        Parameters
        ----------
        length_scale : array-like
        output_scale_sq : float
        white_noise_level : float
        alpha_noise : float or array-like
        X_train, y_train : array-like, optional
        """
        self._length_scale = jnp.array(
            np.atleast_1d(length_scale), dtype=jnp.float64)
        self._output_scale_sq = float(output_scale_sq)
        self._white_noise_level = float(white_noise_level)
        if isinstance(alpha_noise, (float, int)):
            self._alpha_noise = float(alpha_noise)
        else:
            self._alpha_noise = jnp.array(alpha_noise, dtype=jnp.float64)
        if X_train is not None:
            self._X_train = jnp.array(X_train, dtype=jnp.float64)
            if self._n_dims is None and len(self._X_train.shape) == 2:
                self._n_dims = self._X_train.shape[1]
        if y_train is not None:
            self._y_train = jnp.array(y_train, dtype=jnp.float64)
        # If training data was touched, padded buffers need to follow. Caches
        # (alpha/V/L) are not refreshed here -- caller must call
        # fit_precompute next, which re-pads with the new caches.
        if X_train is not None or y_train is not None:
            if self._X_train is not None and self._y_train is not None:
                self._refresh_padded_state(self._X_train, self._y_train)

    # ------------------------------------------------------------------
    # JAX hyperparameter optimization (Item 1)
    # ------------------------------------------------------------------

    def optimize_hyperparameters(
        self, bounds, theta_candidates=None, n_restarts=None, theta_initial=None, rng=None
    ):
        """Optimize kernel hyperparameters using jaxopt.LBFGSB.

        Uses the LML function from update_from_gpr (compiled once per kernel
        type change) with training data passed as extra args, so no
        recompilation occurs between calls.

        Parameters
        ----------
        bounds : array, shape (n_params, 2)
            Bounds on log-space hyperparameters.
        theta_candidates : sequence of array-like, optional
            Explicit shortlist of starting points to refine. Preferred over random
            restart generation when provided.
        n_restarts : int, optional
            Number of optimizer restarts when ``theta_candidates`` is not provided.
        theta_initial : array-like, optional
            Legacy starting point for first restart when ``theta_candidates`` is not
            provided.
        rng : numpy.random.Generator, optional
            For sampling random starting points.

        Returns
        -------
        theta_opt : numpy array
            Best hyperparameters found.
        neg_lml_min : float
            Negative log marginal likelihood at optimum.
        """
        import jaxopt
        import logging

        bounds_jax = (
            jnp.array(bounds[:, 0], dtype=jnp.float64),
            jnp.array(bounds[:, 1], dtype=jnp.float64),
        )

        # Use the already-compiled LML function, passing data as args
        # to avoid creating a new closure/recompilation each call. Since
        # Phase B-2 the LML is the padded variant: it takes the padded
        # buffers + mask, and is shape-stable across acquisition iters as
        # long as ``_capacity`` does not change.
        lml_fn = self._lml_value_and_grad_fn  # from update_from_gpr

        @jit
        def neg_lml(theta, X_padded, y_padded, mask, alpha_noise):
            lml_val = lml_fn(theta, X_padded, y_padded, mask, alpha_noise)[0]
            return -lml_val

        # Build/cache solver keyed on the neg_lml function identity
        fn_id = id(lml_fn)
        if (not hasattr(self, '_cached_hyper_solver')
                or self._cached_hyper_solver is None
                or self._cached_hyper_solver_fn_id != fn_id):
            for name in ("jaxopt", "jaxopt._src", "absl"):
                logging.getLogger(name).setLevel(logging.ERROR)
            # ``implicit_diff=False``: jaxopt's default ``True`` wraps the inner
            # solver in a ``custom_root`` decorator at every call to ``run()``,
            # which has a fresh Python identity and forces JAX to recompile the
            # outer ``while`` loop on every call. We don't differentiate through
            # the hyperopt solver (its outputs feed Cholesky / predict, not a
            # downstream gradient through the optimizer), so this is safe and
            # cuts per-call wall time by ~75x in isolated micro-benchmark.
            self._cached_hyper_solver = jaxopt.LBFGSB(
                fun=neg_lml, maxiter=100, tol=1e-6, implicit_diff=False)
            self._cached_hyper_solver_fn_id = fn_id

        solver = self._cached_hyper_solver
        # Pad arrays + mask are what the LML JIT now consumes.
        X_padded = self._X_train_padded
        y_padded = self._y_train_padded
        mask = self._mask
        alpha_noise = (jnp.float64(self._alpha_noise_padded) if isinstance(
            self._alpha_noise_padded, (float, int)) else self._alpha_noise_padded)
        data_args = (X_padded, y_padded, mask, alpha_noise)

        best_theta = None
        best_val = np.inf

        if rng is None:
            rng = np.random.default_rng()

        if theta_candidates is not None:
            start_points = [
                jnp.array(theta0, dtype=jnp.float64)
                for theta0 in theta_candidates
            ]
        else:
            if n_restarts is None:
                raise ValueError(
                    "Pass either theta_candidates or n_restarts for JAX hyperopt."
                )
            start_points = []
            for i in range(n_restarts):
                if i == 0 and theta_initial is not None:
                    theta0 = jnp.array(theta_initial, dtype=jnp.float64)
                else:
                    theta0 = jnp.array(
                        rng.uniform(bounds[:, 0], bounds[:, 1]),
                        dtype=jnp.float64,
                    )
                    try:
                        val = float(neg_lml(theta0, *data_args))
                        if not np.isfinite(val):
                            continue
                    except Exception as excpt:
                        warnings.warn(
                            "JAX hyperopt start-point screening failed; "
                            f"skipping candidate: {excpt!r}"
                        )
                        continue
                start_points.append(theta0)

        for theta0 in start_points:

            try:
                result = solver.run(theta0, *data_args, bounds=bounds_jax)
                val = float(result.state.value)
                if val < best_val:
                    best_val = val
                    best_theta = np.asarray(result.params)
            except Exception as excpt:
                warnings.warn(
                    "JAX hyperopt restart failed; skipping this start point: "
                    f"{excpt!r}"
                )
                continue

        if best_theta is None:
            raise RuntimeError("All JAX hyperopt restarts failed")
        return best_theta, best_val

    # ------------------------------------------------------------------
    # JAX acquisition function optimization (Item 2)
    # ------------------------------------------------------------------

    def optimize_acq(self, x0, bounds, zeta, noise_var, baseline):
        """Optimize the LogExp acquisition function using jaxopt.LBFGSB.

        Uses a static JIT-compiled acquisition function that takes
        zeta/noise_var/baseline as explicit JAX arguments, so the compiled
        trace is reused across calls (no recompilation). The jaxopt solver
        is also cached and reused.

        Parameters
        ----------
        x0 : array-like, shape (n_dims,)
            Starting point (in transformed space).
        bounds : array, shape (n_dims, 2)
            Bounds in transformed space.
        zeta : float
            Exploration-exploitation parameter.
        noise_var : float
            Noise standard deviation.
        baseline : float
            Reference y value (y_max in transformed space).

        Returns
        -------
        x_opt : numpy array, shape (n_dims,)
            Optimal point.
        func_min : float
            Negative acquisition value at optimum.
        """
        import jaxopt
        import logging

        # Cache the solver: only rebuild when the underlying JIT function changes
        # (i.e., when kernel type/dims/noise config changes via need_rebuild)
        fn_id = id(self._neg_acq_fn)
        if (self._cached_acq_solver is None
                or self._cached_acq_solver_fn_id != fn_id):
            for name in ("jaxopt", "jaxopt._src", "absl"):
                logging.getLogger(name).setLevel(logging.ERROR)
            # ``implicit_diff=False`` to avoid jaxopt's per-call ``custom_root``
            # wrapper that defeats JAX's cache (~75x penalty in micro-benchmark).
            # The acquisition optimizer's output is consumed as a *point*, not
            # differentiated through, so no functionality is lost.
            self._cached_acq_solver = jaxopt.LBFGSB(
                fun=self._neg_acq_fn, maxiter=50, tol=1e-5, implicit_diff=False)
            self._cached_acq_solver_fn_id = fn_id

        x0_jax = jnp.array(x0, dtype=jnp.float64)
        bounds_jax = (
            jnp.array(bounds[:, 0], dtype=jnp.float64),
            jnp.array(bounds[:, 1], dtype=jnp.float64),
        )
        # Phase B-3: feed padded buffers + mask. Shape-stable across acq
        # iterations as long as ``_capacity`` is stable.
        acq_args = (
            self._X_train_padded, self._alpha_padded, self._V_padded,
            self._mask,
            self._length_scale,
            jnp.float64(self._output_scale_sq),
            jnp.float64(self._white_noise_level),
            jnp.float64(zeta), jnp.float64(noise_var), jnp.float64(baseline),
        )

        try:
            result = self._cached_acq_solver.run(
                x0_jax, *acq_args, bounds=bounds_jax)
            x_opt = np.asarray(result.params)
            func_min = float(result.state.value)
        except Exception as excpt:
            warnings.warn(
                "JAX acquisition optimization failed; returning the "
                f"un-optimized start point x0 (acquisition not improved): {excpt!r}"
            )
            x_opt = np.asarray(x0)
            func_min = float(self._neg_acq_fn(x0_jax, *acq_args))

        return x_opt, func_min

    def build_surrogate_loglikelihood_builder(
        self, preprocessing_y, clip_factor, y_clip_min, y_clip_max
    ):
        """Build a transformed-space JAX log-likelihood callback for BlackJAX."""

        def _build_jax_loglikelihood(param_names_list):
            def _loglikelihood_fn(params):
                x = jnp.array(
                    [params[name] for name in param_names_list],
                    dtype=jnp.float64,
                )
                y_ = self.predict_mean_single_jax(x)
                y = preprocessing_y.inverse_transform_jax(y_)
                if clip_factor is not None:
                    upper = clip_factor * y_clip_max - (clip_factor - 1) * y_clip_min
                    y = jnp.clip(y, None, upper)
                return y

            return _loglikelihood_fn

        return _build_jax_loglikelihood
