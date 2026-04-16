"""
Experimental Learned Feature GP (LFGP) backend.

This module is not part of the conservative GPry baseline. It is kept for
opt-in tests of learned neural feature maps with Bayesian linear regression.

This module implements a Gaussian Process approximation that replaces the random
Fourier features of RandomFeatureGP with a small learned MLP feature map, while
keeping the Bayesian linear regression (BLR) last layer for exact uncertainty.

Architecture:

    phi(x) = NN(x; theta)   # shape: (d,) -> (M,)

where NN is a 2-layer MLP:

    x -> Linear(d, H) -> tanh -> Linear(H, M)

The BLR on top is identical to RFGP:

    A        = Phi^T Phi + lam * I_M
    w        = A^{-1} Phi^T y
    mean(x*) = phi(x*)^T w
    var(x*)  = sigma_f^2 * phi(x*)^T A^{-1} phi(x*)

Training procedure (two-phase at each fit call):

1. Phase 1 -- Learn features: optimise NN weights theta and kernel
   hyperparameters (sigma_f, length_scales) by maximising the BLR log marginal
   likelihood using Adam.  Fully differentiable through JAX.

2. Phase 2 -- BLR fit: given the learned features, compute the closed-form BLR
   solution (same as RFGP).

Interface
---------
Matches :class:`experimental_random_feature_gp.RandomFeatureGP` and therefore also
:class:`gpr.GaussianProcessRegressor`.  All public attributes and methods used
by :class:`surrogate.SurrogateModel` are present.

References
----------
Ober, S. W. & Rasmussen, C. E. (2021). Benchmarking the neural linear model for
regression.  arXiv 2106.01386.

Wilson, A. G. et al. (2016). Deep kernel learning.  AISTATS.
"""

# Builtin
import warnings
from typing import Mapping

# External
import numpy as np

# JAX (required)
import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
import jax.random as jr

# Enable 64-bit for numerical accuracy (critical for log marginal likelihood)
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Stub kernel object (same pattern as RandomFeatureGP)
# ---------------------------------------------------------------------------

class _LFGPKernelStub:
    """
    Minimal stub satisfying the ``kernel`` / ``kernel_`` attribute contract.

    Provides:
      - ``requires_vector_input`` (bool) for input validation dispatch
      - ``str(kernel)`` for logging
      - ``hyperparameters`` (empty list) iterated in SurrogateModel.__str__
    """

    def __init__(self, output_scale, length_scales):
        self.output_scale = float(output_scale)
        self.length_scales = np.atleast_1d(np.asarray(length_scales, dtype=float))
        self.requires_vector_input = True
        self.hyperparameters = []

    def __str__(self):
        ls = ", ".join(f"{v:.4g}" for v in self.length_scales)
        return (
            f"LFGP-RBF(output_scale={self.output_scale:.4g}, "
            f"length_scales=[{ls}])"
        )

    def __repr__(self):
        return self.__str__()


# ---------------------------------------------------------------------------
# Pure-JAX MLP utilities
# ---------------------------------------------------------------------------

def _init_mlp_params(d, H, M, length_scale_prior_mean, rng_key):
    """
    Initialise a 2-layer MLP: Linear(d->H) -> tanh -> Linear(H->M).

    Parameters
    ----------
    d   : int - input dimension
    H   : int - hidden layer width
    M   : int - output features (number of learned features)
    length_scale_prior_mean : (d,) array - geometric-mean length scales;
        used to scale the first-layer weights so the network has roughly
        unit sensitivity at initialisation.
    rng_key : JAX PRNGKey

    Returns
    -------
    params : dict with keys 'W1', 'b1', 'W2', 'b2' as float64 JAX arrays
    """
    k1, k2, k3, k4 = jr.split(rng_key, 4)
    ls_arr = jnp.array(length_scale_prior_mean, dtype=jnp.float64)

    # Xavier (Glorot) uniform initialisation
    limit1 = jnp.sqrt(6.0 / (d + H))
    W1_raw = jr.uniform(k1, shape=(H, d), minval=-limit1, maxval=limit1,
                        dtype=jnp.float64)
    # Scale first-layer rows by 1/length_scale so the network sees inputs of
    # order 1 when x has typical spread equal to the length scale.
    W1 = W1_raw / ls_arr[None, :]

    b1 = jnp.zeros((H,), dtype=jnp.float64)

    limit2 = jnp.sqrt(6.0 / (H + M))
    W2 = jr.uniform(k2, shape=(M, H), minval=-limit2, maxval=limit2,
                    dtype=jnp.float64)
    b2 = jnp.zeros((M,), dtype=jnp.float64)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def _mlp_features(params, X):
    """
    Evaluate the 2-layer MLP feature map.

    phi(X) = X @ W1^T + b1  -> tanh -> @ W2^T + b2
    Then normalise rows to unit norm (prevents scale issues with BLR).

    Parameters
    ----------
    params : dict with 'W1'(H,d), 'b1'(H,), 'W2'(M,H), 'b2'(M,)
    X : (N, d) float64

    Returns
    -------
    Phi : (N, M) float64  -- normalised features
    """
    h = jnp.tanh(X @ params["W1"].T + params["b1"])  # (N, H)
    out = h @ params["W2"].T + params["b2"]           # (N, M)

    # Row-wise normalisation: divide each feature vector by its L2 norm.
    # Clip norm from below to avoid division by zero.
    norms = jnp.linalg.norm(out, axis=1, keepdims=True)  # (N, 1)
    norms = jnp.maximum(norms, 1e-8)
    return out / norms


# ---------------------------------------------------------------------------
# BLR core -- identical to RandomFeatureGP (copied to avoid import coupling)
# ---------------------------------------------------------------------------

@jit
def _blr_fit_jax(Phi, y, lam):
    """
    BLR normal equations: A = Phi^T Phi + lam I,  w = A^{-1} Phi^T y.

    Returns (L_A, w) where L_A is the lower Cholesky of A.
    """
    M = Phi.shape[1]
    A = Phi.T @ Phi + lam * jnp.eye(M)
    L_A = jnp.linalg.cholesky(A)
    rhs = Phi.T @ y
    w = jax.scipy.linalg.cho_solve((L_A, True), rhs)
    return L_A, w


@jit
def _predict_var_batch(Phi_star, L_A, sigma_f_sq):
    """var(x_i) = sigma_f^2 * ||L_A^{-1} phi(x_i)||^2 for each row of Phi_star."""
    V = jax.scipy.linalg.solve_triangular(L_A, Phi_star.T, lower=True)  # (M, N)
    return sigma_f_sq * jnp.sum(V ** 2, axis=0)                         # (N,)


@jit
def _predict_var_single(phi_x, L_A, sigma_f_sq):
    """var(x) = sigma_f^2 * ||L_A^{-1} phi(x)||^2 for a single point."""
    v = jax.scipy.linalg.solve_triangular(L_A, phi_x, lower=True)  # (M,)
    return sigma_f_sq * jnp.dot(v, v)


# ---------------------------------------------------------------------------
# BLR log marginal likelihood with MLP features
# ---------------------------------------------------------------------------

def _lml_lfgp(log_params, mlp_params, X_j, y_j, noise_sq):
    """
    BLR log marginal likelihood with MLP-computed features.  Fully
    differentiable through JAX (both log_params and mlp_params).

    Model:  y ~ N(0, sigma_f^2 Phi Phi^T + sigma_n^2 I_N)

    Uses the same Sylvester / Woodbury reduction as RandomFeatureGP:

        lam = sigma_n^2 / sigma_f^2
        A   = Phi^T Phi + lam I_M

        log-det = N log(sigma_f^2) + (N-M) log(lam) + 2 sum_k log(L_A[k,k])
        quad    = (||y||^2 - ||L_A^{-1} Phi^T y||^2) / sigma_n^2
        LML     = -0.5 * (quad + log-det + N log(2 pi))

    Parameters
    ----------
    log_params  : (1,) JAX array -- [log_sigma_f]  (length scales folded into NN)
    mlp_params  : pytree -- MLP weights {'W1', 'b1', 'W2', 'b2'}
    X_j         : (N, d) JAX float64
    y_j         : (N,) JAX float64
    noise_sq    : float -- sigma_n^2 (fixed)

    Returns
    -------
    lml : scalar float64
    """
    log_sigma_f = log_params[0]
    sigma_f_sq = jnp.exp(2.0 * log_sigma_f)

    Phi = _mlp_features(mlp_params, X_j)  # (N, M)

    M = Phi.shape[1]
    N = X_j.shape[0]

    lam = noise_sq / sigma_f_sq  # dimensionless regulariser

    # M-dim Cholesky
    A = Phi.T @ Phi + lam * jnp.eye(M)
    L_A = jnp.linalg.cholesky(A)

    # Woodbury quadratic
    Phi_T_y = Phi.T @ y_j                                          # (M,)
    v = jax.scipy.linalg.solve_triangular(L_A, Phi_T_y, lower=True)  # (M,)
    quadratic = (jnp.dot(y_j, y_j) - jnp.dot(v, v)) / noise_sq

    # Log-determinant
    log_det_A = 2.0 * jnp.sum(jnp.log(jnp.diag(L_A)))
    log_det = N * jnp.log(sigma_f_sq) + (N - M) * jnp.log(lam) + log_det_A

    lml = -0.5 * (quadratic + log_det + N * jnp.log(2.0 * jnp.pi))
    return lml


# ---------------------------------------------------------------------------
# Adam optimiser (pure JAX, no optax dependency)
# ---------------------------------------------------------------------------

def _adam_init(params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """Initialise Adam optimiser state for a pytree of parameters."""
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"params": params, "m": m, "v": v, "t": 0,
            "lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps}


def _adam_step(state, grads, weight_decay=0.0):
    """
    Single Adam update step.

    Parameters
    ----------
    state  : dict returned by _adam_init (or previous _adam_step)
    grads  : pytree with same structure as state['params']
    weight_decay : float  -- L2 regularisation coefficient applied to params

    Returns
    -------
    new_state : dict
    """
    t = state["t"] + 1
    lr = state["lr"]
    b1, b2, eps = state["beta1"], state["beta2"], state["eps"]

    # Bias-corrected learning rate
    lr_t = lr * jnp.sqrt(1.0 - b2 ** t) / (1.0 - b1 ** t)

    # tree_map applies function leaf-wise; each call produces one output pytree
    new_m = jax.tree_util.tree_map(
        lambda m, g, p: b1 * m + (1.0 - b1) * (g + weight_decay * p),
        state["m"], grads, state["params"]
    )
    new_v = jax.tree_util.tree_map(
        lambda v, g, p: b2 * v + (1.0 - b2) * (g + weight_decay * p) ** 2,
        state["v"], grads, state["params"]
    )
    new_p = jax.tree_util.tree_map(
        lambda p, m, v: p - lr_t * m / (jnp.sqrt(v) + eps),
        state["params"], new_m, new_v
    )
    return {**state, "params": new_p, "m": new_m, "v": new_v, "t": t}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LearnedFeatureGP:
    r"""
    Learned Feature Gaussian Process -- a scalable GP backend with a learned
    neural feature map.

    Replaces the random Fourier features of :class:`RandomFeatureGP` with a
    small MLP, trained end-to-end by maximising the BLR log marginal likelihood
    via Adam.  The BLR last layer retains closed-form Bayesian uncertainty.

    Parameters
    ----------
    kernel : str, optional (default: "RBF")
        Kernel type.  Only "RBF" is currently supported.  Kept for API
        compatibility.

    output_scale_prior : list [min, max], optional (default: [1e-2, 1e3])
        Bounds for the output scale sigma_f.

    length_scale_prior : array-like, shape (d, 2), optional
        Per-dimension bounds for length scales.  The number of rows encodes
        the input dimensionality d.  Defaults to [[1e-3, 1e1]] (1-d).

    noise_level : float, optional (default: 1e-2)
        Initial noise standard deviation sigma_n.

    noise_fixed : bool, optional (default: True)
        If True, sigma_n is not optimised.

    n_features : int or None, optional (default: None)
        Number of output features M.  If None, defaults to
        ``max(100, 20*d)``.

    n_hidden : int or None, optional (default: None)
        Hidden layer width H.  If None, defaults to
        ``min(128, max(64, 4*d))``.

    n_train_steps : int, optional (default: 300)
        Number of Adam optimisation steps per fit call.

    learning_rate : float, optional (default: 1e-3)
        Adam learning rate.

    n_restarts_optimizer : int, optional (default: 3)
        Number of random re-initialisations of the MLP; the best result
        (highest LML) is kept.

    random_state : int, numpy.random.Generator, or None, optional
        Random state for reproducibility.

    use_jax : bool, optional (default: True)
        Kept for API compatibility; JAX is always used.

    Attributes
    ----------
    X_train_, y_train_, kernel_, kernel, alpha, alpha_,
    log_marginal_likelihood_value_, n_eval, n_eval_loglike,
    _fitted, copy_X_train, is_noise_in_kernel, n_restarts_optimizer,
    random_state, scales, noise_level
        All match the interface of :class:`experimental_random_feature_gp.RandomFeatureGP`.
    """

    def __init__(
        self,
        kernel="RBF",
        output_scale_prior=None,
        length_scale_prior=None,
        noise_level=1e-2,
        noise_fixed=True,
        n_features=None,
        n_hidden=None,
        n_train_steps=300,
        learning_rate=1e-3,
        n_restarts_optimizer=3,
        random_state=None,
        use_jax=True,
    ):
        # ---- priors / bounds -------------------------------------------------
        self.output_scale_prior = (
            [1e-2, 1e3] if output_scale_prior is None else list(output_scale_prior)
        )
        if length_scale_prior is None:
            length_scale_prior = np.array([[1e-3, 1e1]])
        self.length_scale_prior = np.atleast_2d(
            np.asarray(length_scale_prior, dtype=float)
        )
        # Clamp lower bound
        self.length_scale_prior[:, 0] = np.maximum(
            self.length_scale_prior[:, 0], 0.02
        )
        d = len(self.length_scale_prior)

        # ---- hyperparameter initialisations ----------------------------------
        self._output_scale = float(
            np.sqrt(self.output_scale_prior[0] * self.output_scale_prior[1])
        )
        self._length_scales = np.array(
            [np.sqrt(lo * hi) for lo, hi in self.length_scale_prior], dtype=float
        )  # (d,)

        # ---- noise -----------------------------------------------------------
        self._noise_level = float(noise_level)
        self.noise_fixed = noise_fixed
        self.is_noise_in_kernel = False

        # ---- network / feature sizes -----------------------------------------
        self._n_features_arg = n_features
        self._n_hidden_arg = n_hidden
        if n_features is None:
            self._M = max(100, 20 * d)
        else:
            self._M = int(n_features)
        if n_hidden is None:
            self._H = min(128, max(64, 4 * d))
        else:
            self._H = int(n_hidden)

        # ---- training settings -----------------------------------------------
        self.n_train_steps = int(n_train_steps)
        self.learning_rate = float(learning_rate)

        # ---- misc bookkeeping ------------------------------------------------
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.copy_X_train = True
        self.use_jax = use_jax

        # ---- counters / state ------------------------------------------------
        self.n_eval = 0
        self.n_eval_loglike = 0
        self._fitted = False
        self.log_marginal_likelihood_value_ = -np.inf

        # ---- internal cache (populated in fit) --------------------------------
        self._mlp_params = None    # pytree dict of MLP weights
        self._L_A = None           # (M, M) JAX float64 - Cholesky of A
        self._w = None             # (M,)   JAX float64 - MAP weights

        # ---- training data (populated in fit) --------------------------------
        self.X_train_ = None
        self.y_train_ = None

        # alpha: squared noise level, as in GPR
        self.alpha = float(noise_level) ** 2
        # alpha_: BLR dual coefficients (shape M)
        self.alpha_ = None

        # ---- kernel stubs ----------------------------------------------------
        self.kernel = _LFGPKernelStub(self._output_scale, self._length_scales)
        self.kernel_ = None

    # ------------------------------------------------------------------
    # Random state helpers
    # ------------------------------------------------------------------

    def _get_rng(self):
        """Return a numpy.random.Generator respecting ``self.random_state``."""
        rs = self.random_state
        if rs is None:
            return np.random.default_rng()
        if isinstance(rs, (int, np.integer)):
            return np.random.default_rng(int(rs))
        if isinstance(rs, np.random.Generator):
            return rs
        try:
            seed = int(rs.integers(0, 2 ** 31))
        except AttributeError:
            try:
                seed = int(rs.randint(0, 2 ** 31))
            except Exception:
                seed = 0
        return np.random.default_rng(seed)

    def _get_jax_key(self, rng=None):
        """Return a JAX PRNGKey from numpy rng."""
        if rng is None:
            rng = self._get_rng()
        seed = int(rng.integers(0, 2 ** 31))
        return jr.PRNGKey(seed)

    # ------------------------------------------------------------------
    # Network size helpers
    # ------------------------------------------------------------------

    def _compute_sizes(self, d):
        """Compute M and H for input dimensionality d."""
        if self._n_features_arg is None:
            M = max(100, 20 * d)
        else:
            M = self._M  # user-specified, keep fixed
        if self._n_hidden_arg is None:
            H = min(128, max(64, 4 * d))
        else:
            H = self._H
        return M, H

    # ------------------------------------------------------------------
    # Feature evaluation helpers
    # ------------------------------------------------------------------

    def _phi_jax(self, X_j):
        """Evaluate MLP features on JAX array X_j -> (N, M)."""
        return _mlp_features(self._mlp_params, X_j)

    def _phi_numpy(self, X):
        """Evaluate MLP features on numpy X -> numpy (N, M)."""
        X_j = jnp.array(X, dtype=jnp.float64)
        return np.asarray(self._phi_jax(X_j))

    # ------------------------------------------------------------------
    # Hyperparameter optimisation (Phase 1)
    # ------------------------------------------------------------------

    def _fit_hyperparameters(self, X, y, noise_sq, n_restarts=None):
        """
        Jointly optimise MLP weights and output scale by maximising BLR LML.

        Uses Adam for ``self.n_train_steps`` steps.  Runs ``n_restarts``
        independent random re-initialisations and keeps the best.

        Parameters
        ----------
        X : (N, d) ndarray
        y : (N,) ndarray
        noise_sq : float -- sigma_n^2 (fixed)
        n_restarts : int or None

        Returns
        -------
        best_lml : float
        """
        if n_restarts is None:
            n_restarts = self.n_restarts_optimizer

        d = X.shape[1]
        N = X.shape[0]
        M, H = self._compute_sizes(d)
        self._M = M
        self._H = H

        X_j = jnp.array(X, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)

        # Length scale prior geometric mean -- used for first-layer init scaling
        ls_prior_mean = np.array(
            [np.sqrt(self.length_scale_prior[j % len(self.length_scale_prior), 0]
                     * self.length_scale_prior[j % len(self.length_scale_prior), 1])
             for j in range(d)],
            dtype=float,
        )

        # Warm-start: fewer steps when refining from previous fit
        if self._fitted and self._mlp_params is not None:
            n_steps = max(30, self.n_train_steps // 5)
            # Only 1 restart on warm-start (from current params)
            n_restarts = 1
        else:
            n_steps = self.n_train_steps

        # L2 weight decay: 1/N (stronger regularisation with few points)
        weight_decay = 1.0 / max(N, 1)

        # log_sigma_f bounds
        log_os_lo = np.log(self.output_scale_prior[0])
        log_os_hi = np.log(self.output_scale_prior[1])
        log_sigma_f_init = float(np.log(self._output_scale))
        log_sigma_f_init = float(np.clip(log_sigma_f_init, log_os_lo, log_os_hi))

        # Gradient function w.r.t. (log_params, mlp_params)
        def neg_lml(log_params, mlp_params):
            return -_lml_lfgp(log_params, mlp_params, X_j, y_j, float(noise_sq))

        val_and_grad_fn = value_and_grad(neg_lml, argnums=(0, 1))

        rng = self._get_rng()
        best_lml = -np.inf
        best_mlp_params = None
        best_log_sigma_f = log_sigma_f_init

        n_total = max(1, n_restarts)

        for i_restart in range(n_total):
            key = self._get_jax_key(rng)

            # On the first restart (if already fitted), warm-start from the
            # current network weights; otherwise random init.
            if i_restart == 0 and self._fitted and self._mlp_params is not None:
                mlp_init = self._mlp_params
                lsf_init = jnp.array([log_sigma_f_init], dtype=jnp.float64)
            else:
                mlp_init = _init_mlp_params(d, H, M, ls_prior_mean, key)
                lsf_init = jnp.array(
                    [rng.uniform(log_os_lo, log_os_hi)], dtype=jnp.float64
                )

            # Initialise Adam states
            state_lsf = _adam_init(lsf_init, lr=self.learning_rate)
            state_mlp = _adam_init(mlp_init, lr=self.learning_rate)

            lml_val = -np.inf
            try:
                for step in range(n_steps):
                    neg_val, (g_lsf, g_mlp) = val_and_grad_fn(
                        state_lsf["params"], state_mlp["params"]
                    )
                    if not jnp.isfinite(neg_val):
                        break
                    # Update with weight decay on MLP params only
                    state_lsf = _adam_step(state_lsf, g_lsf, weight_decay=0.0)
                    state_mlp = _adam_step(state_mlp, g_mlp,
                                           weight_decay=weight_decay)
                    # Clip log_sigma_f to bounds
                    clipped = jnp.clip(state_lsf["params"][0],
                                       log_os_lo, log_os_hi)
                    state_lsf["params"] = jnp.array([clipped], dtype=jnp.float64)

                lml_val = float(-neg_val) if jnp.isfinite(neg_val) else -np.inf
            except Exception:
                pass

            self.n_eval_loglike += n_steps

            if np.isfinite(lml_val) and lml_val > best_lml:
                best_lml = lml_val
                best_mlp_params = state_mlp["params"]
                best_log_sigma_f = float(state_lsf["params"][0])

        # Fallback: keep current params if all restarts failed
        if best_mlp_params is None:
            if self._mlp_params is not None:
                best_mlp_params = self._mlp_params
            else:
                key_fb = self._get_jax_key(rng)
                best_mlp_params = _init_mlp_params(d, H, M, ls_prior_mean, key_fb)
            log_params_fb = jnp.array([log_sigma_f_init], dtype=jnp.float64)
            try:
                best_lml = float(
                    _lml_lfgp(log_params_fb, best_mlp_params, X_j, y_j,
                               float(noise_sq))
                )
                self.n_eval_loglike += 1
            except Exception:
                best_lml = -np.inf

        # Commit
        self._mlp_params = best_mlp_params
        self._output_scale = float(np.exp(best_log_sigma_f))
        # Also update _length_scales to prior mean (NN absorbs length scale info)
        if len(self._length_scales) != d:
            self._length_scales = np.array(ls_prior_mean, dtype=float)

        return best_lml

    # ------------------------------------------------------------------
    # BLR fit (Phase 2, given fixed MLP params)
    # ------------------------------------------------------------------

    def _blr_fit(self, X, y, noise_sq_scalar, output_scale_sq):
        """
        Compute BLR normal equations and cache L_A and MAP weights w.

        Parameters
        ----------
        X : (N, d) ndarray
        y : (N,) ndarray
        noise_sq_scalar : float
        output_scale_sq : float
        """
        X_j = jnp.array(X, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)
        Phi_j = self._phi_jax(X_j)  # (N, M)

        lam = float(noise_sq_scalar) / float(output_scale_sq)
        L_A, w = _blr_fit_jax(Phi_j, y_j, lam)

        self._L_A = L_A
        self._w = w
        self.alpha_ = np.asarray(w)

    # ------------------------------------------------------------------
    # Properties matching GaussianProcessRegressor interface
    # ------------------------------------------------------------------

    @property
    def scales(self):
        """(output_scale, length_scales) tuple expected by SurrogateModel."""
        return (self._output_scale, np.array(self._length_scales))

    @property
    def noise_level(self):
        """Noise standard deviation sigma_n."""
        return self._noise_level

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------

    def fit(self, X=None, y=None, noise_level=None, fit_hyperparameters=True,
            validate=True):
        r"""
        Fit the Learned Feature GP to training data.

        Parameters
        ----------
        X : (N, d) array or None
        y : (N,) array or None
        noise_level : float, (N,) array, or None
        fit_hyperparameters : bool or dict (default: True)
        validate : bool (default: True)

        Returns
        -------
        self
        """
        # -- Handle X, y = None: refit to cached data -------------------------
        if X is None and y is None:
            if not self._fitted:
                return self
            X = self.X_train_
            y = self.y_train_
            if noise_level is None:
                noise_sq = np.atleast_1d(np.asarray(self.alpha, dtype=float))
            else:
                noise_sq = np.maximum(
                    np.atleast_1d(np.asarray(noise_level, dtype=float)) ** 2, 1e-12
                )
        else:
            if X is None or y is None:
                raise ValueError("Pass both X and y, or neither (X=None, y=None).")
            X = np.atleast_2d(np.asarray(X, dtype=float))
            y = np.atleast_1d(np.asarray(y, dtype=float))
            if self.copy_X_train:
                self.X_train_ = np.copy(X)
                self.y_train_ = np.copy(y)
            else:
                self.X_train_ = X
                self.y_train_ = y

            if noise_level is not None:
                nl = np.atleast_1d(np.asarray(noise_level, dtype=float))
                self.alpha = np.maximum(nl ** 2, 1e-12)
                self._noise_level = float(np.sqrt(np.mean(self.alpha)))
                noise_sq = self.alpha
            else:
                noise_sq = np.atleast_1d(np.asarray(self.alpha, dtype=float))

        # -- Adapt sizes to current dimensionality ----------------------------
        d = X.shape[1]
        M, H = self._compute_sizes(d)
        self._M = M
        self._H = H

        # -- Initialise MLP if needed (first call or d changed) ---------------
        need_new_mlp = (
            self._mlp_params is None
            or self._mlp_params["W1"].shape[1] != d
        )
        if need_new_mlp:
            if len(self._length_scales) != d:
                self._length_scales = np.array(
                    [
                        np.sqrt(
                            self.length_scale_prior[j % len(self.length_scale_prior), 0]
                            * self.length_scale_prior[j % len(self.length_scale_prior), 1]
                        )
                        for j in range(d)
                    ],
                    dtype=float,
                )
            rng = self._get_rng()
            key = self._get_jax_key(rng)
            self._mlp_params = _init_mlp_params(
                d, H, M, self._length_scales, key
            )

        # -- Parse fit_hyperparameters ----------------------------------------
        if fit_hyperparameters is False:
            do_fit_hp = False
            hp_kwargs = {}
        elif fit_hyperparameters is True:
            do_fit_hp = True
            hp_kwargs = {}
        elif isinstance(fit_hyperparameters, Mapping):
            do_fit_hp = True
            hp_kwargs = dict(fit_hyperparameters)
        else:
            raise TypeError(
                f"'fit_hyperparameters' must be bool or dict, "
                f"got {type(fit_hyperparameters)}"
            )

        noise_sq_scalar = float(np.mean(noise_sq))

        # -- Phase 1: learn features + hyperparameters ------------------------
        if do_fit_hp:
            lml = self._fit_hyperparameters(
                X, y,
                noise_sq=noise_sq_scalar,
                n_restarts=hp_kwargs.get("n_restarts", None),
            )
            self.log_marginal_likelihood_value_ = lml
        else:
            # Evaluate LML at current params without optimising
            try:
                log_params_j = jnp.array(
                    [np.log(self._output_scale)], dtype=jnp.float64
                )
                lml = float(
                    _lml_lfgp(
                        log_params_j, self._mlp_params,
                        jnp.array(X, dtype=jnp.float64),
                        jnp.array(y, dtype=jnp.float64),
                        noise_sq_scalar,
                    )
                )
                self.n_eval_loglike += 1
            except Exception:
                lml = -np.inf
            self.log_marginal_likelihood_value_ = lml

        # -- Phase 2: BLR fit with learned features ---------------------------
        self._blr_fit(X, y, noise_sq_scalar, self._output_scale ** 2)

        # -- Update kernel stubs ----------------------------------------------
        self.kernel_ = _LFGPKernelStub(self._output_scale, self._length_scales)
        if not self._fitted:
            self.kernel = _LFGPKernelStub(self._output_scale, self._length_scales)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Public: predict
    # ------------------------------------------------------------------

    def predict(self, X, return_std=False, return_mean_grad=False,
                return_std_grad=False, validate=True):
        """
        Predict output for X.

        Parameters
        ----------
        X : (N, d) array
        return_std : bool (default: False)
        return_mean_grad : bool (default: False)
            Gradient of mean w.r.t. X[0].  Only for N=1.
        return_std_grad : bool (default: False)
            Gradient of std w.r.t. X[0].  Requires return_std and
            return_mean_grad.  Only for N=1.
        validate : bool (default: True)

        Returns
        -------
        list  -- [y_mean, (y_std,) (y_mean_grad,) (y_std_grad,)]
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        N = X.shape[0]
        self.n_eval += N

        if return_std_grad and not (return_std and return_mean_grad):
            raise ValueError(
                "Cannot return std_gradient without also returning "
                "std (return_std=True) and mean_grad (return_mean_grad=True)."
            )
        if N != 1 and (return_mean_grad or return_std_grad):
            raise ValueError(
                "Gradient outputs (return_mean_grad / return_std_grad) are only "
                "supported for a single query point (N=1)."
            )

        # --- Prior predictions (not yet fitted) --------------------------------
        if not self._fitted:
            y_mean = np.zeros(N)
            return_values = [y_mean]
            if return_std:
                return_values.append(np.full(N, self._output_scale))
            if return_mean_grad:
                return_values.append(np.zeros(X.shape[1]))
                if return_std_grad:
                    return_values.append(np.zeros(X.shape[1]))
            return return_values

        # --- Posterior predictions (fitted) ------------------------------------
        X_j = jnp.array(X, dtype=jnp.float64)
        sigma_f_sq = float(self._output_scale ** 2)

        # --- Gradient mode: single point, use JAX autodiff ---------------------
        if return_mean_grad or return_std_grad:
            x_j = X_j[0]  # (d,)
            w_j = self._w
            L_A_j = self._L_A
            mlp_params = self._mlp_params

            def mean_of_x(x):
                phi_x = _mlp_features(mlp_params, x[None, :])[0]  # (M,)
                return jnp.dot(phi_x, w_j)

            def std_of_x(x):
                phi_x = _mlp_features(mlp_params, x[None, :])[0]  # (M,)
                var = _predict_var_single(phi_x, L_A_j, sigma_f_sq)
                return jnp.sqrt(jnp.maximum(var, 0.0))

            mean_val = float(mean_of_x(x_j))
            return_values = [np.array([mean_val])]

            if return_std:
                std_val = float(std_of_x(x_j))
                return_values.append(np.array([std_val]))

            grad_mean = np.asarray(grad(mean_of_x)(x_j))
            return_values.append(grad_mean)

            if return_std_grad:
                grad_std = np.asarray(grad(std_of_x)(x_j))
                return_values.append(grad_std)

            return return_values

        # --- Batch prediction (no gradients) -----------------------------------
        Phi_j = self._phi_jax(X_j)          # (N, M)
        y_mean_j = Phi_j @ self._w           # (N,)
        y_mean = np.asarray(y_mean_j)
        return_values = [y_mean]

        if return_std:
            y_var_j = _predict_var_batch(Phi_j, self._L_A, sigma_f_sq)  # (N,)
            y_var = np.asarray(y_var_j)
            neg_mask = y_var < 0.0
            if np.any(neg_mask):
                warnings.warn(
                    "Predicted variances smaller than 0. Setting those to 0."
                )
                y_var = np.where(neg_mask, 0.0, y_var)
            return_values.append(np.sqrt(y_var))

        return return_values

    # ------------------------------------------------------------------
    # Public: predict_std (fast std-only path)
    # ------------------------------------------------------------------

    def predict_std(self, X, validate=True):
        """
        Predictive standard deviation at query points X.

        Parameters
        ----------
        X : (N, d) array
        validate : bool (default: True)

        Returns
        -------
        y_std : (N,) ndarray
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        N = X.shape[0]
        self.n_eval += N

        if not self._fitted:
            return np.full(N, self._output_scale)

        X_j = jnp.array(X, dtype=jnp.float64)
        Phi_j = self._phi_jax(X_j)
        sigma_f_sq = float(self._output_scale ** 2)
        y_var_j = _predict_var_batch(Phi_j, self._L_A, sigma_f_sq)
        y_var = np.asarray(y_var_j)

        neg_mask = y_var < 0.0
        if np.any(neg_mask):
            warnings.warn("Predicted variances smaller than 0. Setting those to 0.")
            y_var = np.where(neg_mask, 0.0, y_var)

        return np.sqrt(y_var)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"LearnedFeatureGP(M={self._M}, H={self._H}, "
            f"output_scale={self._output_scale:.4g}, "
            f"length_scales={np.round(self._length_scales, 4)}, "
            f"noise_level={self._noise_level:.4g}, "
            f"n_train_steps={self.n_train_steps}, "
            f"{status})"
        )

    def __str__(self):
        return self.__repr__()
