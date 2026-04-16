"""
Experimental Random Feature GP (RFGP) backend.

This module is not part of the conservative GPry baseline. It is kept for
opt-in tests of scalable approximate GP behavior.

This module implements a Gaussian Process approximation using Random Fourier Features
(RFF), turning O(N^3) GP inference into O(N*M^2 + M^3) Bayesian linear regression,
where M is the number of random features and N is the number of training points.

The key idea (Rahimi & Recht 2007) is that shift-invariant kernels can be approximated
by an explicit feature map:

    k(x, x') ≈ phi(x)^T phi(x')

where for the RBF kernel with output scale sigma_f and length scales l_j:

    phi(x) = sqrt(2/M) * cos(x @ W.T + b)
    W_ij   ~ N(0, 1/l_j^2)   (spectral frequencies from the kernel's Fourier transform)
    b_i    ~ Uniform(0, 2*pi)

With this feature map, GP inference reduces to Bayesian linear regression:

    y = Phi w + eps,   w ~ N(0, sigma_f^2 I_M),   eps ~ N(0, sigma_n^2 I_N)

giving a marginal distribution y ~ N(0, sigma_f^2 Phi Phi^T + sigma_n^2 I_N) and
posterior predictive:

    A        = Phi^T Phi + (sigma_n^2 / sigma_f^2) I_M
    w_post   = sigma_f^2 A^{-1} Phi^T y
    mean(x*) = phi(x*)^T w_post
    var(x*)  = sigma_f^2 phi(x*)^T A^{-1} phi(x*)

All core numerics use JAX for automatic differentiation, enabling gradient-based
hyperparameter optimisation and efficient computation of mean/std gradients needed by
acquisition functions.

Interface
---------
The class matches the interface expected by :class:`surrogate.SurrogateModel`. All public
attributes and methods (``fit``, ``predict``, ``predict_std``, ``kernel``, ``kernel_``,
``scales``, ``noise_level``, ``is_noise_in_kernel``, ``alpha``, ``alpha_``, etc.) are
present and behave analogously to :class:`gpr.GaussianProcessRegressor`.

References
----------
Rahimi, A. & Recht, B. (2007). Random features for large-scale kernel machines.
Advances in Neural Information Processing Systems, 20.
"""

# Builtin
import warnings
from typing import Mapping

# External
import numpy as np
from scipy.optimize import minimize

# JAX (required)
import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad

# Enable 64-bit for numerical accuracy (critical for log marginal likelihood)
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Stub kernel object returned via gpr.kernel / gpr.kernel_
# ---------------------------------------------------------------------------

class _RFFKernelStub:
    """
    Minimal stub that satisfies the ``kernel`` / ``kernel_`` attribute contract.

    ``surrogate.py`` and ``gpr.py`` access:
      - ``gpr.kernel.requires_vector_input`` (for input validation dispatch)
      - ``str(gpr.kernel)`` (for logging / __str__ of SurrogateModel)
      - ``gpr.kernel.hyperparameters`` (iterated in SurrogateModel.__str__)

    This stub satisfies all three without importing the full kernel machinery.
    """

    def __init__(self, output_scale, length_scales):
        self.output_scale = float(output_scale)
        self.length_scales = np.atleast_1d(np.asarray(length_scales, dtype=float))
        # Tell the surrogate that we always expect numeric 2-d arrays as input
        self.requires_vector_input = True
        # Empty sequence -- iterated in surrogate.__str__ but produces no output
        self.hyperparameters = []

    def __str__(self):
        ls = ", ".join(f"{v:.4g}" for v in self.length_scales)
        return f"RFF-RBF(output_scale={self.output_scale:.4g}, length_scales=[{ls}])"

    def __repr__(self):
        return self.__str__()


# ---------------------------------------------------------------------------
# Pure-JAX core computations (module-level, JIT-compiled)
# ---------------------------------------------------------------------------

def _make_feature_map(W, b):
    """
    Return a JIT-compiled RFF feature-map function for fixed spectral samples W, b.

    Parameters
    ----------
    W : (M, d) JAX array - spectral frequencies
    b : (M,)   JAX array - phase offsets in [0, 2*pi)

    Returns
    -------
    phi : callable (N, d) -> (N, M)
        phi(X) = sqrt(2/M) * cos(X @ W.T + b)
    """
    M_val = W.shape[0]

    @jit
    def phi(X):
        return jnp.sqrt(2.0 / M_val) * jnp.cos(X @ W.T + b)

    return phi


@jit
def _blr_fit_jax(Phi, y, lam):
    """
    Bayesian linear regression: compute Cholesky of A and MAP weights.

    Model: y = Phi w + eps,  w ~ N(0, I),  eps ~ N(0, lam I)
    where lam = sigma_n^2 / sigma_f^2  (sigma_f^2 is folded into the final predictions).

    A = Phi^T Phi + lam I_M
    L_A = cholesky(A)   [lower triangular]
    w   = A^{-1} Phi^T y

    Parameters
    ----------
    Phi : (N, M) float64
    y   : (N,) float64
    lam : float - sigma_n^2 / sigma_f^2

    Returns
    -------
    L_A : (M, M) lower-triangular Cholesky of A
    w   : (M,) MAP weight vector
    """
    M = Phi.shape[1]
    A = Phi.T @ Phi + lam * jnp.eye(M)
    L_A = jnp.linalg.cholesky(A)
    rhs = Phi.T @ y
    w = jax.scipy.linalg.cho_solve((L_A, True), rhs)
    return L_A, w


@jit
def _predict_var_batch(Phi_star, L_A, sigma_f_sq):
    """
    Predictive variance for a batch of N query points.

    var(x_i) = sigma_f^2 * phi(x_i)^T A^{-1} phi(x_i)
             = sigma_f^2 * ||L_A^{-1} phi(x_i)||^2

    Parameters
    ----------
    Phi_star : (N, M) float64
    L_A      : (M, M) lower-triangular Cholesky of A
    sigma_f_sq : float

    Returns
    -------
    var : (N,) float64
    """
    # Solve L_A V^T = Phi_star^T  ->  V^T = L_A^{-1} Phi_star^T  (shape M x N)
    V = jax.scipy.linalg.solve_triangular(L_A, Phi_star.T, lower=True)  # (M, N)
    return sigma_f_sq * jnp.sum(V ** 2, axis=0)  # (N,)


@jit
def _predict_var_single(phi_x, L_A, sigma_f_sq):
    """
    Predictive variance for a single query point.

    Parameters
    ----------
    phi_x  : (M,) float64
    L_A    : (M, M) lower-triangular Cholesky of A
    sigma_f_sq : float

    Returns
    -------
    var : float
    """
    v = jax.scipy.linalg.solve_triangular(L_A, phi_x, lower=True)  # (M,)
    return sigma_f_sq * jnp.dot(v, v)


def _lml_jax(log_params, X_j, y_j, noise_sq, W_fixed, b_fixed, old_ls_j):
    """
    Bayesian linear regression log marginal likelihood, fully differentiable via JAX.

    Model (in full): y ~ N(0, sigma_f^2 Phi Phi^T + sigma_n^2 I_N)

    Using the matrix determinant lemma (Sylvester identity) and Woodbury identity
    to reduce all N-dim linear algebra to M-dim (where M << N is typical):

    Let lam = sigma_n^2 / sigma_f^2,  A = Phi^T Phi + lam I_M.

    log |sigma_f^2 Phi Phi^T + sigma_n^2 I_N|
      = log |sigma_f^2 (Phi Phi^T + lam I_N)|
      = N log sigma_f^2  +  log |lam I_N + Phi Phi^T|

    By Sylvester:  |lam I_N + Phi Phi^T| = lam^{N-M} |lam I_M + Phi^T Phi|
                                          = lam^{N-M} |A|

    So:  log-det = N log(sigma_f^2) + (N-M) log(lam) + log|A|
                 = N log(sigma_f^2) + (N-M) log(lam) + 2 sum_k log(L_A[k,k])

    Woodbury quadratic form:
      (sigma_f^2 Phi Phi^T + sigma_n^2 I)^{-1}
        = (1/sigma_n^2) [I - Phi (lam I_M + Phi^T Phi)^{-1} Phi^T]
        = (1/sigma_n^2) [I - Phi A^{-1} Phi^T]

    y^T (...) y  =  (1/sigma_n^2) [||y||^2 - ||L_A^{-1} Phi^T y||^2]

    LML = -0.5 * [y^T (...) y + log-det + N log(2 pi)]

    Parameters
    ----------
    log_params  : (1 + d,) JAX array - [log_sigma_f, log_l_1, ..., log_l_d]
    X_j         : (N, d) JAX float64
    y_j         : (N,) JAX float64
    noise_sq    : float - sigma_n^2 (fixed)
    W_fixed     : (M, d) JAX float64 - sampled frequencies (treated as constant)
    b_fixed     : (M,) JAX float64 - sampled phases (treated as constant)
    old_ls_j    : (d,) JAX float64 - length scales used when W_fixed was sampled

    Returns
    -------
    lml : scalar float64
    """
    log_sigma_f = log_params[0]
    log_ls = log_params[1:]
    sigma_f_sq = jnp.exp(2.0 * log_sigma_f)
    length_scales = jnp.exp(log_ls)  # (d,)

    # Rescale W from old length scales to new ones.
    # W_fixed was drawn with std = 1/old_ls; we want std = 1/new_ls.
    # Scaling W_fixed by old_ls / new_ls achieves this.
    W_eff = W_fixed * (old_ls_j / length_scales)[None, :]  # (M, d)

    M = W_eff.shape[0]
    N = X_j.shape[0]

    # Feature matrix Phi = sqrt(2/M) * cos(X W_eff^T + b)
    Phi = jnp.sqrt(2.0 / M) * jnp.cos(X_j @ W_eff.T + b_fixed)  # (N, M)

    sigma_n_sq = noise_sq  # scalar constant
    lam = sigma_n_sq / sigma_f_sq  # dimensionless regulariser

    # M-dim Cholesky
    A = Phi.T @ Phi + lam * jnp.eye(M)
    L_A = jnp.linalg.cholesky(A)

    # Woodbury quadratic
    Phi_T_y = Phi.T @ y_j  # (M,)
    v = jax.scipy.linalg.solve_triangular(L_A, Phi_T_y, lower=True)  # (M,)
    quadratic = (jnp.dot(y_j, y_j) - jnp.dot(v, v)) / sigma_n_sq

    # Log-determinant (M-dim Cholesky path)
    log_det_A = 2.0 * jnp.sum(jnp.log(jnp.diag(L_A)))
    log_det = N * jnp.log(sigma_f_sq) + (N - M) * jnp.log(lam) + log_det_A

    lml = -0.5 * (quadratic + log_det + N * jnp.log(2.0 * jnp.pi))
    return lml


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RandomFeatureGP:
    r"""
    Random Feature Gaussian Process -- a scalable drop-in GP backend for GPry.

    Approximates a GP with an RBF kernel using Random Fourier Features (Rahimi &
    Recht 2007), reducing the O(N^3) complexity of exact GPs to O(N*M^2 + M^3)
    where M is the number of random features.

    This class is a drop-in replacement for :class:`gpr.GaussianProcessRegressor`
    and can be passed as the ``gpr`` argument to :class:`surrogate.SurrogateModel`.
    All public attributes and methods used by the surrogate are implemented.

    Parameters
    ----------
    kernel : str, optional (default: "RBF")
        Kernel type. Only "RBF" is currently supported. Kept for API compatibility.

    output_scale_prior : list [min, max], optional (default: [1e-2, 1e3])
        Bounds for the output scale sigma_f (in normalised logp units).

    length_scale_prior : array-like, shape (d, 2), optional
        Per-dimension bounds for the length scales. The number of rows encodes the
        input dimensionality d. Defaults to [[1e-3, 1e1]] (1-d).

    noise_level : float, optional (default: 1e-2)
        Initial noise standard deviation sigma_n. Updated at each ``fit`` call.

    noise_fixed : bool, optional (default: True)
        If True, sigma_n is not optimised (it is fixed at the value passed to
        ``fit``). If False, sigma_n is treated as an additional hyperparameter.
        Note: noise optimisation is not yet implemented; this flag is reserved.

    n_features : int or None, optional (default: None)
        Number of random Fourier features M. If None, defaults to
        ``max(300, 50 * d)`` where d is the input dimensionality.

    n_restarts_optimizer : int, optional (default: 5)
        Number of random restarts for the hyperparameter optimiser (L-BFGS-B).

    random_state : int, numpy.random.Generator, or None, optional
        Random state for reproducibility.

    use_jax : bool, optional (default: True)
        If True, use JAX for computations. Kept for API compatibility with
        GaussianProcessRegressor; JAX is always used.

    Attributes
    ----------
    X_train_ : ndarray, shape (N, d)
        Training inputs (set after ``fit``).

    y_train_ : ndarray, shape (N,)
        Training targets (set after ``fit``).

    kernel_ : _RFFKernelStub
        Stub carrying the current hyperparameter values; updated after each fit.

    kernel : _RFFKernelStub
        Initial kernel stub; kept for API compatibility.

    alpha : float or ndarray, shape (N,)
        Squared noise level(s): sigma_n^2. Updated by ``fit``.

    alpha_ : ndarray, shape (M,)
        MAP weight vector ``A^{-1} Phi^T y`` (the BLR dual coefficients).
        Stored for API compatibility; shape is M (features), not N (samples).

    log_marginal_likelihood_value_ : float
        LML at the last fit point.

    n_eval : int
        Counter for prediction evaluations (number of individual query points).

    n_eval_loglike : int
        Counter for log marginal likelihood evaluations.

    _fitted : bool
        True after the first successful ``fit``.

    copy_X_train : bool
        Whether to copy training data at fit time (always True for this class).

    is_noise_in_kernel : bool
        False -- noise is always handled as an external diagonal term, not in the
        kernel. (Matches the convention of the standard GPR when noise_fixed=True.)

    n_restarts_optimizer : int
        Number of optimizer restarts (set at construction, used by the surrogate).

    random_state : int, Generator, or None
        Random state; settable (the surrogate may reassign this).

    Notes
    -----
    The spectral frequencies W are resampled whenever ``fit`` is called with
    ``fit_hyperparameters != False`` and the optimised length scales differ from
    the ones used for the last sample. This ensures the feature map is consistent
    with the current kernel hyperparameters.

    Prediction is differentiable via JAX autodiff (required by acquisition functions
    using ``return_mean_grad`` / ``return_std_grad``).
    """

    def __init__(
        self,
        kernel="RBF",
        output_scale_prior=None,
        length_scale_prior=None,
        noise_level=1e-2,
        noise_fixed=True,
        n_features=None,
        n_restarts_optimizer=5,
        random_state=None,
        use_jax=True,
    ):
        # ---- priors / bounds -------------------------------------------------
        self.output_scale_prior = (
            [1e-2, 1e3] if output_scale_prior is None else list(output_scale_prior)
        )
        if length_scale_prior is None:
            length_scale_prior = np.array([[1e-3, 1e1]])
        self.length_scale_prior = np.atleast_2d(np.asarray(length_scale_prior, dtype=float))
        # Clamp lower bound: too-small length scales cause degenerate predictions
        self.length_scale_prior[:, 0] = np.maximum(
            self.length_scale_prior[:, 0], 0.02
        )
        d = len(self.length_scale_prior)

        # ---- hyperparameter initialisations ----------------------------------
        # Geometric mean of bounds as starting point
        self._output_scale = float(
            np.sqrt(self.output_scale_prior[0] * self.output_scale_prior[1])
        )
        self._length_scales = np.array(
            [np.sqrt(lo * hi) for lo, hi in self.length_scale_prior], dtype=float
        )  # shape (d,)

        # ---- noise -----------------------------------------------------------
        self._noise_level = float(noise_level)
        self.noise_fixed = noise_fixed
        # Always False for RFGP: noise is treated as external alpha, not in kernel
        self.is_noise_in_kernel = False

        # ---- random features -------------------------------------------------
        self._n_features_arg = n_features  # keep None to allow auto-resize on d change
        if n_features is None:
            self._M = max(300, 50 * d)
        else:
            self._M = int(n_features)

        # ---- misc bookkeeping ------------------------------------------------
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.copy_X_train = True
        self.use_jax = use_jax  # kept for API compatibility

        # ---- counters / state ------------------------------------------------
        self.n_eval = 0
        self.n_eval_loglike = 0
        self._fitted = False
        self.log_marginal_likelihood_value_ = -np.inf

        # ---- internal cache (populated in fit) -------------------------------
        self._W = None          # (M, d) JAX float64 - spectral frequencies
        self._b = None          # (M,)   JAX float64 - phase offsets
        self._L_A = None        # (M, M) JAX float64 - Cholesky of A
        self._w = None          # (M,)   JAX float64 - MAP weights
        self._phi_fn = None     # compiled feature-map function

        # ---- training data (populated in fit) --------------------------------
        self.X_train_ = None
        self.y_train_ = None

        # alpha: squared noise level (scalar or per-sample array), as in GPR
        self.alpha = float(noise_level) ** 2
        # alpha_: BLR dual coefficients (shape M, not N -- analogous to GPR's alpha_)
        self.alpha_ = None

        # ---- kernel stubs ----------------------------------------------------
        self.kernel = _RFFKernelStub(self._output_scale, self._length_scales)
        self.kernel_ = None  # set to stub after first fit

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
        # Legacy RandomState: extract an integer seed
        try:
            seed = int(rs.integers(0, 2 ** 31))
        except AttributeError:
            try:
                seed = int(rs.randint(0, 2 ** 31))
            except Exception:
                seed = 0
        return np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Feature sampling
    # ------------------------------------------------------------------

    def _sample_features(self, length_scales, rng=None):
        """
        Draw M random frequencies W and phases b for the RBF kernel approximation.

        For the anisotropic RBF kernel with length scales l_j, the spectral
        density (Bochner's theorem) is a product of independent Gaussians:
            w_{ij} ~ N(0, 1/l_j^2)

        Parameters
        ----------
        length_scales : (d,) array_like
        rng : numpy.random.Generator or None

        Returns
        -------
        W : (M, d) JAX float64
        b : (M,)   JAX float64
        """
        if rng is None:
            rng = self._get_rng()
        d = len(length_scales)
        M = self._M
        # Each column j ~ N(0, 1/l_j^2), so scale unit-normal by 1/l_j
        W_np = rng.standard_normal((M, d)) / np.asarray(length_scales)[None, :]
        b_np = rng.uniform(0.0, 2.0 * np.pi, size=(M,))
        return (
            jnp.array(W_np, dtype=jnp.float64),
            jnp.array(b_np, dtype=jnp.float64),
        )

    def _rebuild_feature_map(self, length_scales, rng=None):
        """
        Resample W and b with the given ``length_scales``, update phi function.

        Always call this after changing ``self._length_scales`` so that the stored
        feature map is consistent with the current hyperparameters.
        """
        self._W, self._b = self._sample_features(length_scales, rng=rng)
        self._phi_fn = _make_feature_map(self._W, self._b)

    # ------------------------------------------------------------------
    # Properties matching GaussianProcessRegressor interface
    # ------------------------------------------------------------------

    @property
    def scales(self):
        """
        Kernel scales as ``(output_scale, length_scales_array)``.

        Matches the property expected by :class:`surrogate.SurrogateModel.scales`.
        """
        return (self._output_scale, np.array(self._length_scales))

    @property
    def noise_level(self):
        """Noise standard deviation (not squared): sigma_n."""
        return self._noise_level

    # ------------------------------------------------------------------
    # Internal: feature evaluation helpers
    # ------------------------------------------------------------------

    def _phi_numpy(self, X):
        """Evaluate the feature map on a numpy array X, return numpy (N, M)."""
        X_j = jnp.array(X, dtype=jnp.float64)
        return np.asarray(self._phi_fn(X_j))

    def _phi_jax(self, X_j):
        """Evaluate the feature map on a JAX array X_j, return JAX (N, M)."""
        return self._phi_fn(X_j)

    # ------------------------------------------------------------------
    # Log marginal likelihood (value + gradient via JAX autodiff)
    # ------------------------------------------------------------------

    def _lml_value_and_grad(self, log_params_np, X_j, y_j, noise_sq):
        """
        Return (lml, grad_lml) w.r.t. ``log_params`` using JAX autodiff.

        The feature map is implicitly re-derived inside ``_lml_jax`` by rescaling
        the stored fixed samples W by the ratio (old_ls / new_ls). This makes the
        LML differentiable through the length scales without re-sampling.

        Parameters
        ----------
        log_params_np : (1 + d,) numpy array - [log_sigma_f, log_l_1, ..., log_l_d]
        X_j  : (N, d) JAX float64
        y_j  : (N,) JAX float64
        noise_sq : float

        Returns
        -------
        lml  : float
        grad : (1 + d,) numpy array
        """
        self.n_eval_loglike += 1
        log_params_j = jnp.array(log_params_np, dtype=jnp.float64)
        old_ls_j = jnp.array(self._length_scales, dtype=jnp.float64)

        lml_fn = lambda lp: _lml_jax(
            lp, X_j, y_j, float(noise_sq), self._W, self._b, old_ls_j
        )
        val, g = value_and_grad(lml_fn)(log_params_j)
        return float(val), np.asarray(g)

    # ------------------------------------------------------------------
    # Hyperparameter optimisation
    # ------------------------------------------------------------------

    def _fit_hyperparameters(self, X, y, noise_sq, start_from_current=True,
                              n_restarts=None, maxiter=None):
        """
        Maximise the BLR log marginal likelihood over (log_sigma_f, log_l_1..d).

        Uses scipy L-BFGS-B with JAX-computed gradients and ``n_restarts``
        random initialisations. After finding the optimum, the stored spectral
        samples are resampled with the final length scales.

        Parameters
        ----------
        X : (N, d) ndarray
        y : (N,) ndarray
        noise_sq : float - sigma_n^2 (fixed during optimisation if noise_fixed=True)
        start_from_current : bool
            If True and already fitted, start the first restart from the current
            hyperparameters.
        n_restarts : int or None
            Number of optimiser restarts. If None, uses ``self.n_restarts_optimizer``.

        Returns
        -------
        best_lml : float - LML at the best found hyperparameters
        """
        if n_restarts is None:
            n_restarts = self.n_restarts_optimizer
        if maxiter is None:
            maxiter = 300

        d = X.shape[1]

        # Bounds in log-space
        log_os_lo = np.log(self.output_scale_prior[0])
        log_os_hi = np.log(self.output_scale_prior[1])
        log_ls_bounds = [
            (np.log(self.length_scale_prior[j, 0]),
             np.log(self.length_scale_prior[j, 1]))
            for j in range(d)
        ]
        bounds = [(log_os_lo, log_os_hi)] + log_ls_bounds  # list of (lo, hi) tuples

        X_j = jnp.array(X, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)
        noise_sq_f = float(noise_sq)

        def neg_lml_and_grad(log_params_np):
            lml_val, lml_grad = self._lml_value_and_grad(
                log_params_np, X_j, y_j, noise_sq_f
            )
            return -lml_val, -lml_grad

        rng = self._get_rng()
        best_lml = -np.inf
        best_params = None

        n_total = max(1, n_restarts)
        for i_restart in range(n_total):
            if i_restart == 0 and start_from_current and self._fitted:
                x0 = np.concatenate(
                    [[np.log(self._output_scale)], np.log(self._length_scales)]
                )
            else:
                # Log-uniform random initial point within bounds
                x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = minimize(
                        neg_lml_and_grad,
                        x0,
                        method="L-BFGS-B",
                        jac=True,
                        bounds=bounds,
                        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
                    )
                candidate_lml = float(-result.fun)
                if np.isfinite(candidate_lml) and candidate_lml > best_lml:
                    best_lml = candidate_lml
                    best_params = result.x.copy()
            except Exception:
                continue

        if best_params is None:
            # All restarts failed; keep current hyperparameters
            best_params = np.concatenate(
                [[np.log(self._output_scale)], np.log(self._length_scales)]
            )
            best_lml = float(
                _lml_jax(
                    jnp.array(best_params, dtype=jnp.float64),
                    X_j, y_j, noise_sq_f,
                    self._W, self._b,
                    jnp.array(self._length_scales, dtype=jnp.float64),
                )
            )
            self.n_eval_loglike += 1

        # Commit optimised hyperparameters
        new_output_scale = float(np.exp(best_params[0]))
        new_length_scales = np.exp(best_params[1:])

        # Resample features with the optimised length scales so the stored
        # W is consistent with _length_scales (important for predict)
        self._rebuild_feature_map(new_length_scales, rng=rng)
        self._output_scale = new_output_scale
        self._length_scales = new_length_scales

        return best_lml

    # ------------------------------------------------------------------
    # BLR fit (given fixed W, b and hyperparameters)
    # ------------------------------------------------------------------

    def _blr_fit(self, X, y, noise_sq_scalar, output_scale_sq):
        """
        Solve the BLR normal equations and cache L_A and MAP weights w.

        This is called after ``_fit_hyperparameters`` (which updates self._W / self._b)
        or directly when ``fit_hyperparameters=False``.

        Parameters
        ----------
        X : (N, d) ndarray
        y : (N,) ndarray
        noise_sq_scalar : float - effective sigma_n^2 (scalar)
        output_scale_sq : float - sigma_f^2
        """
        X_j = jnp.array(X, dtype=jnp.float64)
        y_j = jnp.array(y, dtype=jnp.float64)
        Phi_j = self._phi_jax(X_j)  # (N, M)

        lam = float(noise_sq_scalar) / float(output_scale_sq)
        L_A, w = _blr_fit_jax(Phi_j, y_j, lam)

        self._L_A = L_A  # JAX array, kept on device for fast prediction
        self._w = w       # JAX array
        self.alpha_ = np.asarray(w)  # numpy copy for external access

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------

    def fit(self, X=None, y=None, noise_level=None, fit_hyperparameters=True,
            validate=True):
        r"""
        Fit the Random Feature GP to training data.

        Parameters
        ----------
        X : (N, d) array or None
            Training inputs. If None, refits to the cached training data without
            adding new points.

        y : (N,) array or None
            Training targets. Must be None iff X is None.

        noise_level : float, (N,) array, or None
            Noise standard deviation(s). If None, reuses the last value stored in
            ``self.alpha`` (as ``sqrt(alpha)``). If an array is passed, a scalar
            mean is used internally for the BLR (per-sample noise support is a
            planned extension).

        fit_hyperparameters : bool or dict (default: True)
            - ``True``: optimise hyperparameters with default settings.
            - ``False``: skip optimisation; just refit the BLR with current params.
            - ``dict``: optimise with custom settings. Recognised keys:

                * ``n_restarts`` (int): number of optimiser restarts.
                * ``start_from_current`` (bool): start first restart from the
                  current (last fitted) hyperparameters.

        validate : bool (default: True)
            If False, skip type/shape checks for speed in tight loops.

        Returns
        -------
        self
        """
        # -- Handle X, y = None: refit without adding data --------------------
        if X is None and y is None:
            if not self._fitted:
                return self  # nothing to do before first fit
            X = self.X_train_
            y = self.y_train_
            # Keep existing alpha (noise) if not overridden
            if noise_level is None:
                noise_sq = np.atleast_1d(np.asarray(self.alpha, dtype=float))
            else:
                noise_sq = np.maximum(np.atleast_1d(np.asarray(noise_level, dtype=float)) ** 2,
                                      1e-12)
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

            # Update noise
            if noise_level is not None:
                nl = np.atleast_1d(np.asarray(noise_level, dtype=float))
                self.alpha = np.maximum(nl ** 2, 1e-12)
                self._noise_level = float(np.sqrt(np.mean(self.alpha)))
                noise_sq = self.alpha
            else:
                noise_sq = np.atleast_1d(np.asarray(self.alpha, dtype=float))

        # -- Adapt M to current dimensionality --------------------------------
        d = X.shape[1]
        if self._n_features_arg is None:
            self._M = max(300, 50 * d)

        # -- Initialise feature map if needed (first call or d changed) -------
        need_new_features = (
            self._W is None
            or self._W.shape[1] != d
            or self._phi_fn is None
        )
        if need_new_features:
            # Initialise length scales for new dimensionality
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
            self._rebuild_feature_map(self._length_scales, rng=rng)

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
                f"'fit_hyperparameters' must be bool or dict, got {type(fit_hyperparameters)}"
            )

        # Effective scalar noise for LML and BLR (use mean if per-sample)
        noise_sq_scalar = float(np.mean(noise_sq))

        # -- Hyperparameter optimisation --------------------------------------
        if do_fit_hp:
            lml = self._fit_hyperparameters(
                X, y,
                noise_sq=noise_sq_scalar,
                start_from_current=hp_kwargs.get("start_from_current", True),
                n_restarts=hp_kwargs.get("n_restarts", None),
                maxiter=hp_kwargs.get("maxiter", None),
            )
            self.log_marginal_likelihood_value_ = lml
        else:
            # Evaluate LML at current hyperparameters (without optimising)
            try:
                log_params_j = jnp.array(
                    np.concatenate([[np.log(self._output_scale)],
                                    np.log(self._length_scales)]),
                    dtype=jnp.float64,
                )
                old_ls_j = jnp.array(self._length_scales, dtype=jnp.float64)
                lml = float(_lml_jax(
                    log_params_j,
                    jnp.array(X, dtype=jnp.float64),
                    jnp.array(y, dtype=jnp.float64),
                    noise_sq_scalar,
                    self._W, self._b, old_ls_j,
                ))
                self.n_eval_loglike += 1
            except Exception:
                lml = -np.inf
            self.log_marginal_likelihood_value_ = lml

        # -- BLR fit with final hyperparameters --------------------------------
        self._blr_fit(X, y, noise_sq_scalar, self._output_scale ** 2)

        # -- Update kernel stubs -----------------------------------------------
        self.kernel_ = _RFFKernelStub(self._output_scale, self._length_scales)
        if not self._fitted:
            # On first fit, also update the "prior" kernel stub
            self.kernel = _RFFKernelStub(self._output_scale, self._length_scales)

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
            Query points.

        return_std : bool (default: False)
            If True, also return the predictive standard deviation.

        return_mean_grad : bool (default: False)
            If True, also return the gradient of the mean w.r.t. X[0].
            Only supported for a single query point (N=1).

        return_std_grad : bool (default: False)
            If True, also return the gradient of the std w.r.t. X[0].
            Requires both ``return_std=True`` and ``return_mean_grad=True``.
            Only for N=1.

        validate : bool (default: True)
            If False, skip input validation.

        Returns
        -------
        list
            Elements (all numpy arrays):

            * ``y_mean`` - shape (N,), always present
            * ``y_std``  - shape (N,), if ``return_std``
            * ``y_mean_grad`` - shape (d,), if ``return_mean_grad``
            * ``y_std_grad``  - shape (d,), if ``return_std_grad``
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        N = X.shape[0]
        self.n_eval += N

        # Input validation
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
            phi_fn = self._phi_fn

            def mean_of_x(x):
                phi_x = phi_fn(x[None, :])[0]  # (M,)
                return jnp.dot(phi_x, w_j)

            def std_of_x(x):
                phi_x = phi_fn(x[None, :])[0]  # (M,)
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
        Predict the predictive standard deviation at query points X.

        Equivalent to ``predict(X, return_std=True)[1]`` but avoids computing
        the mean, saving one matrix-vector multiply.

        Parameters
        ----------
        X : (N, d) array
            Query points.

        validate : bool (default: True)
            Ignored (kept for API compatibility).

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
        Phi_j = self._phi_jax(X_j)                          # (N, M)
        sigma_f_sq = float(self._output_scale ** 2)
        y_var_j = _predict_var_batch(Phi_j, self._L_A, sigma_f_sq)  # (N,)
        y_var = np.asarray(y_var_j)

        neg_mask = y_var < 0.0
        if np.any(neg_mask):
            warnings.warn(
                "Predicted variances smaller than 0. Setting those to 0."
            )
            y_var = np.where(neg_mask, 0.0, y_var)

        return np.sqrt(y_var)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"RandomFeatureGP(M={self._M}, "
            f"output_scale={self._output_scale:.4g}, "
            f"length_scales={np.round(self._length_scales, 4)}, "
            f"noise_level={self._noise_level:.4g}, "
            f"{status})"
        )

    def __str__(self):
        return self.__repr__()
