"""
JAX-accelerated GP operations for GPry.

Provides JIT-compiled implementations of the core GP operations:
- Kernel matrix computation (RBF and Matern)
- Cholesky decomposition and precomputation
- GP prediction (mean and std)
- Log marginal likelihood
- Automatic differentiation for LML gradients (hyperparameter optimization)
- Automatic differentiation for predict gradients (acquisition functions)

These functions are designed to be drop-in replacements for the scipy/numpy
operations in gpr.py, providing significant speedups through JIT compilation,
especially for repeated calls with the same array shapes (which is typical
in acquisition and hyperparameter optimization).
"""

import warnings
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad, vmap
from jax.scipy.linalg import cho_solve, solve_triangular
import numpy as np

# Ensure JAX uses 64-bit floats for numerical accuracy
jax.config.update("jax_enable_x64", True)

# Small epsilon added inside sqrt(dist^2 + eps) for Matern kernels
# to ensure differentiability at zero distance. Negligible effect on values.
_SQRT_EPS = 1e-30


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

@jit
def _rbf_kernel_matrix(X1, X2, length_scale):
    """RBF kernel matrix: K_ij = exp(-0.5 * sum((x1_i - x2_j)^2 / l^2))."""
    # Scaled differences: (n1, 1, d) - (1, n2, d) / l
    scaled_X1 = X1 / length_scale
    scaled_X2 = X2 / length_scale
    # Squared distances via expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    sq_X1 = jnp.sum(scaled_X1 ** 2, axis=1)
    sq_X2 = jnp.sum(scaled_X2 ** 2, axis=1)
    dist_sq = sq_X1[:, None] + sq_X2[None, :] - 2.0 * scaled_X1 @ scaled_X2.T
    # Clamp to avoid negative values from floating point
    dist_sq = jnp.maximum(dist_sq, 0.0)
    return jnp.exp(-0.5 * dist_sq)


@jit
def _rbf_kernel_diag(X, length_scale):
    """Diagonal of RBF kernel (always 1)."""
    return jnp.ones(X.shape[0])


@jit
def _matern52_kernel_matrix(X1, X2, length_scale):
    """Matern 5/2 kernel matrix."""
    scaled_X1 = X1 / length_scale
    scaled_X2 = X2 / length_scale
    sq_X1 = jnp.sum(scaled_X1 ** 2, axis=1)
    sq_X2 = jnp.sum(scaled_X2 ** 2, axis=1)
    dist_sq = sq_X1[:, None] + sq_X2[None, :] - 2.0 * scaled_X1 @ scaled_X2.T
    dist_sq = jnp.maximum(dist_sq, 0.0)
    # Use safe sqrt for differentiability at zero distance
    r = jnp.sqrt(dist_sq + _SQRT_EPS)
    sqrt5_r = jnp.sqrt(5.0) * r
    return (1.0 + sqrt5_r + 5.0 / 3.0 * dist_sq) * jnp.exp(-sqrt5_r)


@jit
def _matern32_kernel_matrix(X1, X2, length_scale):
    """Matern 3/2 kernel matrix."""
    scaled_X1 = X1 / length_scale
    scaled_X2 = X2 / length_scale
    sq_X1 = jnp.sum(scaled_X1 ** 2, axis=1)
    sq_X2 = jnp.sum(scaled_X2 ** 2, axis=1)
    dist_sq = sq_X1[:, None] + sq_X2[None, :] - 2.0 * scaled_X1 @ scaled_X2.T
    dist_sq = jnp.maximum(dist_sq, 0.0)
    r = jnp.sqrt(dist_sq + _SQRT_EPS)
    sqrt3_r = jnp.sqrt(3.0) * r
    return (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r)


@jit
def _matern12_kernel_matrix(X1, X2, length_scale):
    """Matern 1/2 (exponential) kernel matrix."""
    scaled_X1 = X1 / length_scale
    scaled_X2 = X2 / length_scale
    sq_X1 = jnp.sum(scaled_X1 ** 2, axis=1)
    sq_X2 = jnp.sum(scaled_X2 ** 2, axis=1)
    dist_sq = sq_X1[:, None] + sq_X2[None, :] - 2.0 * scaled_X1 @ scaled_X2.T
    dist_sq = jnp.maximum(dist_sq, 0.0)
    r = jnp.sqrt(dist_sq + _SQRT_EPS)
    return jnp.exp(-r)


def get_kernel_fn(kernel_type, nu=None):
    """Return the appropriate JIT-compiled kernel function.

    Parameters
    ----------
    kernel_type : str
        "rbf" or "matern"
    nu : float, optional
        Matern smoothness parameter (0.5, 1.5, 2.5). Only used for matern.

    Returns
    -------
    kernel_fn : callable
        JIT-compiled kernel function(X1, X2, length_scale) -> K
    """
    if kernel_type == "rbf":
        return _rbf_kernel_matrix
    elif kernel_type == "matern":
        if nu is None:
            nu = 2.5
        if np.isclose(nu, 0.5):
            return _matern12_kernel_matrix
        elif np.isclose(nu, 1.5):
            return _matern32_kernel_matrix
        elif np.isclose(nu, 2.5):
            return _matern52_kernel_matrix
        else:
            raise ValueError(
                f"Matern kernel with nu={nu} not supported for JAX acceleration. "
                "Supported values: 0.5, 1.5, 2.5"
            )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


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
# JAX runtime bundle that wraps GP state
# ---------------------------------------------------------------------------

class JaxRuntimeBundle:
    """JAX runtime bundle for a fitted GP model.

    The bundle holds the JAX-native state derived from a fitted GP and the
    compiled helper functions that operate on that state. It is intentionally
    separate from the canonical sklearn-style GP object so callers can treat
    it as an optional acceleration layer rather than part of the fitted model
    semantics.

    Usage
    -----
    After fitting a GaussianProcessRegressor:

        bundle = JaxRuntimeBundle()
        bundle.update_from_gpr(gpr)  # caches JAX arrays
        y_mean = bundle.predict_mean(X_new)
        y_std = bundle.predict_std(X_new)
        y_mean, y_std = bundle.predict_mean_std(X_new)
    """

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

    def update_from_gpr(self, gpr):
        """Extract kernel parameters and cached arrays from a fitted GPR.

        Parameters
        ----------
        gpr : GaussianProcessRegressor
            A fitted GPR instance (must have X_train_, y_train_, L_, V_, alpha_).
        """
        from gpry.kernels import RBF, Matern

        if not hasattr(gpr, "X_train_") or gpr.L_ is None:
            self._ready = False
            return

        # Detect kernel type and extract parameters
        kernel = gpr.kernel_
        # Handle possible WhiteKernel sum: (C * RBF) + WhiteKernel
        self._white_noise_level = 0.0
        if hasattr(kernel, 'k1') and hasattr(kernel.k1, 'k1'):
            # Sum kernel: k1 = C * RBF/Matern, k2 = WhiteKernel
            product_kernel = kernel.k1
            # k2 is WhiteKernel - its diag contribution is noise_level
            if hasattr(kernel.k2, 'noise_level'):
                self._white_noise_level = float(kernel.k2.noise_level)
        else:
            product_kernel = kernel

        # product_kernel should be ConstantKernel * (RBF or Matern)
        output_scale_sq = product_kernel.k1.constant_value
        length_kernel = product_kernel.k2

        # Determine kernel function — only rebuild JIT functions if kernel type
        # or dimensionality changes (avoids expensive JIT recompilation)
        new_has_white_noise = self._white_noise_level > 0.0
        new_n_dims = gpr.X_train_.shape[1]
        need_rebuild = (self._lml_value_and_grad_fn is None
                        or new_has_white_noise != self._has_white_noise
                        or new_n_dims != self._n_dims)

        if isinstance(length_kernel, RBF):
            new_kernel_fn = _rbf_kernel_matrix
        elif isinstance(length_kernel, Matern):
            nu = length_kernel.nu
            new_kernel_fn = get_kernel_fn("matern", nu)
        else:
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

        # Build gradient functions only when kernel type/dims/noise config changes
        if need_rebuild:
            lml_fn = _make_lml_fn(self._kernel_fn, self._has_white_noise,
                                  self._n_dims)
            self._lml_value_and_grad_fn = value_and_grad(lml_fn, argnums=0)

            predict_mean_single = _make_predict_mean_single_fn(self._kernel_fn)
            predict_std_single = _make_predict_std_single_fn(self._kernel_fn)
            self._predict_mean_single_fn = predict_mean_single
            self._predict_std_single_fn = predict_std_single
            self._predict_mean_grad_fn = jit(grad(predict_mean_single, argnums=0))
            self._predict_std_grad_fn = jit(grad(predict_std_single, argnums=0))
            # Static acq function: compiled once, reused for all zeta/baseline values
            self._neg_acq_fn = _build_neg_acq_fn(predict_mean_single,
                                                  predict_std_single)
            # Invalidate cached solver since the function changed
            self._cached_acq_solver = None
            self._cached_acq_solver_fn_id = None

        self._ready = True

    def _compute_K_trans(self, X_new):
        """Compute cross-covariance K(X_new, X_train) with output scale."""
        X_new_jax = jnp.array(X_new, dtype=jnp.float64) if not isinstance(X_new, jnp.ndarray) else X_new
        K_trans = self._kernel_fn(X_new_jax, self._X_train, self._length_scale)
        return self._output_scale_sq * K_trans

    def _compute_k_diag(self, X_new):
        """Compute prior variance diagonal (output_scale^2 + white_noise for stationary kernels)."""
        n = X_new.shape[0] if hasattr(X_new, 'shape') else len(X_new)
        return (self._output_scale_sq + self._white_noise_level) * jnp.ones(n)

    def predict_mean_std_jax(self, X_new):
        """Predict GP mean and std at X_new. Returns JAX arrays (no copy)."""
        X_new_jax = (X_new if isinstance(X_new, jnp.ndarray)
                     else jnp.array(X_new, dtype=jnp.float64))
        K_trans = self._compute_K_trans(X_new_jax)
        k_diag = self._compute_k_diag(X_new_jax)
        y_mean, y_var = _gp_predict_mean_and_var(
            K_trans, self._alpha_, self._V_, k_diag
        )
        return y_mean, jnp.sqrt(y_var)

    def predict_mean_jax(self, X_new):
        """Predict GP mean at X_new. Returns JAX array (no copy)."""
        X_new_jax = (X_new if isinstance(X_new, jnp.ndarray)
                     else jnp.array(X_new, dtype=jnp.float64))
        K_trans = self._compute_K_trans(X_new_jax)
        return _gp_predict_mean(K_trans, self._alpha_)

    def predict_std_jax(self, X_new):
        """Predict GP std at X_new. Returns JAX array (no copy)."""
        X_new_jax = (X_new if isinstance(X_new, jnp.ndarray)
                     else jnp.array(X_new, dtype=jnp.float64))
        K_trans = self._compute_K_trans(X_new_jax)
        k_diag = self._compute_k_diag(X_new_jax)
        y_var = _gp_predict_var(K_trans, self._V_, k_diag)
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
        alpha_noise = jnp.float64(self._alpha_noise) if isinstance(
            self._alpha_noise, float) else self._alpha_noise

        lml, grad_theta = self._lml_value_and_grad_fn(
            theta_jax, self._X_train, self._y_train, alpha_noise
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
            x_jax, self._X_train, self._alpha_,
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
            x_jax, self._X_train, self._V_,
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
            x_jax, self._X_train, self._alpha_,
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
            x_jax, self._X_train, self._alpha_, self._length_scale, osc))
        std = float(self._predict_std_single_fn(
            x_jax, self._X_train, self._V_, self._length_scale, osc, wnl))
        mean_grad = np.asarray(self._predict_mean_grad_fn(
            x_jax, self._X_train, self._alpha_, self._length_scale, osc))
        std_grad = np.asarray(self._predict_std_grad_fn(
            x_jax, self._X_train, self._V_, self._length_scale, osc, wnl))
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
        K_train = self._kernel_fn(
            self._X_train, self._X_train, self._length_scale)
        K_train = self._output_scale_sq * K_train
        if self._white_noise_level > 0:
            n = K_train.shape[0]
            K_train = K_train + self._white_noise_level * jnp.eye(n)
        alpha_noise = (jnp.float64(self._alpha_noise) if isinstance(
            self._alpha_noise, (float, int)) else self._alpha_noise)
        L, V, alpha_ = _gp_fit_precompute(K_train, self._y_train, alpha_noise)
        self._L_ = L
        self._V_ = V
        self._alpha_ = alpha_
        return L, V, alpha_

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
        if y_train is not None:
            self._y_train = jnp.array(y_train, dtype=jnp.float64)

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
        # to avoid creating a new closure/recompilation each call.
        lml_fn = self._lml_value_and_grad_fn  # from update_from_gpr

        @jit
        def neg_lml(theta, X_train, y_train, alpha_noise):
            lml_val = lml_fn(theta, X_train, y_train, alpha_noise)[0]
            return -lml_val

        # Build/cache solver keyed on the neg_lml function identity
        fn_id = id(lml_fn)
        if (not hasattr(self, '_cached_hyper_solver')
                or self._cached_hyper_solver is None
                or self._cached_hyper_solver_fn_id != fn_id):
            for name in ("jaxopt", "jaxopt._src", "absl"):
                logging.getLogger(name).setLevel(logging.ERROR)
            self._cached_hyper_solver = jaxopt.LBFGSB(
                fun=neg_lml, maxiter=100, tol=1e-6)
            self._cached_hyper_solver_fn_id = fn_id

        solver = self._cached_hyper_solver
        X_train = self._X_train
        y_train = self._y_train
        alpha_noise = (jnp.float64(self._alpha_noise) if isinstance(
            self._alpha_noise, (float, int)) else self._alpha_noise)
        data_args = (X_train, y_train, alpha_noise)

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
                    except Exception:
                        continue
                start_points.append(theta0)

        for theta0 in start_points:

            try:
                result = solver.run(theta0, *data_args, bounds=bounds_jax)
                val = float(result.state.value)
                if val < best_val:
                    best_val = val
                    best_theta = np.asarray(result.params)
            except Exception:
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
            self._cached_acq_solver = jaxopt.LBFGSB(
                fun=self._neg_acq_fn, maxiter=50, tol=1e-5)
            self._cached_acq_solver_fn_id = fn_id

        x0_jax = jnp.array(x0, dtype=jnp.float64)
        bounds_jax = (
            jnp.array(bounds[:, 0], dtype=jnp.float64),
            jnp.array(bounds[:, 1], dtype=jnp.float64),
        )
        # Pack the per-call varying parameters as JAX scalars
        # These are traced (not concrete), so no recompilation
        acq_args = (
            self._X_train, self._alpha_, self._V_, self._length_scale,
            jnp.float64(self._output_scale_sq),
            jnp.float64(self._white_noise_level),
            jnp.float64(zeta), jnp.float64(noise_var), jnp.float64(baseline),
        )

        try:
            result = self._cached_acq_solver.run(
                x0_jax, *acq_args, bounds=bounds_jax)
            x_opt = np.asarray(result.params)
            func_min = float(result.state.value)
        except Exception:
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


# Backward-compatible alias while callers are migrated to the explicit
# runtime-bundle terminology.
JaxGPAccelerator = JaxRuntimeBundle
