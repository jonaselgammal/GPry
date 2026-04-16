"""
Experimental non-stationary GP using a Gibbs kernel.

This module is not part of the conservative GPry baseline. It is kept for
opt-in tests of spatially varying length scales.

The Gibbs kernel allows different length scales in different regions of input
space. Near the likelihood peak, short length scales capture detail; far away,
long length scales enforce smoothness. This directly addresses the "snap to mode"
convergence issue of stationary GPs.

The length-scale function is parameterized as:
    l(x) = l_base * softplus(W @ x + b)
where W is (d, d) or low-rank (d, r) and b is (d,).

The Gibbs kernel between points x_i and x_j is:
    K(x_i, x_j) = output_scale^2 * prod_k sqrt(2*l_ik*l_jk / (l_ik^2 + l_jk^2))
                  * exp(-sum_k (x_ik - x_jk)^2 / (l_ik^2 + l_jk^2))

where l_i = l(x_i) is a d-dimensional vector of length scales at x_i.

All computations are in JAX for autodiff through the full pipeline.
"""

import warnings
from typing import Mapping

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from scipy.optimize import minimize as scipy_minimize

jax.config.update("jax_enable_x64", True)

# Pad training arrays to multiples of this size to avoid JIT recompilation
# when N grows by 1-2 points each iteration.
_PAD_CHUNK = 32


def _pad_for_jit(X, y, alpha_noise, chunk_size=_PAD_CHUNK):
    """Pad training arrays to next multiple of chunk_size for stable JIT shapes.

    Padded X entries use the training mean (safe for Gibbs length-scale
    function which passes X through softplus(W@x + b) — extreme values
    would cause overflow/underflow).  Padded y = 0 and noise = 1e10
    effectively decouple the fake points: alpha_pad ≈ 0 and the Schur
    complement correction to the real block is O(1e-10).
    """
    n = X.shape[0]
    n_padded = ((n + chunk_size - 1) // chunk_size) * chunk_size
    # Always convert noise to vector for consistent JIT traces
    if isinstance(alpha_noise, (int, float)):
        noise_vec = np.full(n, alpha_noise)
    else:
        noise_vec = np.asarray(alpha_noise)
    if n_padded == n:
        return (jnp.array(X, dtype=jnp.float64),
                jnp.array(y, dtype=jnp.float64),
                jnp.array(noise_vec, dtype=jnp.float64))
    n_pad = n_padded - n
    d = X.shape[1]
    x_mean = np.mean(X, axis=0, keepdims=True)  # (1, d)
    X_out = np.vstack([X, np.tile(x_mean, (n_pad, 1))])
    y_out = np.concatenate([y, np.zeros(n_pad)])
    noise_out = np.concatenate([noise_vec, np.full(n_pad, 1e10)])
    return (jnp.array(X_out, dtype=jnp.float64),
            jnp.array(y_out, dtype=jnp.float64),
            jnp.array(noise_out, dtype=jnp.float64))


# ---------------------------------------------------------------------------
# Gibbs kernel (JAX, fully differentiable)
# ---------------------------------------------------------------------------

def _gibbs_kernel_matrix(X1, X2, l_base, W, b_vec, output_scale_sq):
    """Compute the Gibbs kernel matrix.

    Parameters
    ----------
    X1 : (n1, d)
    X2 : (n2, d)
    l_base : (d,) - base length scales
    W : (d, r) - weight matrix for length-scale function
    b_vec : (d,) - bias for length-scale function (renamed to avoid clash)
    output_scale_sq : scalar

    Returns
    -------
    K : (n1, n2)
    """
    # Compute spatially varying length scales: l(x) = l_base * softplus(x @ W_padded + b)
    # W is (d, r). Pad to (d, d) so the output is always (n, d).
    d = l_base.shape[0]
    W_padded = jnp.pad(W, ((0, 0), (0, d - W.shape[1])))  # (d, d), zero-padded
    l1 = l_base * jax.nn.softplus(X1 @ W_padded + b_vec)  # (n1, d)
    l2 = l_base * jax.nn.softplus(X2 @ W_padded + b_vec)  # (n2, d)

    # Gibbs kernel: product over dimensions
    # For each pair (i,j) and dimension k:
    #   factor_k = sqrt(2 * l1[i,k] * l2[j,k] / (l1[i,k]^2 + l2[j,k]^2))
    #   exp_k = exp(-(x1[i,k] - x2[j,k])^2 / (l1[i,k]^2 + l2[j,k]^2))

    # Expand for broadcasting: (n1, 1, d) and (1, n2, d)
    l1_exp = l1[:, None, :]  # (n1, 1, d)
    l2_exp = l2[None, :, :]  # (1, n2, d)
    x1_exp = X1[:, None, :]  # (n1, 1, d)
    x2_exp = X2[None, :, :]  # (1, n2, d)

    l1_sq = l1_exp ** 2
    l2_sq = l2_exp ** 2
    sum_l_sq = l1_sq + l2_sq  # (n1, n2, d)

    # Prefactor: product over d of sqrt(2 * l1 * l2 / (l1^2 + l2^2))
    prefactor = jnp.prod(
        jnp.sqrt(2.0 * l1_exp * l2_exp / (sum_l_sq + 1e-30)),
        axis=-1
    )  # (n1, n2)

    # Exponential: exp(-sum_k (x1_k - x2_k)^2 / (l1_k^2 + l2_k^2))
    diff_sq = (x1_exp - x2_exp) ** 2  # (n1, n2, d)
    exp_term = jnp.exp(-jnp.sum(diff_sq / (sum_l_sq + 1e-30), axis=-1))  # (n1, n2)

    return output_scale_sq * prefactor * exp_term


def _gibbs_kernel_diag(X, l_base, W, b_vec, output_scale_sq):
    """Diagonal of the Gibbs kernel (self-covariance)."""
    # K(x,x) = output_scale^2 * prod_k sqrt(2*l*l / (l^2 + l^2))
    #         = output_scale^2 * prod_k sqrt(2*l^2 / 2*l^2)
    #         = output_scale^2 * 1.0
    return output_scale_sq * jnp.ones(X.shape[0])


# ---------------------------------------------------------------------------
# JIT-compiled prediction helpers (module-level for persistent JIT cache)
# ---------------------------------------------------------------------------

@jit
def _jit_predict_mean(x, X_train, alpha, l_base, W, b, osc):
    """Mean prediction at a single point (JIT-cached)."""
    K_trans = _gibbs_kernel_matrix(x[None, :], X_train, l_base, W, b, osc)
    return (K_trans @ alpha)[0]


@jit
def _jit_predict_mean_and_grad(x, X_train, alpha, l_base, W, b, osc):
    """Mean + gradient at a single point (JIT-cached)."""
    def _mean(x_):
        K_trans = _gibbs_kernel_matrix(x_[None, :], X_train, l_base, W, b, osc)
        return (K_trans @ alpha)[0]
    return jax.value_and_grad(_mean)(x)


@jit
def _jit_predict_std_and_grad(x, X_train, V, l_base, W, b, osc, alpha_noise):
    """Std + gradient at a single point (JIT-cached)."""
    def _std(x_):
        K_trans = _gibbs_kernel_matrix(x_[None, :], X_train, l_base, W, b, osc)
        k_diag = osc + alpha_noise
        M = V @ K_trans[0]
        var = k_diag - jnp.dot(M, M)
        var = jnp.maximum(var, 1e-30)
        return jnp.sqrt(var)
    return jax.value_and_grad(_std)(x)


# ---------------------------------------------------------------------------
# GibbsGPR class
# ---------------------------------------------------------------------------

class GibbsGPR:
    """Gaussian Process Regressor with a non-stationary Gibbs kernel.

    Parameters
    ----------
    kernel : str
        Ignored (always uses Gibbs kernel), kept for interface compatibility.
    output_scale_prior : list
        [lower, upper] bounds for output scale.
    length_scale_prior : array, shape (d, 2)
        Per-dimension [lower, upper] bounds for base length scales.
    noise_level : float
        Default noise standard deviation.
    noise_fixed : bool
        Whether noise is fixed or optimized.
    rank : int or None
        Rank of the W matrix. If None, uses min(d, 3).
        Lower rank = fewer parameters = less overfitting risk.
    l2_reg : float
        L2 regularization on W matrix to prevent overfitting.
    n_restarts_optimizer : int
        Number of optimizer restarts.
    random_state : int or None
        Random seed.
    use_jax : bool
        Ignored (always uses JAX), kept for interface compatibility.
    """

    def __init__(
        self,
        kernel="RBF",
        output_scale_prior=None,
        length_scale_prior=None,
        noise_level=1e-2,
        noise_fixed=True,
        rank=None,
        l2_reg=1.0,
        n_restarts_optimizer=5,
        random_state=None,
        use_jax=True,
    ):
        if output_scale_prior is None:
            output_scale_prior = [1e-2, 1e3]
        if length_scale_prior is None:
            raise ValueError("length_scale_prior must be provided")
        self._output_scale_prior = np.array(output_scale_prior)
        self._length_scale_prior = np.atleast_2d(length_scale_prior)
        # Clamp lower bound: too-small length scales cause diagonal kernel matrices
        # and degenerate GP predictions (pure interpolation, zero gradients).
        # For NormalizeBounds data in [0,1], 0.02 is a reasonable floor.
        self._length_scale_prior[:, 0] = np.maximum(
            self._length_scale_prior[:, 0], 0.02
        )
        self._d = len(self._length_scale_prior)
        self._rank = rank if rank is not None else min(self._d, 3)
        self._l2_reg = l2_reg
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.is_noise_in_kernel = not noise_fixed
        self._noise_level_init = noise_level
        self.copy_X_train = True
        self.n_eval = 0
        self.n_eval_loglike = 0
        self._fitted = False
        # Kernel stub for interface compatibility
        self.kernel = None
        self.kernel_ = None
        # GP state
        self.X_train_ = None
        self.y_train_ = None
        self.alpha = noise_level ** 2
        self.log_marginal_likelihood_value_ = None
        self.L_ = None
        self.V_ = None
        self.alpha_ = None
        self._X_train_padded = None
        # Gibbs kernel parameters
        self._output_scale_sq = float(
            np.sqrt(output_scale_prior[0] * output_scale_prior[1]) ** 2
        )
        self._l_base = np.sqrt(
            self._length_scale_prior[:, 0] * self._length_scale_prior[:, 1]
        )
        self._W = np.zeros((self._d, self._rank))
        # Initialize b so softplus(b) = 1, i.e. l(x) ≈ l_base at initialization
        self._b = np.full(self._d, np.log(np.exp(1.0) - 1.0))  # softplus_inv(1) ≈ 0.541
        self._alpha_noise = noise_level ** 2
        # JAX accel stub
        self._jax_accel = None

    @property
    def scales(self):
        """Return (output_scale, length_scales) for interface compatibility."""
        return np.sqrt(self._output_scale_sq), self._l_base

    @property
    def noise_level(self):
        return np.sqrt(self._alpha_noise)

    # ------------------------------------------------------------------
    # Parameter packing/unpacking
    # ------------------------------------------------------------------

    def _pack_theta(self):
        """Pack all hyperparameters into a flat vector (log space for scales)."""
        parts = [
            np.log(np.array([self._output_scale_sq])),  # 1
            np.log(self._l_base),                        # d
            self._W.ravel(),                             # d * rank
            self._b,                                     # d
        ]
        return np.concatenate(parts)

    def _unpack_theta(self, theta):
        """Unpack flat theta vector into individual parameters."""
        idx = 0
        log_osc = theta[idx]; idx += 1
        log_l_base = theta[idx:idx + self._d]; idx += self._d
        W_flat = theta[idx:idx + self._d * self._rank]; idx += self._d * self._rank
        b = theta[idx:idx + self._d]; idx += self._d
        return {
            "output_scale_sq": jnp.exp(log_osc),
            "l_base": jnp.exp(log_l_base),
            "W": W_flat.reshape(self._d, self._rank),
            "b": b,
        }

    def _theta_bounds(self):
        """Bounds for the flattened theta vector."""
        bounds = []
        # log(output_scale_sq)
        bounds.append((
            np.log(self._output_scale_prior[0] ** 2),
            np.log(self._output_scale_prior[1] ** 2),
        ))
        # log(l_base) per dimension
        for i in range(self._d):
            bounds.append((
                np.log(self._length_scale_prior[i, 0]),
                np.log(self._length_scale_prior[i, 1]),
            ))
        # W entries: unconstrained but bounded for stability
        for _ in range(self._d * self._rank):
            bounds.append((-5.0, 5.0))
        # b entries: bounded to keep softplus(b) in a reasonable range
        # softplus(-3) ≈ 0.049, softplus(5) ≈ 5.007
        for _ in range(self._d):
            bounds.append((-3.0, 5.0))
        return bounds

    # ------------------------------------------------------------------
    # LML computation
    # ------------------------------------------------------------------

    def _compute_lml(self, theta, X_train, y_train, alpha_noise):
        """Compute log marginal likelihood (JAX, differentiable)."""
        params = self._unpack_theta(theta)
        N = X_train.shape[0]

        K = _gibbs_kernel_matrix(
            X_train, X_train,
            params["l_base"], params["W"], params["b"],
            params["output_scale_sq"],
        )
        K = K + alpha_noise * jnp.eye(N)

        # Cholesky
        L = jnp.linalg.cholesky(K)
        alpha_vec = jax.scipy.linalg.cho_solve((L, True), y_train)

        # LML = -0.5 * y.T @ alpha - sum(log(diag(L))) - N/2 * log(2*pi)
        lml = (
            -0.5 * jnp.dot(y_train, alpha_vec)
            - jnp.sum(jnp.log(jnp.diag(L)))
            - 0.5 * N * jnp.log(2.0 * jnp.pi)
        )

        # L2 regularization on W
        lml = lml - self._l2_reg * jnp.sum(params["W"] ** 2)

        return lml

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X=None, y=None, noise_level=None, fit_hyperparameters=True,
            validate=True):
        """Fit the Gibbs GP to training data.

        Parameters
        ----------
        X : array, shape (n, d) or None
        y : array, shape (n,) or None
        noise_level : float or array or None
        fit_hyperparameters : bool or dict
        validate : bool

        Returns
        -------
        self
        """
        if X is not None:
            self.X_train_ = np.copy(X) if self.copy_X_train else X
        if y is not None:
            self.y_train_ = np.copy(y)
        if noise_level is not None:
            if np.iterable(noise_level):
                self.alpha = np.asarray(noise_level) ** 2
            else:
                self.alpha = noise_level ** 2
            self._alpha_noise = self.alpha
        if X is None and y is None:
            return self

        # Optimize hyperparameters
        if fit_hyperparameters is True:
            fit_hyperparameters = {}
        if isinstance(fit_hyperparameters, Mapping):
            self._fit_hyperparameters(**fit_hyperparameters)
        elif fit_hyperparameters is not False:
            self._fit_hyperparameters()

        # Precompute Cholesky with current parameters
        self._precompute()
        self._fitted = True
        return self

    def _fit_hyperparameters(self, n_restarts=None, start_from_current=True,
                             hyperparameter_bounds=None, maxiter=None, **kwargs):
        """Optimize kernel hyperparameters via LML maximization."""
        if n_restarts is None:
            n_restarts = self.n_restarts_optimizer
        if maxiter is None:
            maxiter = 200
        if n_restarts <= 0:
            X_j, y_j, noise_j = _pad_for_jit(
                self.X_train_, self.y_train_, self._alpha_noise
            )
            self.log_marginal_likelihood_value_ = float(self._compute_lml(
                jnp.array(self._pack_theta()), X_j, y_j, noise_j,
            ))
            return self.log_marginal_likelihood_value_

        # Pad arrays to stable shapes to avoid JIT recompilation each iteration
        X_j, y_j, noise_j = _pad_for_jit(
            self.X_train_, self.y_train_, self._alpha_noise
        )

        lml_and_grad = jit(value_and_grad(self._compute_lml))
        bounds = self._theta_bounds()

        def obj_func(theta_np):
            theta_j = jnp.array(theta_np, dtype=jnp.float64)
            try:
                lml, grad = lml_and_grad(theta_j, X_j, y_j, noise_j)
                return -float(lml), -np.array(grad, dtype=np.float64)
            except Exception:
                return 1e10, np.zeros_like(theta_np)

        best_lml = -np.inf
        best_theta = self._pack_theta()

        rng = np.random.RandomState(
            self.random_state if isinstance(self.random_state, int) else 42
        )

        for i in range(n_restarts):
            if i == 0 and start_from_current and self._fitted:
                theta0 = self._pack_theta()
            else:
                # Random initialization
                theta0 = np.array([
                    rng.uniform(lo, hi) for lo, hi in bounds
                ])
                # Keep W near zero for stability — large W causes
                # degenerate length scales and diagonal kernel matrices
                w_start = 1 + self._d  # after log_osc and log_l_base
                w_end = w_start + self._d * self._rank
                theta0[w_start:w_end] *= 0.01

            try:
                result = scipy_minimize(
                    obj_func, theta0, method='L-BFGS-B', jac=True,
                    bounds=bounds, options={"maxiter": maxiter},
                )
                lml_val = -result.fun
                if lml_val > best_lml:
                    best_lml = lml_val
                    best_theta = result.x
            except Exception:
                continue

        # Update parameters from best theta
        params = self._unpack_theta(jnp.array(best_theta))
        self._output_scale_sq = float(params["output_scale_sq"])
        self._l_base = np.array(params["l_base"])
        self._W = np.array(params["W"])
        self._b = np.array(params["b"])
        self.log_marginal_likelihood_value_ = best_lml
        self.kernel_ = (
            f"Gibbs(output_scale={np.sqrt(self._output_scale_sq):.3g}, "
            f"l_base={np.array2string(self._l_base, precision=3)}, rank={self._rank})"
        )
        return best_lml

    def _precompute(self):
        """Precompute Cholesky, V, alpha_ for predictions.

        Uses padded arrays so that JIT-compiled kernel/Cholesky functions
        keep a stable shape across iterations (avoids 300ms recompilation
        each time N grows by 1-2 points).
        """
        X_j, y_j, noise_j = _pad_for_jit(
            self.X_train_, self.y_train_, self._alpha_noise
        )
        n_pad = X_j.shape[0]

        K = _gibbs_kernel_matrix(
            X_j, X_j,
            jnp.array(self._l_base), jnp.array(self._W),
            jnp.array(self._b), jnp.float64(self._output_scale_sq),
        )
        K = K + noise_j * jnp.eye(n_pad)

        L = jnp.linalg.cholesky(K)
        V = jax.scipy.linalg.solve_triangular(L, jnp.eye(n_pad), lower=True)
        alpha_vec = jax.scipy.linalg.cho_solve((L, True), y_j)

        self.L_ = np.array(L)
        self.V_ = np.array(V)
        self.alpha_ = np.array(alpha_vec)
        # Store padded X_train and JAX parameter arrays for predict
        # (same shape → no JIT recompile; cached arrays → no re-conversion)
        self._X_train_padded = X_j
        self._l_base_j = jnp.array(self._l_base)
        self._W_j = jnp.array(self._W)
        self._b_j = jnp.array(self._b)
        self._osc_j = jnp.float64(self._output_scale_sq)
        self._alpha_j = jnp.array(self.alpha_)
        self._V_j = jnp.array(self.V_)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X, return_std=False, return_mean_grad=False,
                return_std_grad=False, validate=True):
        """Predict GP mean (and optionally std, gradients).

        Returns a list for interface compatibility with GPR.
        """
        X = np.atleast_2d(X)
        result = []

        if return_mean_grad or return_std_grad:
            return self._predict_with_grads(X, return_std, return_mean_grad,
                                            return_std_grad)

        X_j = jnp.array(X, dtype=jnp.float64)
        # Use cached padded JAX arrays (set in _precompute)
        X_train_j = self._X_train_padded
        l_base_j = self._l_base_j
        W_j = self._W_j
        b_j = self._b_j
        osc = self._osc_j

        K_trans = _gibbs_kernel_matrix(X_j, X_train_j, l_base_j, W_j, b_j, osc)
        y_mean = np.array(K_trans @ self._alpha_j)
        result.append(y_mean)

        if return_std:
            V_j = self._V_j
            k_diag = _gibbs_kernel_diag(X_j, l_base_j, W_j, b_j, osc)
            alpha_noise = (
                self._alpha_noise if isinstance(self._alpha_noise, float)
                else np.mean(self._alpha_noise)
            )
            k_diag = k_diag + alpha_noise
            M = V_j @ K_trans.T
            y_var = k_diag - jnp.sum(M ** 2, axis=0)
            y_var = jnp.maximum(y_var, 1e-30)
            result.append(np.array(jnp.sqrt(y_var)))

        return result

    def predict_std(self, X, validate=True):
        """Predict standard deviation only."""
        return self.predict(X, return_std=True, validate=validate)[1]

    def _predict_with_grads(self, X, return_std, return_mean_grad, return_std_grad):
        """Predict with gradients using module-level JIT-compiled functions.

        Using module-level @jit functions instead of per-call closures ensures
        the JIT cache persists across calls.  With padded training arrays the
        shapes are stable, so compilation happens only once per chunk size.
        """
        if X.shape[0] != 1:
            raise ValueError("Gradient prediction only supported for single points")
        x = jnp.array(X[0], dtype=jnp.float64)
        # Use cached JAX arrays from _precompute
        X_train_j = self._X_train_padded
        alpha_j = self._alpha_j
        l_base_j = self._l_base_j
        W_j = self._W_j
        b_j = self._b_j
        osc = self._osc_j
        alpha_noise = (
            jnp.float64(self._alpha_noise)
            if isinstance(self._alpha_noise, float) else
            jnp.float64(np.mean(self._alpha_noise))
        )

        result = []

        if return_mean_grad:
            y_mean, grad_mean = _jit_predict_mean_and_grad(
                x, X_train_j, alpha_j, l_base_j, W_j, b_j, osc)
            result.append(np.array([float(y_mean)]))
        else:
            y_mean = _jit_predict_mean(
                x, X_train_j, alpha_j, l_base_j, W_j, b_j, osc)
            result.append(np.array([float(y_mean)]))

        if return_std or return_std_grad:
            V_j = self._V_j
            y_std, grad_std = _jit_predict_std_and_grad(
                x, X_train_j, V_j, l_base_j, W_j, b_j, osc, alpha_noise)
            if return_std:
                result.append(np.array([float(y_std)]))
            if return_mean_grad:
                result.append(np.array(grad_mean).reshape(1, -1))
            if return_std_grad:
                result.append(np.array(grad_std).reshape(1, -1))
        else:
            if return_mean_grad:
                result.append(np.array(grad_mean).reshape(1, -1))

        return result

    # ------------------------------------------------------------------
    # Log marginal likelihood (for external callers)
    # ------------------------------------------------------------------

    def log_marginal_likelihood(self, theta=None, eval_gradient=False,
                                clone_kernel=True):
        """Compute LML at given theta or current parameters."""
        if theta is None:
            theta_j = jnp.array(self._pack_theta())
        else:
            theta_j = jnp.array(theta, dtype=jnp.float64)
        X_j, y_j, noise_j = _pad_for_jit(
            self.X_train_, self.y_train_, self._alpha_noise
        )
        if eval_gradient:
            lml, grad = value_and_grad(self._compute_lml)(
                theta_j, X_j, y_j, noise_j)
            return float(lml), np.array(grad)
        else:
            return float(self._compute_lml(theta_j, X_j, y_j, noise_j))
