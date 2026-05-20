"""JAX-native kernel math for the GPry JAX GP backend.

This module owns the JIT-compiled kernel-matrix implementations used by the
JAX backend (``gpr_jax.py`` / ``gpr_jax_linalg.py``). Each function takes
``(X1, X2, length_scale)`` and returns the (unscaled) kernel matrix; output
scale and white noise are applied by the caller.

The kernel-family dispatch (``get_kernel_fn``) is used by the numpy kernel
classes in ``kernels_sklearn.py``: each class implements ``evaluate_jax_fn()``
returning one of the JIT functions defined here. The JAX GP backend never
inspects kernel object types directly.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np

# Ensure JAX uses 64-bit floats for numerical accuracy
jax.config.update("jax_enable_x64", True)

# Small epsilon added inside sqrt(dist^2 + eps) for Matern kernels
# to ensure differentiability at zero distance. Negligible effect on values.
_SQRT_EPS = 1e-30


@jit
def _rbf_kernel_matrix(X1, X2, length_scale):
    """RBF kernel matrix: K_ij = exp(-0.5 * sum((x1_i - x2_j)^2 / l^2))."""
    scaled_X1 = X1 / length_scale
    scaled_X2 = X2 / length_scale
    sq_X1 = jnp.sum(scaled_X1 ** 2, axis=1)
    sq_X2 = jnp.sum(scaled_X2 ** 2, axis=1)
    dist_sq = sq_X1[:, None] + sq_X2[None, :] - 2.0 * scaled_X1 @ scaled_X2.T
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
