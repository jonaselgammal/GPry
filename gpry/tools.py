"""
This module contains general tools used in different parts of the code.
"""

import numpy as np
from numpy import trace as tr
from numpy.linalg import det
from scipy.linalg import eigh
from scipy.special import erf, gamma
from scipy.optimize import newton


def kl_norm(mean_0, cov_0, mean_1, cov_1):
    """
    Computes the KL divergence between two normal distributions defined by their means
    and covariance matrices.

    May raise ``numpy.linalg.LinAlgError``.
    """
    cov_1_inv = np.linalg.inv(cov_1)
    dim = len(mean_0)
    return 0.5 * (np.log(det(cov_1)) - np.log(det(cov_0)) - dim + tr(cov_1_inv @ cov_0) +
                  (mean_1 - mean_0).T @ cov_1_inv @ (mean_1 - mean_0))


def is_valid_covmat(covmat):
    """Returns True for a Real, positive-definite, symmetric matrix."""
    try:
        if np.allclose(covmat.T, covmat) and np.all(eigh(covmat, eigvals_only=True) > 0):
            return True
        return False
    except (AttributeError, np.linalg.LinAlgError):
        return False


def cl_of_nstd(d, n):
    """
    Confidence level of hypervolume corresponding to n std's distance
    on a normalised multivariate Gaussian of dimension d.

    From https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118537
    """
    if d == 1:
        return erf(n / np.sqrt(2))
    if d == 2:
        return 1 - np.exp(-n**2 / 2)
    # d > 2
    return cl_of_nstd(d - 2, n) - \
        (n / np.sqrt(2))**(d - 2) * np.exp(-n**2 / 2) / gamma(d / 2)


def partial_n_cl_of_nstd(d, n):
    """
    Derivative w.r.t. n of `cl_of_std`.

    From https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118537
    """
    if d == 1:
        return np.sqrt(2 / np.pi) * np.exp(-n**2 / 2)
    if d == 2:
        return n * np.exp(-n**2 / 2)
    # d > 2
    return partial_n_cl_of_nstd(d - 2, n) + \
        (n ** (d - 3) * (n**2 - d + 2) * 2**(1 - d / 2) *
         np.exp(-n**2 / 2) / gamma(d / 2))


def nstd_of_cl(d, p):
    """
    Radius of hypervolume for a given confidence level in units of std's of a
    normalised multivariate Gaussian of dimension d.

    From https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118537
    """
    if d == 2:  # analytic!
        return np.sqrt(-2 * np.log(1 - p))
    return newton(lambda n: cl_of_nstd(d, n) - p, np.sqrt(d - 1),
                  fprime=lambda n: partial_n_cl_of_nstd(d, n))


def volume_sphere(r, dim=3):
    """Volume of a sphere of radius ``r`` in dimension ``dim``."""
    return np.pi**(dim / 2) / gamma(dim / 2 + 1) * r**dim


def check_random_state(seed):
    """Extension to sklearn.utils for numpy *Generators* to pass through."""
    if isinstance(seed, np.random.Generator):
        return seed
    from sklearn.utils import check_random_state
    return check_random_state(seed)


def generic_params_names(dimension, prefix="x_"):
    """Returns generic parameter names up to `dimension` (1-based) with `prefix`."""
    return [prefix + str(i) for i in range(dimension)]
