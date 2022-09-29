"""
This module contains general tools used in different parts of the code.
"""

import numpy as np
from numpy import trace as tr
from numpy.linalg import det
from scipy.linalg import eigh
from scipy.special import gamma, erfc
from scipy.stats import chi2


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


def gaussian_distance(points, mean, covmat):
    """
    Computes radial Gaussian distance in units of standar deviations
    (Mahalanobis distance).
    """
    dim = np.atleast_2d(points).shape[1]
    mean = np.atleast_1d(mean)
    covmat = np.atleast_2d(covmat)
    assert (mean.shape == (dim,) and covmat.shape == (dim, dim)), \
        (f"Mean and/or covmat have wrong dimensionality: dim={dim}, "
         f"mean.shape={mean.shape} and covmat.shape={covmat.shape}.")
    assert is_valid_covmat(covmat), "Covmat passed is not a valid covariance matrix."
    # Transform to normalised gaussian
    std_diag = np.diag(np.sqrt(np.diag(covmat)))
    invstd_diag = np.linalg.inv(std_diag)
    corrmat = invstd_diag.dot(covmat).dot(invstd_diag)
    Lscalefree = np.linalg.cholesky(corrmat)
    L = np.linalg.inv(std_diag).dot(Lscalefree)
    points_transf = L.dot((points - mean).T).T
    # Compute distance
    return np.sqrt(np.sum(points_transf**2, axis=1))


def nstd_of_1d_nstd(n1, d):
    """
    Radius of (hyper)volume in units of std's of a multivariate Gaussian of dimension
    ``d`` for a credible (hyper)volume defined by the equivalent 1-dimensional
    ``n1``-sigma interval.
    """
    return np.sqrt(chi2.isf(erfc(n1 / np.sqrt(2)), d))


def credibility_of_nstd(n, d):
    """
    Posterior mass inside of the (hyper)volume of radius ``n`` (in units of std's) of a
    multivariate Gaussian of dimension ``d``.
    """
    return chi2.cdf(n**2, d)


def volume_sphere(r, dim=3):
    """Volume of a sphere of radius ``r`` in dimension ``dim``."""
    return np.pi**(dim / 2) / gamma(dim / 2 + 1) * r**dim


def check_random_state(seed, convert_to_random_state=False):
    """
    Extension to sklearn.utils for numpy *Generators* to pass through.

    Includes workaround from https://github.com/scikit-learn/scikit-learn/issues/16988
    """
    if isinstance(seed, np.random.Generator):
        if convert_to_random_state:
            seed = np.random.RandomState(seed.bit_generator)
        return seed
    from sklearn.utils import check_random_state
    return check_random_state(seed)


def generic_params_names(dimension, prefix="x_"):
    """Returns generic parameter names up to `dimension` (1-based) with `prefix`."""
    return [prefix + str(i) for i in range(dimension)]
