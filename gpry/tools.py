"""
This module contains general tools used in different parts of the code.
"""

from copy import deepcopy
from inspect import signature
from typing import Mapping, Iterable

import numpy as np
from numpy import trace as tr
from numpy.linalg import det
from scipy.linalg import eigh
from scipy.special import gamma, erfc
from scipy.stats import chi2
from cobaya.model import get_model


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


def kl_mc(X, logq_func, logp=None, logp_func=None, weight=None):
    """
    Computes KL(P||Q) divergence using Monte Carlo samples X of P, and the logpdf of Q.

    All functions assumed vectorised.

    The logpdf's must be both normalised, or with the same off-normalisation factor, e.g.
    different normalised likelihods with the same prior which is much larger than the
    mode.
    """
    if logp is None and logp_func is None:
        raise ValueError("Needs either logp at X, or a logp(X) function.")
    if logp is None:
        logp = logp_func(X)
    mask = np.isfinite(logp)
    logp = logp[mask]
    logq_at_X = logq_func(X[mask])
    if weight is None:
        weight = np.ones(len(logp))
        total_weight = len(logp)
    else:
        weight = weight[mask]
        total_weight = np.sum(weight)
    return np.dot(weight, logp - logq_at_X) / total_weight


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


def generic_params_names(n, prefix="x_"):
    """
    Returns ``n`` generic parameter names (1-based) as ``[prefix]1``, ``[prefix]2``, etc.
    """
    if (
        not isinstance(n, int)
        and not (isinstance(n, float) and np.isclose(n, int(n)))
        or n <= 0
    ):
        raise TypeError(f"'n' must be a positive integer. Got {n!r} of type {type(n)}).")
    if not isinstance(prefix, str):
        raise TypeError(
            f"'prefix' must be a string. Got {prefix!r} of type {type(prefix)})."
        )
    return [prefix + str(i + 1) for i in range(int(n))]


# TODO -- this will be inside Cobaya eventually -> remove
def create_cobaya_model(likelihood, bounds, prefix="x_"):
    """
    Creates a cobaya Model from a likelihood and some bounds.

    Parameters
    ----------
    likelihood : callable
        Function returning a log-likelihood. It needs to take at least as many parameters
        as contained in ``bounds``.

    bounds: List of [min, max], or Dict {name: [min, max],...}
        List or dictionary of parameter bounds. If it is a dictionary, the keys need to
        correspond to the argument names of the ``likelihood`` function.

    Returns
    -------
    model: a cobaya Model.
    """
    if not callable(likelihood):
        raise TypeError(
            f"'likelihood' needs to be a callable function. Got {likelihood!r} of type"
            f" {type(likelihood)}."
        )
    params_names = list(signature(likelihood).parameters)
    if isinstance(bounds, Mapping):
        assert set(bounds).issubset(params_names), (
            f"Parameters passed in 'bounds' {list(bounds)} not compatible with the"
            f" likelihood arguments {params_names}."
        )
        params_input = {
            k: (v if isinstance(v, Mapping) else {"prior": v}) for k, v in bounds.items()
        }
    elif isinstance(bounds, Iterable) and not isinstance(bounds, str):
        assert len(bounds) <= len(params_names), (
            f"Bounds for {len(bounds)} parameter(s) were passed, but the likelihood "
            f"depends on {len(params_names)} parameter(s) only (namely {params_names})."
        )
        params_input = {
            params_names[i]: {"prior": bound}
            for i, bound in enumerate(bounds)
        }
    else:
        raise TypeError(
            "'bounds' must be a list of [min, max] bounds, or a dictionary "
            f"containing parameter names as keys and bounds as values. Got {bounds!r}"
        )
    likelihood_input = {"likelihood_function": {"external": likelihood}}
    return get_model({"params": params_input, "likelihood": likelihood_input})


class NumpyErrorHandling():
    """
    Context for manual handling of numpy errors (e.g. ignoring, just printing...).

    NB: the call to ``deepcopy`` at init can become expensive if this ``with`` context
        is used repeatedly. One may want to put it at an upper level then.
    """
    def __init__(self, all):
        self.all = all
        self.error_handler = deepcopy(np.geterr())

    def __enter__(self):
        np.seterr(all=self.all)

    def __exit__(self, error_type, error_value, error_traceback):
        np.seterr(**self.error_handler)
        if error_type is not None:
            raise


def get_Xnumber(value, X_letter, X_value=None, dtype=int, varname=None):
    """
    Reads a value out of an X-number, e.g.: "5X" as 5 times the value of X.

    If ``X_value`` is not defined, returns a tuple ``(value, value.endswith(X))``.
    """
    if not isinstance(dtype, type):
        raise ValueError(f"'dtype' arg must be a type, not {type(dtype)}.")
    if value == X_letter:
        value = "1" + X_letter
    if isinstance(value, str) and value.endswith(X_letter):
        num_value = value.rstrip(X_letter)
        has_X = True
    else:
        num_value = value
        has_X = False
    try:
        num_value = dtype(num_value)
        if X_value is None:
            return (num_value, has_X)
        return dtype(num_value * (1 if not has_X else X_value))
    except (ValueError, TypeError) as excpt:
        pre = f"Error setting variable '{varname}': " if varname else ""
        raise ValueError(
            pre + f"Could not convert {value} of type {type(value)} into type "
            f"{dtype.__name__}. Pass either a string ending in '{X_letter}' or a valid "
            f"{dtype.__name__} value."
        ) from excpt
    
def check_candidates(gpr, new_X, tol=1e-8):
    """
    Method for determining whether points which have been found by the
    acquisition algorithm are already in the GP or appear multiple times
    so that they can be removed.
    Returns two boolean arrays, the first one indicating whether the point
    is already in the GP and the second one indicating whether the point
    appears multiple times.
    """
    if gpr.preprocessing_X is not None:
        new_X = gpr.preprocessing_X.transform(np.copy(new_X))
    X_train = np.copy(gpr.X_train_)

    new_X_r = np.round(new_X, decimals=int(-np.log10(tol)))
    X_train_r = np.round(X_train, decimals=int(-np.log10(tol)))
    in_training_set = np.any(np.all(X_train_r[:, None, :] == new_X_r, axis=2), axis=0)

    unique_rows, indices, counts = np.unique(new_X_r, axis=0, return_index=True, return_counts=True)
    is_duplicate = counts > 1
    duplicates = np.isin(new_X_r, unique_rows[is_duplicate]).all(axis=1)
    duplicates[indices[is_duplicate]] = False
    return in_training_set, duplicates
