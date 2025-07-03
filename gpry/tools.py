"""
This module contains general tools used in different parts of the code.
"""

from copy import deepcopy
import inspect
from warnings import warn

import numpy as np
from numpy import trace as tr
from numpy.linalg import det
from scipy.linalg import eigh  # type: ignore
from scipy.special import gamma, erfc  # type: ignore
from scipy.stats import chi2  # type: ignore
from sklearn.utils import check_random_state as check_random_state_sklearn  # type: ignore


def kl_norm(mean_0, cov_0, mean_1, cov_1):
    """
    Computes the KL divergence between two normal distributions defined by their means
    and covariance matrices.

    May raise ``numpy.linalg.LinAlgError``.
    """
    cov_1_inv = np.linalg.inv(cov_1)
    dim = len(mean_0)
    with NumpyErrorHandling(all="ignore") as _:
        return 0.5 * (
            np.log(det(cov_1))
            - np.log(det(cov_0))
            - dim
            + tr(cov_1_inv @ cov_0)
            + (mean_1 - mean_0).T @ cov_1_inv @ (mean_1 - mean_0)
        )


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
    covmat = np.atleast_2d(covmat)
    try:
        if np.allclose(covmat.T, covmat) and np.all(
            eigh(covmat, eigvals_only=True) > 0
        ):
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
    assert mean.shape == (dim,) and covmat.shape == (dim, dim), (
        f"Mean and/or covmat have wrong dimensionality: dim={dim}, "
        f"mean.shape={mean.shape} and covmat.shape={covmat.shape}."
    )
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


def nstd_of_1d_nstd(n1, d, warn_inf=True):
    """
    Radius of (hyper)volume in units of std's of a multivariate Gaussian of dimension
    ``d`` for a credible (hyper)volume defined by the equivalent 1-dimensional
    ``n1``-sigma interval.
    """
    nstd = np.sqrt(chi2.isf(erfc(n1 / np.sqrt(2)), d))
    if warn_inf and not np.isfinite(nstd):
        warn(f"Got -inf for n1={n1} and d={d}. This may cause errors.")
    return nstd


def delta_logp_of_1d_nstd(n1, d):
    """
    Difference between the peak/mean of a Gaussian log-probability and the level
    corresponding to the credible (hyper)volume defined by the equivalent 1-dimensional
    ``n1``-sigma interval in ``d`` dimensions.
    """
    return 0.5 * nstd_of_1d_nstd(n1, d) ** 2


def credibility_of_nstd(n, d):
    """
    Posterior mass inside of the (hyper)volume of radius ``n`` (in units of std's) of a
    multivariate Gaussian of dimension ``d``.
    """
    return chi2.cdf(n**2, d)


def volume_sphere(r, dim=3):
    """Volume of a sphere of radius ``r`` in dimension ``dim``."""
    return np.pi ** (dim / 2) / gamma(dim / 2 + 1) * r**dim


def check_random_state(seed, convert_to_random_state=False):
    """
    Extension to sklearn.utils for numpy *Generators* to pass through.

    Includes workaround from https://github.com/scikit-learn/scikit-learn/issues/16988
    """
    if isinstance(seed, np.random.Generator):
        if convert_to_random_state:
            seed = np.random.RandomState(seed.bit_generator)
        return seed
    return check_random_state_sklearn(seed)


def generic_params_names(n, prefix="x_"):
    """
    Returns ``n`` generic parameter names (1-based) as ``[prefix]1``, ``[prefix]2``, etc.
    """
    if (
        not isinstance(n, int)
        and not (isinstance(n, float) and np.isclose(n, int(n)))
        or n <= 0
    ):
        raise TypeError(
            f"'n' must be a positive integer. Got {n!r} of type {type(n)})."
        )
    if not isinstance(prefix, str):
        raise TypeError(
            f"'prefix' must be a string. Got {prefix!r} of type {type(prefix)})."
        )
    return [prefix + str(i + 1) for i in range(int(n))]


class NumpyErrorHandling:
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

    If ``X_value`` is not defined, returns a tuple ``(value, has_X, X_power)``.
    """
    not_allowed = [" ", ".", "-", "+", "e", "E", ",", ";"]
    if X_letter in not_allowed:
        raise ValueError(
            f"X_letter not allowed: '{X_letter}'. It cannot any of {not_allowed}"
        )
    if not isinstance(dtype, type):
        raise ValueError(f"'dtype' arg must be a type, not {type(dtype)}.")
    if value == X_letter:
        value = "1" + X_letter
    # Avoid exceptions until the 'try' block
    if isinstance(value, str) and X_letter in value:
        has_X = True
        num_value, X_power = value.split(X_letter)
        if not num_value:
            num_value = 1
        if not X_power:
            X_power = None
    else:
        has_X = False
        num_value = value
        X_power = None
    try:
        # Start with float. Impose dtype only at return
        num_value = float(num_value)
        if X_value is None:  # special case: X value undefined (see docstring)
            return (
                dtype(num_value),
                has_X,
                X_power if X_power is None else float(X_power),
            )
        if has_X:
            X_multiplier = X_value
            if X_power is not None:
                X_multiplier = X_multiplier ** float(X_power)
        else:
            X_multiplier = 1
        return dtype(num_value * X_multiplier)
    except (ValueError, TypeError) as excpt:
        pre = f"Error setting variable '{varname}': " if varname else ""
        raise ValueError(
            pre + f"Could not convert {value} of type {type(value)} into type "
            f"{dtype.__name__}. Pass either a string ending in '{X_letter}' or a valid "
            f"{dtype.__name__} value."
        ) from excpt


def check_candidates(surrogate, new_X, tol=1e-8):
    """
    Method for determining whether points which have been found by the
    acquisition algorithm are already in the GPR or the surrogate model or appear multiple
    times so that they can be removed.
    Returns two boolean arrays, the first one indicating whether the point
    is already in the GPR of the surrogate model and the second one indicating whether the
    point appears multiple times.
    """
    new_X_ = surrogate.preprocessing_X.transform(np.copy(new_X))
    X_train_ = np.copy(surrogate._X_)
    new_X_r_ = np.round(new_X_, decimals=int(-np.log10(tol)))
    X_train_r_ = np.round(X_train_, decimals=int(-np.log10(tol)))
    in_training_set = np.any(np.all(X_train_r_[:, None, :] == new_X_r_, axis=2), axis=0)
    unique_rows, indices, counts = np.unique(
        new_X_r_, axis=0, return_index=True, return_counts=True
    )
    is_duplicate = counts > 1
    duplicates = np.isin(new_X_r_, unique_rows[is_duplicate]).all(axis=1)
    duplicates[indices[is_duplicate]] = False
    return in_training_set, duplicates


def is_in_bounds(points, bounds, check_shape=False):
    """
    Checks if a point or set of points is within the given bounds.

    Parameters
    ----------
    points: numpy.ndarray
        An (N, d) array of points to check
    bounds: numpy.ndarray
        An (d, 2) array of parameter bounds
    check_shape : bool (default: False)
        Whether to check for consistency of array shapes.

    Returns
    -------
    numpy.ndarray:
        A boolean array of length N indicating whether each point is within the bounds.
    """
    points = np.atleast_2d(points)
    if check_shape:
        bounds = check_and_return_bounds(bounds)
        if bounds.shape[0] != points.shape[1]:
            raise ValueError(
                "bounds and point appear to have different dimensionalities: "
                f"{bounds.shape[0]} for bounds and {points.shape[1]} for point."
            )
    return np.all((points >= bounds[:, 0]) & (points <= bounds[:, 1]), axis=1)


def check_and_return_bounds(bounds):
    """
    Returns the passed bounds as a (dim, 2)-shaped array if it can be mapped to one,
    and raises TypeError otherwise.
    """
    try:
        bounds_ = np.atleast_2d(bounds)
        if bounds_.shape[1] != 2:
            raise ValueError
    except ValueError as excpt:
        raise TypeError(
            f"bounds must be a (dim, 2) array of bounds, but is {bounds}"
        ) from excpt
    return bounds_


def shrink_bounds(bounds, samples, factor=1):
    """
    Reduces the given bounds to the minimal hypercube containing a set of `samples`.

    If ``factor != 1``, the width is multiplied by that factor, while keeping the
    hypercube centered.

    If the samples span a longer region that that defined by the bounds, the given
    bounds are preferred.

    Parameters
    ----------
    bounds: numpy.ndarray
        An (d, 2) array of parameter bounds
    samples: numpy.ndarray
        An (N, d) array of sampling locations
    factor: float
        A factor by which to multiply the hypercube width

    Returns
    -------
    numpy.ndarray:
        An (d, 2) array of updated parameter bounds

    Raises
    ------
    TypeError:
        If bounds or samples are not well formatted or inconsistent with each other.
    """
    bounds = check_and_return_bounds(bounds)
    try:
        samples = np.atleast_2d(samples)
    except ValueError as excpt:
        raise TypeError(
            "samples are not correctly formatted as an array with sha[e (nsamples, dim)"
        ) from excpt
    if bounds.shape[0] != samples.shape[1]:
        raise TypeError(
            "bounds and samples appear to have different dimensionalities: "
            f"{bounds.shape[0]} for bounds and {samples.shape[1]} for samples."
        )
    updated_bounds = np.empty(shape=bounds.shape, dtype=float)
    # Find training bounds
    updated_bounds[:, 0] = samples.min(axis=0)
    updated_bounds[:, 1] = samples.max(axis=0)
    # Apply factor
    width = updated_bounds[:, 1] - updated_bounds[:, 0]
    Delta = (factor - 1) / 2 * width
    updated_bounds[:, 0] -= Delta
    updated_bounds[:, 1] += Delta
    # Restrict to prior
    updated_bounds[:, 0] = np.array([updated_bounds[:, 0], bounds[:, 0]]).max(axis=0)
    updated_bounds[:, 1] = np.array([updated_bounds[:, 1], bounds[:, 1]]).min(axis=0)
    return updated_bounds


def wrap_likelihood(loglike, ndim):
    """
    Wraps a likelihood function to accept a single argument (a vector of parameters) if it takes
    multiple arguments.
    """
    sig = inspect.signature(loglike)
    params = sig.parameters

    # Count parameters by kind
    positional = [
        p
        for p in params.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    keyword = [p for p in params.values() if p.kind == inspect.Parameter.KEYWORD_ONLY]
    var_positional = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()
    )
    var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    n_args = len(positional)
    n_all = n_args + len(keyword)

    # Case 1: function takes exactly ndim arguments (positional/keyword), wrap it
    if n_args == ndim or n_all == ndim:

        def wrapped(X):
            return loglike(*X)

        return wrapped

    # Case 2: function takes one argument (e.g. already accepts a vector), pass through
    elif n_args == 1 and not var_positional and not var_keyword:
        return loglike

    else:
        raise ValueError(
            f"Function has signature {sig} which is incompatible with ndim={ndim}"
        )


def remove_0_weight_samples(weights, *arrays):
    """
    Removes the elements of ``arrays`` (at axis 0) corresponding to the null ``weights``.

    Returns a tuple with the non-null weights as the first element, and the rest of them
    being copies of the given arrays without null-weighted samples in the order with which
    they were passed. If an element of the list of arrays is ``None``, ``None`` is
    returned in its place.
    """
    i_zero_w = np.where(weights == 0)[0]
    new_arrays = [np.delete(weights, i_zero_w)]
    for array in arrays:
        if array is None:
            new_arrays.append(None)
        elif array.shape[0] != len(weights):
            raise ValueError("weights and some of the arrays have different lengths.")
        else:
            new_arrays.append(np.delete(array, i_zero_w, axis=0))
    return new_arrays


def mean_covmat_from_samples(X, w=None):
    """
    Returns an estimation of the mean and covariance of a set ``X`` of points, using their
    ``logp`` as weights.
    """
    mean = np.average(X, weights=w, axis=0)
    cov = np.cov(X.T, aweights=w)
    return mean, cov


def mean_covmat_from_evals(X, logp):
    """
    Returns an estimation of the mean and covariance of a set ``X`` of points, using their
    ``logp`` as weights.
    """
    weights = np.exp(logp - max(logp))
    weights_, X_ = remove_0_weight_samples(weights, X)
    mean = np.average(X_, axis=0, weights=weights_)
    covmat = np.cov(X_.T, aweights=weights_, ddof=0)
    return mean, covmat
