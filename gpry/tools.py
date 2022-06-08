"""
This module contains general tools used in different parts of the code.
"""

import numpy as np
from numpy import trace as tr
from numpy.linalg import det
from gpry.mpi import mpi_rank
from cobaya.model import Model
import warnings
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


def generic_params_names(dimension, prefix="x_"):
    """Returns generic parameter names up to `dimension` (1-based) with `prefix`."""
    return [prefix + str(i) for i in range(dimension)]


def check_params_names_len(sampled_parameter_names, dim):
    if sampled_parameter_names is None:
        sampled_parameter_names = generic_params_names(dim)
    else:
        if len(sampled_parameter_names) != dim:
            raise ValueError(
                f"Passed `sampled_parameter_names` with {len(sampled_parameter_names)} "
                f"parameters, but the prior has dimension {dim}")
    return sampled_parameter_names


def cobaya_input_prior(cobaya_prior):
    """
    Returns a Cobaya-compatible prior input dict for GP regression,
    ignoring the prior density.
    """
    bounds = cobaya_prior.bounds(confidence_for_unbounded=0.99995)
    #bounds = cobaya_prior.bounds(confidence_for_unbounded=1-1e-10)
    epsilon = [1e-3*(bounds[i, 1]-bounds[i, 0]) for i in range(cobaya_prior.d())]
    #epsilon = [0 for i in range(cobaya_prior.d())]
    return {"params": {p: {"prior": {
                            "min": bounds[i, 0]-epsilon[i],
                            "max": bounds[i, 1]+epsilon[i]}}
                       for i, p in enumerate(cobaya_prior.params)}}

def cobaya_input_likelihood(gpr, sampled_parameter_names=None):
    """
    Returns a Cobaya-compatible likelihood input dict for the `gpr`.
    Generic parameter names used if not given.
    """
    sampled_parameter_names = check_params_names_len(
        sampled_parameter_names, gpr.X_train[0].shape[0])

    def lkl(**kwargs):
        values = [kwargs[name] for name in sampled_parameter_names]
        return gpr.predict(np.atleast_2d(values), do_check_array=False)[0]

    return {"likelihood": {"gp": {
        "external": lkl, "input_params": sampled_parameter_names}}}


def cobaya_gp_model_input(cobaya_prior, gpr):
    """
    Returns a Cobaya model input dict corresponding to a given true model and a gp
    surrogate model (which models both likelihood and the priors not defined in the
    ``params`` block.
    """
    info = cobaya_input_prior(cobaya_prior)
    print(info)
    info.update(cobaya_input_likelihood(gpr, list(cobaya_prior.params)))
    return info

def cobaya_generate_gp_input(gpr, paramnames=None, bounds=None):
    """
    Returns a Cobaya model input dict corresponding to the gp surrogate model
    The parameter names are going to be set to paramnames, while the bounds will be set from bounds. Both are optional. If no bounds are put, ???
    """
    d = len(bounds)
    if bounds is None:
        raise Exception("Sorry, it is not yet possible to provide no bounds to sample the GP on")
    else:
        if gpr.X_train[0].shape[0]!=d:
            raise ValueError("Cannot provide bounds of size {}, while gpry.X_train has length {}".format(d,gpr.X_train[0].shape[0]))
    paramnames = check_params_names_len(paramnames, d)

    epsilon = [1e-3*(bounds[i, 1]-bounds[i, 0]) for i in range(d)] ## TODO :: this is very artificial and probably should be removed eventually. It was added here by Jonas, so I am leaving it for now until we discuss further
    info = {"params": {p: {"prior": {
                            "min": bounds[i, 0]-epsilon[i],
                            "max": bounds[i, 1]+epsilon[i]}}
                       for i, p in enumerate(paramnames)}}

    def lkl(**kwargs):
        values = [kwargs[name] for name in paramnames]
        return gpr.predict(np.atleast_2d(values), do_check_array=False)[0]

    info.update({"likelihood": {"gp": {
        "external": lkl, "input_params": paramnames}}})

    return info
    

def mcmc_info_from_run(model, gpr, convergence=None):
    """
    Creates appropriate MCMC sampler inputs from the results of a run.

    Changes ``model`` reference point to the best training sample
    (or the rank-th best if running in parallel).
    """
    # Set the reference point of the prior to the sampled location with maximum
    # posterior value
    try:
        i_max_location = np.argsort(gpr.y_train)[-mpi_rank-1]
        max_location = gpr.X_train[i_max_location]
    except IndexError:  # more MPI processes than training points: sample from prior
        max_location = [None] * gpr.X_train.shape[-1]
    #model.prior.set_reference(dict(zip(model.prior.params, max_location)))
    # Create sampler info
    sampler_info = {"mcmc": {"measure_speeds": False, "max_tries": 100000}}
    # Check if convergence_criterion is given and if so try to extract the
    # covariance matrix
    if convergence is not None:
        if isinstance(convergence, str):
            _, _, _, convergence, _ = _read_checkpoint(model)
            if convergence is None:
                raise RuntimeError("Could not load the convergence criterion "
                                   "from checkpoint")
        elif not isinstance(model, Model):
            raise TypeError("convergence needs to be a gpry "
                            "Convergence_criterion instance.")
        try:
            covariance_matrix = convergence.cov
        except AttributeError:
            covariance_matrix = None
            warnings.warn("The convergence criterion does not provide a "
                          "covariance matrix. This will make the convergence "
                          "of the sampler slower.")
    else:
        covariance_matrix = None
    # Temporary solution used at some point. Eventually delete...
    # covariance_matrix = np.cov(gpr.X_train, rowvar=False)
    # Add the covariance matrix to the sampler if it exists
    if covariance_matrix is not None and is_valid_covmat(covariance_matrix):
        sampler_info["mcmc"]["covmat"] = covariance_matrix
        sampler_info["mcmc"]["covmat_params"] = list(model.prior.params)
    return sampler_info


def polychord_info_from_run(model, gpr, convergence=None):
    """
    Creates appropriate PolyChord sampler inputs from the results of a run.
    """
    # Create sampler info
    sampler_info = {"polychord": {"measure_speeds": False}}
    return sampler_info


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
