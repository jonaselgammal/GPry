"""
This module contains general tools used in different parts of the code.
"""

import numpy as np
from numpy import trace as tr
from numpy.linalg import det


def kl_norm(mean_0, cov_0, mean_1, cov_1):
    """
    Computes the KL divergence between two normal distributions defined by their means
    and covariance matrices.
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


def cobaya_input_prior(cobaya_prior, sampled_parameter_names=None):
    """
    Returns a Cobaya-compatible prior input dict.
    Generic parameter names used if not given.
    """
    sampled_parameter_names = check_params_names_len(
        sampled_parameter_names, cobaya_prior.d())
    bounds = cobaya_prior.bounds(confidence_for_unbounded=0.99995)
    return {"params": {p: {"prior": {"min": bounds[i, 0], "max": bounds[i, 1]}}
                       for i, p in enumerate(sampled_parameter_names)}}


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


def cobaya_gp_model_input(cobaya_prior, gpr, sampled_parameter_names=None):
    """
    Returns a Cobaya model input dict corresponding to a given true model and a gp
    surrogate model (which models both likelihood and the priors not defined in the
    ``params`` block.
    """
    info = cobaya_input_prior(cobaya_prior, sampled_parameter_names)
    info.update(cobaya_input_likelihood(gpr, sampled_parameter_names))
    return info
