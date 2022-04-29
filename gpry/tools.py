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
        

def generate_sampler_for_gp(gp, bounds=None, paramnames=None, sampler="mcmc", convergence=None, options=None,
                      output=None, add_options=None, restart=False):
    """
    This function is essentially just a wrapper for the Cobaya MCMC sampler
    (monte python) which can run an MCMC on the fitted GP regressor.
    It does NOT yet run any chaisn

    Parameters
    ----------

    gp : GaussianProcessRegressor, which has been fit to data and returned from
        the ``run`` function.
        Alternatively a string containing a path with the
        location of a saved GP run (checkpoint) can be provided (the same path
        that was used to save the checkpoint in the ``run`` function).

    bounds : List of boundaries (lower,upper), optional
        By default it doesn't use boundaries

    paramnames : List of parameter strings, optional
        By default it uses some dummy strings, which affects the updated_info
    
    convergence : Convergence_criterion, optional
        The convergence criterion which has been used to fit the GP. This is
        used to extract the covariance matrix if it is available from the
        Convergence_criterion class. Alternatively a string containing a path
        with the location of a saved GP run (checkpoint) can be provided (the
        same path that was used to save the checkpoint in the ``run`` function).

    options: dict, optional
        Containing the options for the mcmc sampler
        defined in the "sampler" block of the Cobaya input. For more
        information see
        `here <https://cobaya.readthedocs.io/en/latest/sampler.html>`.

        .. note::
            If you specify any options here you need to define the whole
            "sampler" block. This leaves room for the possibility to also use
            other samplers which are built into Cobaya (i.e. PolyChord).

    output: path, optional
        The path where the output of the MCMC (chains) shall be stored.

    Returns
    -------

    surr_info : dict
        The dictionary that was used to initialize the sampler, corresponding to the surrogate model

    sampler : Sampler instance
        The sampler instance that is NOT yet run
    """

    # Check GP
    if isinstance(gp, GaussianProcessRegressor):
        if hasattr(gp, 'y_train'):
            gpr = gp
        else:
            warnings.warn("The provided GP hasn't been trained to data "
                          "before. This is likely unintentional...")
    elif isinstance(gp, str):
        _, gpr, _, _, _ = _read_checkpoint(gp)
        if gpr is None:
            raise RuntimeError("Could not load the GP regressor from checkpoint")
        if not hasattr(gpr, "y_train"):
            warnings.warn("The provided GP hasn't been trained to data "
                          "before. This is likely unintentional...")
    else:
        raise TypeError("The GP needs to be a gpry GP Regressor or a string "
                        "with a path to a checkpoint file.")

    model_surrogate = get_model(
        cobaya_generate_gp_input(gpr, paramnames=paramnames,bounds=bounds))

    # Check if options for the sampler are given else build the sampler
    if options is None:
        if sampler.lower() == "mcmc":
            sampler_info = mcmc_info_from_run(model_surrogate, gpr, convergence)
        elif sampler.lower() == "polychord":
            sampler_info = polychord_info_from_run(model_surrogate, gpr, convergence)
        else:
            raise ValueError("`sampler` must be `mcmc|polychord`")
    else:
        sampler_info = options
    if add_options is not None:
        for key in add_options:
          sampler_info[key].update(add_options[key])

    out = None
    if output is not None:
        if not restart:
            out = get_output(prefix=output, resume=False, force=True)
        else:
            out = get_output(prefix=output, resume=restart, force=False)

    # Create the sampler
    sampler = get_sampler(sampler_info, model=model_surrogate, output=out)

    # Return also surrogate info
    surr_info = model_surrogate.info()

    return surr_info, sampler

