import warnings
import logging
import numpy as np
from copy import deepcopy
from gpry.gpr import GaussianProcessRegressor
from gpry.mpi import mpi_rank
from gpry.tools import generic_params_names, is_valid_covmat
from cobaya.model import Model, get_model
from cobaya.output import get_output
from cobaya.sampler import get_sampler


def get_cobaya_log_level(verbose):
    """Given GPry's verbosity level, returns the corresponding Cobaya debug level."""
    if verbose is None or verbose == 3:
        return logging.INFO
    elif verbose > 3:
        return logging.DEBUG
    elif verbose == 2:
        return logging.WARNING
    elif verbose == 1:
        return logging.ERROR
    elif verbose < 1 or verbose is False:
        return logging.CRITICAL
    else:
        raise ValueError(f"Verbosity level {verbose} not understood.")


def cobaya_generate_gp_model_input(gpr, bounds=None, paramnames=None, true_model=None):
    """
    Returns a Cobaya model input dict corresponding to the GP surrogate model ``gpr``.

    If no other argument is passed, it samples within the bounds of the GP model,
    and uses generic parameter names.

    Parameters
    ----------
    gpr : GaussianProcessRegressor, which has been fit to data and returned from
        the ``run`` function.

    bounds : List of boundaries (lower,upper), optional
        If none are provided it tries to extract the bounds from ``true_model``. If
        that fails it tries extracting them from gpr. If none of those methods succeed
        an error is raised.

    paramnames : List of parameter strings, optional
        If none are provided it tries to extract parameter names from ``true_model``.
        If that fails it uses some dummy strings.

    true_model : Cobaya Model, optional
        If passed, it uses it to get bounds and parameter names (unless overriden by
        the corresponding kwargs).

    Returns
    -------
    info : dict
        A dict containing the ``prior`` and ``likelihood`` blocks.
    """
    if bounds is not None:
        if np.array(bounds).shape != np.array(gpr.bounds).shape \
                and gpr.bounds is not None:
            raise ValueError(f"``bounds`` has the wrong shape {np.array(bounds).shape}. "
                             f"Expected {np.array(gpr.bounds).shape}.")
    elif true_model is not None:
        bounds = true_model.prior.bounds(confidence_for_unbounded=0.99995)
        if np.array(bounds).shape != np.array(gpr.bounds).shape \
                and gpr.bounds is not None:
            raise ValueError("The dimensionality of the prior of `true_model` "
                             f"({true_model.prior.d()}) does not correspond to that of "
                             f"the GP ({gpr.d}).")
    elif gpr.bounds is not None:
        bounds = deepcopy(gpr.bounds)
    else:
        raise ValueError("You need to either provide bounds, a model or a GP regressor "
                         "with bounds.")
    paramlabels = None
    if paramnames is not None:
        if len(paramnames) != gpr.d:
            raise ValueError(f"Passed `paramnames` with {len(paramnames)} "
                             f"parameters, but the prior has dimension {gpr.d}")
    elif true_model is not None:
        if not isinstance(true_model, Model):
            raise ValueError("`true_model` must be a Cobaya model.")
        paramnames = list(true_model.parameterization.sampled_params())
        all_labels = true_model.parameterization.labels()
        paramlabels = [all_labels[p] for p in paramnames]
    else:
        paramnames = generic_params_names(gpr.d)
    info = {"params": {p: {"prior": list(b)} for p, b in zip(paramnames, bounds)}}
    if paramlabels:
        for p, l in zip(paramnames, paramlabels):
            info["params"][p]["latex"] = l
    # TODO :: this is very artificial and probably should be removed eventually.
    # It was added here by Jonas, so I am leaving it for now until we discuss further
    epsilon = [1e-8 * (bounds[i, 1] - bounds[i, 0]) for i in range(gpr.d)]
    for p, eps in zip(info["params"].values(), epsilon):
        p["prior"] = [p["prior"][0] - eps, p["prior"][1] + eps]

    def lkl(**kwargs):
        values = [kwargs[name] for name in paramnames]
        # we need to add the log-volume of the prior we define here as the GP
        # interpolates the posterior, not the likelihood.
        log_prior_volume = np.sum(np.log(bounds[:,1] - bounds[:,0]))
        return gpr.predict(np.atleast_2d(values), validate=False)[0] + \
            log_prior_volume

    info.update({"likelihood": {"gp": {
        "external": lkl, "input_params": paramnames}}})
    return info


def mcmc_info_from_run(model, gpr, cov=None, cov_params=None, verbose=3):
    """
    Creates appropriate MCMC sampler inputs from the results of a run.

    Changes ``model`` reference point to the best training sample
    (or the rank-th best if running in parallel).

    Parameters
    ----------
    model : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_
        Contains all information about the parameters in the likelihood and
        their priors as well as the likelihood itself.

    gpr : GaussianProcessRegressor, which has been fit to data and returned from
        the ``run`` function.

    cov : Covariance matrix, optional
        A covariance matrix to speed up convergence of the MCMC. If none is provided the
        MCMC will run without but it will be slower at converging.

    Returns
    -------
    sampler : dict
        a dict with the ``sampler`` block for Cobaya's run function.
    """
    # Set the reference point of the prior to the sampled location with maximum
    # posterior value
    try:
        i_max_location = np.argsort(gpr.y_train)[-mpi_rank - 1]
        max_location = gpr.X_train[i_max_location]
    except IndexError:  # more MPI processes than training points: sample from prior
        max_location = [None] * gpr.X_train.shape[-1]
    model.prior.set_reference(dict(zip(model.prior.params, max_location)))
    # Create sampler info
    sampler_info = {"mcmc": {"measure_speeds": False, "max_tries": 100000}}
    if (cov is None or not is_valid_covmat(cov)) and verbose >= 2:
        warnings.warn("No covariance matrix or invalid one provided for the `mcmc` "
                      "sampler. This will make the convergence of the sampler slower.")
    else:
        sampler_info["mcmc"]["covmat"] = cov
        sampler_info["mcmc"]["covmat_params"] = cov_params or list(model.prior.params)
    return sampler_info


def polychord_info_from_run():
    """
    Creates a PolyChord sampler with standard parameters.

    Returns
    -------
    sampler : dict
        a dict with the ``sampler`` block for Cobaya's run function.
    """
    # Create sampler info
    sampler_info = {"polychord": {"measure_speeds": False}}
    return sampler_info


def mc_sample_from_gp(gpr, bounds=None, paramnames=None, true_model=None,
                      sampler="mcmc", add_options=None,
                      output=None, run=True, resume=False, convergence=None, verbose=3):
    """
    Generates a `Cobaya Sampler <https://cobaya.readthedocs.io/en/latest/sampler.html>`_
    and runs it on the surrogate model.

    Parameters
    ----------
    gpr : GaussianProcessRegressor, which has been fit to data and returned from
        the ``run`` function.
        Alternatively a string containing a path with the
        location of a saved GP run (checkpoint) can be provided (the same path
        that was used to save the checkpoint in the ``run`` function).

    bounds : List of boundaries (lower,upper), optional
        By default it reads them from the GP regressor.

    paramnames : List of parameter strings, optional
        By default it uses some dummy strings.

    true_model : Cobaya Model, optional
        If passed, it uses it to get bounds and parameter names (unless overriden by
        the corresponding kwargs).

    sampler : string (default `"mcmc"`) or dict
        Sampler to be initialised. If a string, it must be `"mcmc"` or `"polychord"`.
        It can also be a dict as ``{sampler: {option: value, ...}}``, containing a full
        sampler definition, see `here
        <https://cobaya.readthedocs.io/en/latest/sampler.html>`_. In this case, any
        sampler understood by Cobaya can be used.

    add_options : dict, optional
        Dict of additional options to be passed to the sampler.

    output: path, optional
        The path where the resulting Monte Carlo sample shall be stored.

    run: bool, default: True
        Whether to run the sampler. If ``False``, returns just an initialised sampler.

    resume: bool, optional (default=False)
        Whether to resume from existing output files (True) or force overwrite (False)

    convergence: Convergence_criterion, optional
        The convergence criterion which has been used to fit the GP. This is
        used to extract the covariance matrix if it is available from the
        ConvergenceCriterion class.

    verbose: int (default 3)
        Verbosity level, similarly valued to that of the Runner, e.g. 3 indicates cobaya's
        'info' level, 4 the 'debug' level, and lower-than-three values print only warnings
        and errors.

    Returns
    -------
    surr_info : dict
        The dictionary that was used to run (or initialized) the sampler, corresponding to
        the surrogate model, and populated with the sampler input specification.

    sampler : Sampler instance
        The sampler instance that has been run (or just initialised). The sampler products
        can be retrieved with the `Sampler.products()` method.
    """
    loaded_convergence = None
    if isinstance(gpr, str):
        from gpry.run import _read_checkpoint
        _, gpr, _, loaded_convergence, _ = _read_checkpoint(gpr)
        if gpr is None:
            raise RuntimeError("Could not load the GP regressor from checkpoint")
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError("The GP `gpr` needs to be a gpry GP Regressor or a string "
                        "with a path to a checkpoint file.")
    if not hasattr(gpr, "y_train"):
        warnings.warn("The provided GP hasn't been trained to data "
                      "before. This is likely unintentional...")
    model_input = cobaya_generate_gp_model_input(
        gpr, bounds=bounds, paramnames=paramnames, true_model=true_model)
    model_input["debug"] = get_cobaya_log_level(verbose)
    model_surrogate = get_model(model_input)
    # Check if convergence_criterion is given/loaded: it may contain a covariance matrix
    covariance_matrix = None
    covariance_params = paramnames
    for conv in [convergence, loaded_convergence]:
        try:
            covariance_matrix = conv.cov
        except AttributeError:
            pass
    # Otherwise, maybe passed in add_options (prefer this one)
    if "covmat" in (add_options or {}):
        covariance_matrix = add_options.pop("covmat")
        covariance_params = add_options.pop("covmat_params", None)
    if sampler.lower() == "mcmc":
        sampler_input = mcmc_info_from_run(model_surrogate, gpr, cov=covariance_matrix,
                                           cov_params=covariance_params, verbose=verbose)
    elif sampler.lower() == "polychord":
        if output is False:
            warnings.warn("Polychord cannot run without output. Mind that it defaults "
                          "to /tmp/polychord_raw")
        sampler_input = polychord_info_from_run()
    elif isinstance(sampler, str):
        raise ValueError("`sampler` must be `mcmc|polychord`")
    elif not isinstance(sampler, dict):
        raise ValueError("`sampler` must be `mcmc|polychord` or a full sampler "
                         "specification as a dict.")
    else:  # dict
        sampler_input = sampler
    sampler_name = list(sampler_input.keys())[0]
    sampler_input[sampler_name].update(add_options or {})
    out = None
    if output is not None:
        if not resume:
            out = get_output(prefix=output, resume=False, force=True)
        else:
            out = get_output(prefix=output, resume=True, force=False)
    sampler = get_sampler(sampler_input, model=model_surrogate, output=out)
    surr_info = model_surrogate.info()
    if not run:
        surr_info["sampler"] = {sampler_name: sampler.info()}
        return surr_info, sampler
    sampler.run()
    # TODO: share chains MPI!!!
    surr_info["sampler"] = {sampler_name: sampler.info()}
    return surr_info, sampler
