"""
This module provides some helper methods to interface between GPry and MC samplers
(Cobaya and the NS' used for NORA).

Under normal circumstances you shouldn't have to use any of the methods in here if you use
the :class:`run.Runner` class to run GPry.
"""

import os
import warnings
import logging
import tempfile
from copy import deepcopy

import numpy as np
from getdist.mcsamples import MCSamples, loadMCSamples  # type: ignore
from getdist.gaussian_mixtures import GaussianND  # type: ignore

from gpry import mpi
from gpry.tools import generic_params_names, is_valid_covmat
from gpry.io import ensure_surrogate, create_path
import gpry.ns_interfaces as nsint

# Keys and plot labels for the MC samples dict
_name_logp, _label_logp = "logpost", r"\log(p)"
_name_logprior, _label_logprior = "logprior", r"\log(\pi)"
_name_loglike, _label_loglike = "loglike", r"\log(\mathrm{L})"


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


def cobaya_generate_surr_model_input(surrogate, bounds=None, params=None):
    """
    Returns a Cobaya model input dict corresponding to the GP surrogate model
    ``surrogate``.

    If no other argument is passed, it samples within the bounds of the GP model,
    and uses generic parameter names.

    Parameters
    ----------
    surrogate : SurrogateModel, which has been fit to data and returned from
        the ``run`` function.

    bounds : List of boundaries (lower,upper), optional
        If none are provided it tries to extract the bounds from ``true_model``. If
        that fails it tries extracting them from the surrogate. If none of those methods
        succeed an error is raised.

    paramnames : List of parameter strings, optional
        If none are provided it tries to extract parameter names from ``true_model``.
        If that fails it uses some dummy strings.

    Returns
    -------
    info : dict
        A dict containing the ``prior`` and ``likelihood`` blocks.
    """
    if bounds is not None:
        if (
            np.array(bounds).shape != np.array(surrogate.bounds).shape
            and surrogate.bounds is not None
        ):
            raise ValueError(
                f"``bounds`` has the wrong shape {np.array(bounds).shape}. "
                f"Expected {np.array(surrogate.bounds).shape}."
            )
    elif surrogate.bounds is not None:
        bounds = deepcopy(surrogate.bounds)
    else:
        raise ValueError(
            "You need to either provide bounds, a model or a GP regressor with bounds."
        )
    # Get labels
    if params is not None:
        if len(params) != surrogate.d:
            raise ValueError(
                f"Passed `params` with {len(params)} "
                f"parameters, but the prior has dimension {surrogate.d}"
            )
    else:
        params = generic_params_names(surrogate.d)
    info = {"params": {p: {"prior": list(b)} for p, b in zip(params, bounds)}}
    log_prior_volume = np.sum(np.log(bounds[:, 1] - bounds[:, 0]))

    def lkl(**kwargs):
        values = [kwargs[name] for name in params]
        # we need to add the log-volume of the prior we define here as the GP
        # interpolates the posterior, not the likelihood.
        return (
            surrogate.predict(np.atleast_2d(values), validate=False)[0]
            + log_prior_volume
        )

    info.update({"likelihood": {"gp": {"external": lkl, "input_params": params}}})
    return info


def mcmc_info_from_run(model, surrogate, cov=None, cov_params=None, verbose=3):
    """
    Creates appropriate MCMC sampler inputs from the results of a run.

    Changes ``model`` reference point to the best training sample
    (or the rank-th best if running in parallel).

    Parameters
    ----------
    model : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_
        Contains all information about the parameters in the likelihood and
        their priors as well as the likelihood itself.

    surrogate : SurrogateModel, which has been fit to data and returned from
        the ``run`` function.

    cov : Covariance matrix, optional
        A covariance matrix to speed up convergence of the MCMC. If none is provided the
        MCMC will run without but it will be slower at converging.

    cov_params : List of strings, optional
        List of parameters corresponding to the rows and columns of the covariance matrix
        passed via ``cov``.

    verbose : int (default: 3)
        Verbosity of the MC sampler.

    Returns
    -------
    sampler : dict
        a dict with the ``sampler`` block for Cobaya's run function.
    """
    # Set the reference point of the prior to the sampled location with maximum
    # posterior value
    try:
        i_max_location = np.argsort(surrogate.y_regress)[-mpi.RANK - 1]
        max_location = surrogate.X_regress[i_max_location]
    except IndexError:  # more MPI processes than training points: sample from prior
        max_location = [None] * surrogate.d
    model.prior.set_reference(dict(zip(model.prior.params, max_location)))
    # Create sampler info
    sampler_info = {"mcmc": {"measure_speeds": False, "max_tries": 100000}}
    if (cov is None or not is_valid_covmat(cov)) and verbose >= 2:
        warnings.warn(
            "No covariance matrix or invalid one provided for the `mcmc` "
            "sampler. This will make the convergence of the sampler slower."
        )
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


def mc_sample_from_gp_cobaya(
    surrogate,
    bounds=None,
    params=None,
    sampler="mcmc",
    sampler_options=None,
    covmat=None,
    covmat_params=None,
    output=None,
    run=True,
    resume=False,
    verbose=3,
):
    """
    Generates a `Cobaya Sampler <https://cobaya.readthedocs.io/en/latest/sampler.html>`_
    and runs it on the surrogate model.

    Parameters
    ----------
    surrogate : SurrogateModel, which has been fit to data and returned from the
        ``run`` function.
        Alternatively a string containing a path with the location of a saved GP run
        (checkpoint) can be provided (the same path that was used to save the checkpoint
        in the ``run`` function).

    bounds : List of boundaries (lower,upper), optional
        By default it reads them from the GP regressor.

    params : List of parameter strings, optional
        By default it uses some dummy strings.

    true_model : Cobaya Model, optional
        If passed, it uses it to get bounds and parameter names (unless overriden by
        the corresponding kwargs).

    sampler : string (default `"mcmc"`). or dict
        Cobaya sampler to be used.

    sampler_options : dict, optional
        Dictionary of options to be passed to the sampler (see Cobaya documentation for
        the interface of that sampler).

    covmat: array, optional
        Approximate covariance matrix of the posterior to be used e.g. for the proposal
        distribution of an MCMC run.

    covmat_params: list of str, optional
        List of parameter names for the rows and columns of the passed ``covmat``, if
        different from ``params``, or in different order.

    output: path, optional
        The path where the resulting Monte Carlo sample shall be stored.

    run: bool, default: True
        Whether to run the sampler. If ``False``, returns just an initialised sampler.

    verbose: int (default 3)
        Verbosity level, similarly valued to that of the Runner, e.g. 3 indicates cobaya's
        'info' level, 4 the 'debug' level, and lower-than-three values print only warnings
        and errors.

    resume: bool, optional (default=False)
        Whether to resume from existing output files (True) or force overwrite (False)

    Returns
    -------
    surr_info : dict
        The dictionary that was used to run (or initialized) the sampler, corresponding to
        the surrogate model, and populated with the sampler input specification.

    sampler : Sampler instance
        The sampler instance that has been run (or just initialised). The sampler products
        can be retrieved with the `Sampler.products()` method.
    """
    from gpry import check_cobaya_installed

    if not check_cobaya_installed():
        raise ModuleNotFoundError(
            "You need to install Cobaya ('python -m pip install cobaya) in order to use "
            "Cobaya as a sampler."
        )
    from cobaya.model import get_model  # type: ignore
    from cobaya.output import get_output  # type: ignore
    from cobaya.sampler import get_sampler  # type: ignore

    if not isinstance(sampler, str):
        raise ValueError(
            "`sampler` must be a string specifying a Cobaya sampler interface."
        )
    sampler_options = sampler_options or {}
    _, surrogate, acquisition, convergence, _, _ = ensure_surrogate(surrogate)
    if surrogate is None:
        raise ValueError("Could not load the GP regressor from checkpoint")
    if not surrogate.fitted:
        raise ValueError("Cannot run an MC sampler on a GPR that has not been fitted.")
    # Prepare model
    model_input = cobaya_generate_surr_model_input(
        surrogate, bounds=bounds, params=params
    )
    model_input["debug"] = get_cobaya_log_level(verbose)
    model_surrogate = get_model(model_input)
    # Prepare covariance matrix -- prefer the one passed directly
    if covmat is not None:
        covariance_matrix = covmat
        if covmat_params is not None:
            covariance_params = covmat_params
        else:
            covariance_params = params
    else:
        if acquisition is not None:
            covariance_matrix = getattr(acquisition, "cov", None)
        if covariance_matrix is None and convergence is not None:
            covariance_matrix = getattr(convergence, "cov", None)
        covariance_params = params
    # Prepare rest of sampler input
    if sampler.lower() == "mcmc":
        sampler_input = mcmc_info_from_run(
            model_surrogate,
            surrogate,
            cov=covariance_matrix,
            cov_params=covariance_params,
            verbose=verbose,
        )
        # "ref" from available info (not used at the moment)
        # best_point_per_mpi_rank = \
        #     surrogate.X_regress[np.argsort(surrogate.y_regress)[-1 + mpi.RANK]]
        # ref = {
        #     p: val for p, val in zip(
        #         paramnames, best_point_per_mpi_rank
        #     )
        # }
    elif sampler.lower() == "polychord":
        if output is False:
            warnings.warn(
                "Polychord cannot run without output. Mind that it defaults "
                "to /tmp/polychord_raw"
            )
        sampler_input = polychord_info_from_run()
    else:
        sampler_input = {sampler: {"measure_speeds": False}}
    sampler_input[sampler].update(sampler_options or {})
    out = None
    if output is not None:
        if not resume:
            out = get_output(prefix=output, resume=False, force=True)
        else:
            out = get_output(prefix=output, resume=True, force=False)
    sampler = get_sampler(sampler_input, model=model_surrogate, output=out)
    surr_info = model_surrogate.info()
    if not run:
        surr_info["sampler"] = {sampler: sampler.info()}
        return surr_info, sampler
    sampler.run()
    surr_info["sampler"] = {sampler: sampler.info()}
    return surr_info, sampler


def mc_sample_from_gp_ns(
    surrogate,
    bounds=None,
    params=None,
    sampler=None,
    sampler_options=None,
    output=None,
    run=True,
    verbose=3,
):
    """
    Generates an MC sample of the surrogate model using one of the NS interfaces.

    Parameters
    ----------
    surrogate : SurrogateModel, which has been fit to data and returned from the
        ``run`` function.
        Alternatively a string containing a path with the location of a saved GP run
        (checkpoint) can be provided (the same path that was used to save the checkpoint
        in the ``run`` function).

    bounds : List of boundaries (lower,upper), optional
        By default it reads them from the GP regressor.

    params : List of parameter strings, optional
        By default it uses some dummy strings.

    sampler : string, optional
        Nested sampler to be used. If undefined, uses PolyChord if available, otherwise
        UltraNest.

    sampler_options : dict, optional
        Dictionary of options to be passed to the nested sampler.

    output: path, optional
        The path where the resulting Monte Carlo sample shall be stored.

    run: bool, default: True
        Whether to run the sampler. If ``False``, returns just an initialised sampler.

    verbose: int (default 3)
        Verbosity level, similarly valued to that of the Runner, e.g. 3 indicates normal
        output, and 4 'debug' level output; lower-than-three values print only warnings
        and errors.

    Returns
    -------
    (X_MC, y_MC, w_MC) : arrays of samples parameters, surrogate posteriors and weights
                         (None if equal weights).
    """
    # Prepare surrogate model
    _, surrogate, _, _, _, _ = ensure_surrogate(surrogate)
    if surrogate is None:
        raise ValueError("Could not load the GP regressor from checkpoint")
    if not surrogate.fitted:
        raise ValueError(
            "Cannot run an MC sampler on a surrogate model that has not been fitted."
        )
    if bounds is None:
        bounds = surrogate.trust_bounds

    def logp(X):
        y = surrogate.predict(np.atleast_2d(X), return_std=False, validate=False)
        if verbose >= 4:
            print(f"SurrogateModel: got {X}, mean GP prediction {y}")
        return y

    # Prepare and initialise sampler
    if sampler is None:
        sampler = "nested"
    if not isinstance(sampler, str):
        raise ValueError(
            "`sampler` must be a string specifying an interfaced nested sampler: "
            f"{list(nsint._ns_interfaces)}"
        )
    sampler_name = sampler.lower()
    if sampler_name == "nested":
        interface = nsint._ns_interfaces["polychord"]
    else:
        try:
            interface = nsint._ns_interfaces[sampler_name]
        except KeyError as kerr:
            raise ValueError(
                f"Nested sampler {sampler_name} unknown. Did you mean any of "
                f"{list(nsint._ns_interfaces)}?"
            ) from kerr
    try:
        sampler = interface(bounds, verbosity=verbose)
    except nsint.NestedSamplerNotInstalledError as excpt:
        # Exception: if "nested" passed, default to UltraNest
        if sampler_name == "nested":
            warnings.warn(
                f"Importing the default NS PolyChord failed (Err msg: {excpt}). "
                "Defaulting to UltraNest."
            )
            sampler = nsint._ns_interfaces["ultranest"](bounds, verbosity=verbose)
        else:
            raise excpt
    sampler.set_precision(**(sampler_options or {}))
    if not run:
        return sampler
    # Run sampler
    out_dir_raw = (
        tempfile.TemporaryDirectory().name + "/"
    )  # make sure it's read as a dir
    X_MC, y_MC, w_MC = sampler.run(logp, param_names=params, out_dir=out_dir_raw)
    # Delete the "raw" output and write the unified-format one
    sampler.delete_output()
    if output is not None and mpi.is_main_process:
        # Prepare file
        base_dir, file_name = os.path.split(output)
        if not base_dir:
            base_dir = os.path.curdir
        base_dir = os.path.abspath(base_dir)
        create_path(base_dir, verbose=False)
        if file_name == "":
            file_name = "mc_samples.txt"
        file_root, file_ext = os.path.splitext(file_name)
        if not file_ext:
            file_ext = ".txt"
        output = os.path.abspath(os.path.join(base_dir, file_root + file_ext))
        # Write file
        if params is None:
            params = generic_params_names(X_MC.shape[1])
        w_MC_write = w_MC if w_MC is not None else np.ones(shape=(1, len(y_MC)))
        np.savetxt(
            output,
            np.concatenate([np.atleast_2d(w_MC_write), np.atleast_2d(-y_MC), X_MC.T]).T,
            header="w minuslogp " + " ".join(params),
        )
    return X_MC, y_MC, w_MC


def process_gdsamples(gdsamples_dict):
    """
    Returns a dict with values as getdist.MCSamples, transforming/loading the original
    dict values as appropriate.
    """
    return_dict = {}
    for k, v in gdsamples_dict.items():
        if isinstance(v, str):
            root = os.path.abspath(v)
            if os.path.isdir(root):
                root += "/"  # to force GetDist to treat it as folder, not prefix
            return_dict[k] = loadMCSamples(root)
        elif isinstance(v, (MCSamples, GaussianND)):
            return_dict[k] = v
        else:
            if check_cobaya_installed():
                from cobaya.collection import SampleCollection  # type: ignore

                if isinstance(v, SampleCollection):
                    return_dict[k] = v.to_getdist(label=k)
            raise ValueError(
                f"I don't know how to transform object of type {type(v)} "
                "into getdist.MCSamples."
            )
    return return_dict


def samples_dict_to_getdist(samples_dict, params=None, bounds=None, sampler_type=None):
    """
    Expects ``samples_dict`` with keys ``w``, ``X``, ``logpost``, ``logprior`` (optional),
    ``loglike`` (optional).

    ``params`` should be a list of strings, or of tuples ``(name, latex_label)`` where the
    ``latex_label`` should **not** include the ``$`` delimiters.

    ``bounds`` should be a list of boundaries for the parameters.)

    ``sampler_type`` should be ``nested`` or ``mcmc``.
    """
    from getdist import MCSamples  # type: ignore

    if params is None:
        params = generic_params_names(len(samples_dict["X"][0]))
    params_list = []
    labels_list = []
    for i in range(len(params)):
        if isinstance(params[0], str):
            params_list.append(params[i])
            labels_list.append(params[i])
        else:  # assume tuple
            params_list.append(params[i][0])
            labels_list.append(params[i][1])
    mlogp = samples_dict.get(_name_logp)
    if mlogp is not None:
        mlogp = -1 * mlogp
    pnames = (_name_logp, _name_logprior, _name_loglike)
    plabels = (_label_logp, _label_logprior, _label_loglike)
    samples = np.copy(samples_dict["X"])
    for n, label in zip(pnames, plabels):
        y = samples_dict.get(n)
        if y is not None and not np.isclose(max(y) - min(y), 0):
            samples = np.concatenate([samples.T, [y]]).T
            params_list.append(n + "*")
            labels_list.append(label)
    mcsamples = MCSamples(
        samples=samples,
        weights=samples_dict["w"],
        loglikes=mlogp,
        names=params_list,
        labels=labels_list,
        ranges=dict(zip(params_list, bounds)),
        sampler=sampler_type,
        ignore_rows=0,
    )
    return mcsamples
