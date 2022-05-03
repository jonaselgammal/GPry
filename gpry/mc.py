from gpry.gpr import GaussianProcessRegressor
from cobaya.model import get_model
from cobaya.output import get_output
from cobaya.sampler import get_sampler
from gpry.tools import cobaya_generate_gp_input, mcmc_info_from_run, \
    polychord_info_from_run

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
