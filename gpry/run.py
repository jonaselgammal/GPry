"""
Top level run file which constructs the loop for mapping a posterior
distribution and sample the GP to get chains.
"""

from gpry.mpi import mpi_comm, mpi_size, mpi_rank, is_main_process, get_random_state, \
    split_number_for_parallel_processes, multiple_processes
from gpry.gpr import GaussianProcessRegressor
from gpry.gp_acquisition import GP_Acquisition
from gpry.svm import SVM
from gpry.preprocessing import Normalize_bounds, Normalize_y
from gpry.tools import cobaya_gp_model_input, mcmc_info_from_run, polychord_info_from_run
import gpry.convergence as gpryconv
from cobaya.model import Model, get_model
from cobaya.output import get_output
from cobaya.sampler import get_sampler
from copy import deepcopy
import numpy as np
import warnings
import os
import pandas as pd
import time

def run(model, gp="RBF", gp_acquisition="Log_exp",
        convergence_criterion="CorrectCounter",
        callback=None,
        convergence_options=None, options={}, checkpoint=None, verbose=3):
    """
    This function takes care of constructing the Bayesian quadrature/likelihood
    characterization loop. This is the easiest way to make use of the
    gpry algorithm. The minimum requirements for running this are a Cobaya
    prior object and a likelihood. Furthermore the details of the GP and
    and acquisition can be specified by the user.

    Parameters
    ----------

    model : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_
        Contains all information about the parameters in the likelihood and
        their priors as well as the likelihood itself. Cobaya is only used here
        as a wrapper to get the logposterior etc.

    gp : GaussianProcessRegressor, "RBF" or "Matern", optional (default="RBF")
        The GP used for interpolating the posterior. If None or "RBF" is given
        a GP with a constant kernel multiplied with an anisotropic RBF kernel
        and dynamic bounds is generated. The same kernel with a Matern 3/2
        kernel instead of a RBF is generated if "Matern" is passed. This might
        be useful if the posterior is not very smooth.
        Otherwise a custom GP regressor can be created and passed.

    gp_acquisition : GP_Acquisition, optional (default="Log_exp")
        The acquisition object. If None is given the Log_exp acquisition
        function is used (with the :math:`\zeta` value chosen automatically
        depending on the dimensionality of the prior) and the GP's X-values are
        preprocessed to be in the uniform hypercube before optimizing the
        acquistion function.

    convergence_criterion : Convergence_criterion, optional (default="KL")
        The convergence criterion. If None is given the KL-divergence between
        consecutive runs is calculated with an MCMC run and the run converges
        if KL<0.02 for two consecutive steps.

    convergence_option: optional parameters passed to the convergence criterion.

    options : dict, optional (default=None)
        A dict containing all options regarding the bayesian optimization loop.
        The available options are:

            * n_initial : Number of initial samples before starting the BO loop
              (default: 3*number of dimensions)
            * n_points_per_acq : Number of points which are aquired with
              Kriging believer for every acquisition step (default: equals the
              number of parallel processes)
            * max_points : Maximum number of attempted sampling points before the run fails.
              This is useful if you e.g. want to restrict the maximum computation resources (default: 1000)
            * max_accepted : Maximum number of accepted sampling points before the run fails.
              This might be useful if you use the DontConverge convergence criterion,
              specifying exactly how many points you want to have in your GP (default: max_points)
            * max_init : Maximum number of points drawn at initialization
              before the run fails (default: 10 * number of dimensions * n_initial).
              If the run fails repeatadly at initialization try decreasing the volume
              of your prior.

    callback: callable, optional (default=None)
        Function run each iteration after adapting the recently acquired points and
        the computation of the convergence criterion. This function should take arguments
        ``callback(model, current_gpr, gp_acquistion, convergence_criterion, options,
        previous_gpr, new_X, new_y, pred_y)``.
        When running in parallel, the function is run by the main process only.

    checkpoint : str, optional (default=None)
        Path for storing checkpointing information from which to resume in case the
        algorithm crashes. If None is given no checkpoint is saved.

    verbose : 1, 2, 3, optional (default: 3)
        Level of verbosity. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise. Is passed to the GP, Acquisition and Convergence
        criterion if they are built automatically.

    Returns
    -------

    model : Cobaya model
        The model that was used to run the GP on (if running in parallel, needs to be
        passed for all processes).

    gp : GaussianProcessRegressor
        This can be used to call an MCMC sampler for getting marginalized
        properties. This is the most crucial component.

    gp_acquisition : GP_acquisition
        The acquisition object that was used for the active sampling procedure.

    convergence_criterion : Convergence_criterion
        The convergence criterion used for determining convergence. Depending
        on the criterion used this also contains the approximate covariance
        matrix of the posterior distribution which can be used by the MCMC
        sampler.

    options : dict
        The options dict used for the active sampling loop.
    """
    if is_main_process:
        # Check if a checkpoint exists already and if so resume from there
        if checkpoint is not None and verbose > 2:
            print("Checking for checkpoint to resume from...")
        checkpoint_files = _check_checkpoint(checkpoint)
        comes_from_checkpoint = np.all(checkpoint_files)
        if comes_from_checkpoint:
            model, gpr, acquisition, convergence, options = _read_checkpoint(
                checkpoint)
            n_d = model.prior.d()
            if verbose > 2:
                print("#########################################")
                print("Checkpoint found. Resuming from there...")
                print("If this behaviour is unintentional either")
                print("turn the checkpoint option off or rename it")
                print("to a file which doesn't exist.")
                print("#########################################")
        else:
            if np.any(checkpoint_files) and verbose > 1:
                print("warning: Found checkpoint files but they were "
                      "incomplete. Ignoring them...")

            # Check model
            if not isinstance(model, Model):
                raise TypeError(f"'model' needs to be a Cobaya model. got {model}")
            try:
                n_d = model.prior.d()
                prior_bounds = model.prior.bounds(confidence_for_unbounded=0.99995)
            except:
                raise RuntimeError("There seems to be something wrong with "
                                   "the model instance...")

            # Construct GP if it's not already constructed
            if isinstance(gp, str):
                if gp not in ["RBF", "Matern"]:
                    raise ValueError("Supported standard kernels are 'RBF' "
                                     f"and Matern, got {gp}")
                gpr = GaussianProcessRegressor(
                    kernel=gp,
                    n_restarts_optimizer=10 + 2 * n_d,
                    preprocessing_X=Normalize_bounds(prior_bounds),
                    preprocessing_y=Normalize_y(),
                    bounds=prior_bounds,
                    verbose=verbose
                )

            elif isinstance(gp, GaussianProcessRegressor):
                gpr = gp
            else:
                raise TypeError("gp should be a GP regressor, 'RBF' or 'Matern'"
                                f", got {gp}")

            # Construct the acquisition object if it's not already constructed
            if isinstance(gp_acquisition, str):
                if gp_acquisition not in ["Log_exp"]:
                    raise ValueError("Supported acquisition function is "
                                     f"'Log_exp', got {gp_acquisition}")

                bounds = model.prior.bounds(confidence_for_unbounded=0.99995)
                acquisition = GP_Acquisition(bounds,
                                             proposer=None,
                                             acq_func=gp_acquisition,
                                             acq_optimizer="fmin_l_bfgs_b",
                                             n_restarts_optimizer=5 * n_d,
                                             n_repeats_propose=10,
                                             preprocessing_X=Normalize_bounds(
                                                 prior_bounds),
                                             zeta_scaling=options.get("zeta_scaling",1.1),
                                             verbose=verbose)
            elif isinstance(gp_acquisition, GP_Acquisition):
                acquisition = gp_acquisition
            else:
                raise TypeError("gp_acquisition should be an Acquisition "
                                f"object or 'Log_exp', got {gp_acquisition}")

            # Construct the convergence criterion
            if isinstance(convergence_criterion, str):
                try:
                    conv_class = getattr(gpryconv, convergence_criterion)
                except AttributeError:
                    raise ValueError(
                        f"Unknown convergence criterion {convergence_criterion}. "
                        f"Available convergence criteria: {gpryconv.builtin_names()}")
                convergence = conv_class(model.prior, convergence_options or {})
            elif isinstance(convergence_criterion, gpryconv.ConvergenceCriterion):
                convergence = convergence_criterion
            else:
                raise TypeError("convergence_criterion should be a "
                                "Convergence_criterion object or "
                                f"{gpryconv.builtin_names()}, got "
                                f"{convergence_criterion}")

        # Read in options for the run
        if options is None:
            if verbose > 2:
                print("No options dict found. Defaulting to standard parameters.")
            options = {}
        n_initial = options.get("n_initial", 3 * n_d)
        max_points = options.get("max_points", 1000)
        max_accepted = options.get("max_accepted", max_points)
        max_init = options.get("max_init", 10 * n_d * n_initial)
        n_points_per_acq = options.get("n_points_per_acq", mpi_size)
        fit_full_every = options.get("fit_full_every", max(int(2 * np.sqrt(n_d)), 1))
        if n_points_per_acq < mpi_size and verbose > 1:
            print("Warning: parallellisation not fully utilised! It is advised to make "
                  "n_points_per_acq equal to the number of MPI processes (default when "
                  "not specified.")
        if n_points_per_acq > 2 * n_d and verbose > 1:
            print("Warning: The number kriging believer samples per "
                  "acquisition step is larger than 2x number of dimensions of "
                  "the feature space. This may lead to slow convergence."
                  "Consider running it with less cores or decreasing "
                  "n_points_per_acq manually.")

        # Sanity checks
        if n_initial >= max_points:
            raise ValueError("The number of initial samples needs to be "
                             "smaller than the maximum number of points")
        if n_initial <= 0:
            raise ValueError("The number of initial samples needs to be bigger "
                             "than 0")
        if max_accepted > max_points:
            raise ValueError("You manually set max_accepted > max_points, but "
                             " you cannot have more accepted than sampled points")

        # Print resume
        if verbose > 2:
            print("Initialized GPry.")
            if not comes_from_checkpoint:
                print("Starting by drawing initial samples.")
    if multiple_processes:
        n_initial, max_init, max_points, max_accepted, n_points_per_acq = mpi_comm.bcast(
            (n_initial, max_init, max_points, max_accepted, n_points_per_acq)
            if is_main_process else None)
        gpr = mpi_comm.bcast(gpr if is_main_process else None)
        acquisition = mpi_comm.bcast(acquisition if is_main_process else None)
        convergence_is_MPI_aware = mpi_comm.bcast(
            convergence.is_MPI_aware if is_main_process else None)
        if convergence_is_MPI_aware:
            convergence = mpi_comm.bcast(convergence if is_main_process else None)
        comes_from_checkpoint = mpi_comm.bcast(
            comes_from_checkpoint if is_main_process else None)
    else:
        convergence_is_MPI_aware = convergence.is_MPI_aware

    if not comes_from_checkpoint:
        # Define initial tranining set
        get_initial_sample(model, gpr, n_initial, verbose=verbose)
        if is_main_process:
            # Save checkpoint
            _save_checkpoint(checkpoint, model, gpr, acquisition, convergence, options)
            if verbose > 2:
                print("Initial samples drawn, starting with Bayesian "
                      "optimization loop.")
    if multiple_processes:
        n_finite = mpi_comm.bcast(len(gpr.y_train) if is_main_process else None)
    else:
        n_finite = len(gpr.y_train)
    # Prepare progress summary table; the table key is the iteration number
    progress = Progress()
    # Run bayesian optimization loop
    n_iterations = int((max_points - n_finite) / n_points_per_acq)
    n_evals_per_acq_per_process = \
        split_number_for_parallel_processes(n_points_per_acq)
    n_evals_this_process = n_evals_per_acq_per_process[mpi_rank]
    i_evals_this_process = sum(n_evals_per_acq_per_process[:mpi_rank])
    it = 0
    n_left = max_accepted - n_finite
    for it in range(n_iterations):
        progress.add_iteration()
        if is_main_process:
            if verbose > 2:
                if max_accepted != max_points:
                    print(f"+++ Iteration {it} "
                          f"(Accepting at most {n_left} more points) +++++++++")
                else:
                    print(f"+++ Iteration {it} "
                          f"(of at most {n_iterations} iterations) +++++++++")
            # Save old gp for convergence criterion
            old_gpr = deepcopy(gpr)
        gpr = mpi_comm.bcast(gpr if is_main_process else None)
        # Acquire new points in parallel with MPI-aware random state
        with TimerCounter(gpr) as timer_acq:
            new_X, y_pred, acq_vals = acquisition.multi_add(
                gpr, n_points=n_points_per_acq, random_state=get_random_state())
        progress.add_acquisition(timer_acq.time, timer_acq.evals)
        if is_main_process:
            print("run.py :: New X/y_lie/acq = ", new_X, y_pred, acq_vals)
        # Get logposterior value(s) for the acquired points (in parallel)
        if multiple_processes:
            new_X = mpi_comm.bcast(new_X if is_main_process else None)
        new_X_this_process = new_X[
            i_evals_this_process: i_evals_this_process + n_evals_this_process]
        new_y = np.empty(0)
        with Timer() as timer_truth:
            for x in new_X_this_process:
                new_y = np.append(new_y, model.logpost(x))
        progress.add_truth(timer_truth.time, len(x))
        # Collect (if parallel) and append to the current model
        if multiple_processes:
            all_new_y = mpi_comm.gather(new_y)
        else:
            all_new_y = [new_y]
        if is_main_process:
            new_y = np.concatenate(all_new_y)
            do_simplified_fit = (it % fit_full_every != fit_full_every - 1)
            with TimerCounter(gpr) as timer_fit:
                gpr.append_to_data(new_X, new_y,
                                   fit=True, simplified_fit=do_simplified_fit)
            progress.add_fit(timer_fit.time, timer_fit.evals)
            n_left = max_accepted - gpr.n_accepted_evals
            ### TODO :: Possibly this callback should check whether it is MPI aware and be executed in MPI parallel. What happens if the callback doesn't work properly or is massively delayed. Can this cause MPI problems?
            if callback:
                callback(model, gpr, gp_acquisition, convergence, options,
                         old_gpr, new_X, new_y, y_pred)
        # Calculate convergence and break if the run has converged
        if not convergence_is_MPI_aware:
            if is_main_process:
                try:
                    with TimerCounter(gpr, old_gpr) as timer_convergence:
                        is_converged = convergence.is_converged(
                            gpr, old_gpr, new_X, new_y, y_pred)
                    progress.add_convergence(
                        timer_convergence.time, timer_convergence.evals)
                except gpryconv.ConvergenceCheckError:
                    is_converged = False
            if multiple_processes:
                is_converged = mpi_comm.bcast(is_converged if is_main_process else None)
        else:  # run by all processes
            # NB: this assumes that when the criterion fails,
            #     ALL processes raise ConvergenceCheckerror, not just rank 0
            if multiple_processes:
                gpr, old_gpr, new_X, new_y, y_pred = mpi_comm.bcast(
                    (gpr, old_gpr, new_X, new_y, y_pred) if is_main_process else None)
            try:
                with TimerCounter(gpr, old_gpr) as timer_convergence:
                    is_converged = convergence.is_converged(
                        gpr, old_gpr, new_X, new_y, y_pred)
                progress.add_convergence(timer_convergence.time, timer_convergence.evals)
            except gpryconv.ConvergenceCheckError:
                is_converged = False

        if is_converged:
            break
        # If the loop reaches n_left <= 0, then all processes need to break, not just the main process
        n_left = mpi_comm.bcast(n_left if is_main_process else None)
        if n_left<=0:
            break
        if is_main_process:
            # Save
            _save_checkpoint(checkpoint, model, gpr, acquisition, convergence, options)
        # progress.plot_timing(truth=False)
        # progress.plot_evals(truth=False)
    # Save
    if is_main_process:
        _save_checkpoint(checkpoint, model, gpr, acquisition, convergence, options)

    if n_left <= 0 and not isinstance(convergence, gpryconv.DontConverge) \
       and is_main_process and verbose > 1:
        warnings.warn("The maximum number of accepted points was reached before "
                      "convergence. Either increase max_accepted or try to "
                      "choose a smaller prior.")
    if it == n_iterations and is_main_process and verbose > 1:
        warnings.warn("Not enough points were accepted before "
                      "reaching convergence/reaching the specified max_points.")

    # Now that the run has converged we can return the gp and all other
    # relevant quantities which can then be processed with an MCMC or other
    # sampler
    if multiple_processes:
        gpr, acquisition, convergence, options = mpi_comm.bcast(
            (gpr, acquisition, convergence, options) if is_main_process else None)
    return model, gpr, acquisition, convergence, options


def get_initial_sample(model, gpr, n_initial, max_init=None, verbose=3):
    """
    Draws an initial sample for the `gpr` GP model until it has a training set of size
    `n_initial`, counting only finite-target points ("finite" here meaning over the
    threshold of the SVM classifier, if present).

    Parameters
    ----------

    This function is MPI-aware.

    model : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_
        Contains all information about the parameters in the likelihood and
        their priors as well as the likelihood itself. Cobaya is only used here
        as a wrapper to get the logposterior etc.

    gpr : GaussianProcessRegressor
        From the GP only the threshold of the SVM classifier is used to
        identify samples as "finite" and "infinite".

    n_initial : int
        The number of initial (finite) samples that shall be drawn from the
        log-posterior.

    max_init : int, optional (default=None)
        Will fail is it needed to evaluate the target more than `max_init`
        times (defaults to 10 times the dimension of the problem times the
        number of initial samples requested).

    verbose : 1, 2, 3, optional (default: 3)
        Level of verbosity. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise.

    Returns
    -------

    The gpr with the samples appended to (possibly already existing) samples
    and refit hyperparameters.
    """
    n_d = model.prior.d()
    max_init = max_init or 10 * n_d * n_initial
    # Check if there's an SVM and if so read out it's threshold value
    # We will compare it against y - max(y)
    if isinstance(gpr.account_for_inf, SVM):
        # Grab the threshold from the internal SVM (the non-preprocessed one)
        gpr.account_for_inf.update_threshold(dimension=model.prior.d())
        is_finite = lambda y: gpr.account_for_inf.is_finite(y, y_is_preprocessed=False)
    else:
        is_finite = lambda y: np.isfinite(y)
    if is_main_process:
        # Check if the GP already contains points. If so they are reused.
        pretrained = 0
        if hasattr(gpr, "y_train"):
            if len(gpr.y_train) > 0:
                pretrained = len(gpr.y_train)
        n_still_needed = n_initial - pretrained
        n_to_sample_per_process = int(np.ceil(n_still_needed / mpi_size))
        # Arrays to store the initial sample
        X_init = np.empty((0, n_d))
        y_init = np.empty(0)
    if multiple_processes:
        n_to_sample_per_process = mpi_comm.bcast(
            n_to_sample_per_process if is_main_process else None)
    if n_to_sample_per_process == 0 and verbose > 1:  # Enough pre-training
        warnings.warn("The number of pretrained points exceeds the number of initial samples")
        return
    n_iterations_before_giving_up = int(np.ceil(max_init / n_to_sample_per_process))
    # Initial samples loop. The initial samples are drawn from the prior
    # and according to the distribution of the prior.
    for i in range(n_iterations_before_giving_up):
        X_init_loop = np.empty((0, n_d))
        y_init_loop = np.empty(0)
        for j in range(n_to_sample_per_process):
            # Draw point from prior and evaluate logposterior at that point
            X = model.prior.reference(warn_if_no_ref=False)
            if verbose > 2:
                print(f"Evaluating true posterior at {X}")
            y = model.logpost(X)
            if verbose > 2:
                print(f"Got {y}")
            X_init_loop = np.append(X_init_loop, np.atleast_2d(X), axis=0)
            y_init_loop = np.append(y_init_loop, y)
        # Gather points and decide whether to break.
        if multiple_processes:
            all_points = mpi_comm.gather(X_init_loop)
            all_posts = mpi_comm.gather(y_init_loop)
        else:
            all_points = [X_init_loop]
            all_posts = [y_init_loop]
        if is_main_process:
            X_init = np.concatenate([X_init, np.concatenate(all_points)])
            y_init = np.concatenate([y_init, np.concatenate(all_posts)])
            # Only finite values contributes to the number of initial samples
            n_finite_new = sum(is_finite(y_init - max(y_init)))
            # Break loop if the desired number of initial samples is reached
            finished = (n_finite_new >= n_still_needed)
        if multiple_processes:
            finished = mpi_comm.bcast(finished if is_main_process else None)
        if finished:
            break
        else:
            # TODO: maybe re-fit SVM to shrink initial sample region
            pass
    if verbose > 2:
        print("Done getting initial training samples.")
    if is_main_process:
        # Append the initial samples to the gpr
        gpr.append_to_data(X_init, y_init)
        # Raise error if the number of initial samples hasn't been reached
        if not finished:
            raise RuntimeError("The desired number of finite initial "
                               "samples hasn't been reached. Try "
                               "increasing max_init or decreasing the "
                               "volume of the prior")

    return gpr


def mc_sample_from_gp(gp, bounds=None, paramnames=None, sampler="mcmc", convergence=None, options=None,
                      output=None, add_options=None, restart=False):
    """
    This function is essentially just a wrapper for the Cobaya MCMC sampler
    (monte python) which runs an MCMC on the fitted GP regressor. It returns
    the chains which can then be used with GetDist to get the triangle plots or
    be postprocessed in any other way.
    The plotting is explained in the
    `Cobaya documentation <https://cobaya.readthedocs.io/en/latest/example_advanced.html#from-the-shell>`_.

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

    updated_info : dict
        The (expanded) dictionary that was used to run the MCMC on the GP.

    sampler : Sampler instance
        The sampler instance contains the chains etc. and can be used for
        plotting etc.
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

    # Run the sampler on the GP
    print("Starting sampler")
    sampler.run()
    updated_info = model_surrogate.info()
    updated_info["sampler"] = {list(sampler_info)[0]: sampler.info()}

    return updated_info, sampler


# FOR BACKWARDS COMPATIBILITY --> DELETE AT SOME POINT BEFORE RELEASE!
def mcmc(model_truth, gp, convergence=None, options=None, output=None, add_options=None):
    return mc_sample_from_gp(gp, model_truth.prior.params, model_truth.prior.bounds, sampler="mcmc", convergence=None,
                             options=None, output=None, add_options=None)


def _save_checkpoint(path, model, gp, gp_acquisition, convergence_criterion, options):
    """
    This function is used to save all relevant parts of the GP loop for reuse
    as checkpoint in case the procedure crashes.
    This function saves 5 files as .pkl files which contain the instances
    of the different modules.
    The files can be loaded with the _read_checkpoint function.

    Parameters
    ----------

    path : The path where the files shall be saved
        The files will be saved as *path* +(mod, gpr, acq, con, opt).pkl

    model : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_

    gp : GaussianProcessRegressor

    gp_acquisition : GP_Acquisition

    convergence_criterion : Convergence_criterion

    options : dict
    """
    try:
      import dill as pickle
    except ImportError as e:
      raise ImportError("Could not find the 'dill' package. This is not a strict requirement for gpry, but without it the checkpoint functionality does not work.") from e
    if path is not None:
        try:
            with open(os.path.join(path, "mod.pkl"), 'wb') as f:
                # Save model as dict
                model_dict = model.info()
                pickle.dump(model_dict, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path, "gpr.pkl"), 'wb') as f:
                pickle.dump(gp, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path, "acq.pkl"), 'wb') as f:
                pickle.dump(gp_acquisition, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path, "con.pkl"), 'wb') as f:
                # Need to delete the prior object in convergence so it doesn't
                # do weird stuff while pickling
                convergence = deepcopy(convergence_criterion)
                convergence.prior = None
                pickle.dump(convergence, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path, "opt.pkl"), 'wb') as f:
                pickle.dump(options, f, pickle.HIGHEST_PROTOCOL)
        except Exception as excpt:
            raise RuntimeError("Couldn't save the checkpoint. Check if the path "
                               "is correct and exists. Error message: " + str(excpt))


def _check_checkpoint(path):
    """
    Checks if there are checkpoint files in a specific location and if so if they
    are complete. Returns a list of bools.

    Parameters
    ----------

    path : The path where the files are located

    Returns
    -------

    A boolean array containing whether the files exist in the specified
    location in the following order:
    [model, gp, acquisition, convergence, options]
    """
    if path is not None:
        checkpoint_files = [os.path.exists(os.path.join(path, "mod.pkl")),
                            os.path.exists(os.path.join(path, "gpr.pkl")),
                            os.path.exists(os.path.join(path, "acq.pkl")),
                            os.path.exists(os.path.join(path, "con.pkl")),
                            os.path.exists(os.path.join(path, "opt.pkl"))]
    else:
        checkpoint_files = [False] * 5
    return checkpoint_files


def _read_checkpoint(path):
    """
    Loads checkpoint files to be able to resume a run or save the results for
    further processing.

    Parameters
    ----------

    path : The path where the files are located

    Returns
    -------

    model, gp, acquisition, convergence, options.
    If any of the files does not exist or cannot be read the function will
    return None instead.
    """
    try:
      import dill as pickle
    except ImportError as e:
      raise ImportError("Could not find the 'dill' package. This is not a strict requirement for gpry, but without it the checkpoint functionality does not work.") from e
    # Check if a file exists in the checkpoint and if so resume from there.
    checkpoint_files = _check_checkpoint(path)
    # Read in checkpoint
    with open(os.path.join(path, "mod.pkl"), 'rb') as i:
        model = pickle.load(i) if checkpoint_files[0] else None
        # Convert model from dict to model object
        model = get_model(model)
    with open(os.path.join(path, "gpr.pkl"), 'rb') as i:
        gpr = pickle.load(i) if checkpoint_files[1] else None
    with open(os.path.join(path, "acq.pkl"), 'rb') as i:
        acquisition = pickle.load(i) if checkpoint_files[2] else None
    with open(os.path.join(path, "con.pkl"), 'rb') as i:
        if checkpoint_files[3]:
            convergence = pickle.load(i)
            convergence.prior = model.prior
        else:
            convergence = None

    with open(os.path.join(path, "opt.pkl"), 'rb') as i:
        options = pickle.load(i) if checkpoint_files[4] else None

    return model, gpr, acquisition, convergence, options


class Progress:
    """
    Pandas DataFrame to store progress, timing, numbers of evaluations, etc.
    """
    _colnames = {
        "n_train": "number of training points at the start of the iteration",
        "time_acquire": "time needed to acquire candidates for truth evaluation",
        "evals_acquire": ("number of evaluations of the GP needed to acquire candidates "
                          "for truth evaluation"),
        "time_truth": "time needed to evaluate the true model at the candidate points",
        "evals_truth": "number of evaluations of the true model",
        "time_fit": "time of refitting of the GP model after adding new training points",
        "evals_fit": ("number of evaluations of the GP during refitting after adding new"
                      "training points"),
        "time_convergence": "time needed to compute the convergence criterion",
        "evals_convergence": ("number of evaluations of the GP needed to compute the "
                              "convergence criterion"),
        "converge_crit_value": "value of the convergence criterion",
        "n_accepted": "???"}


    def __init__(self):
        """Initialises Progress table."""
        self.data = pd.DataFrame(columns=list(self._colnames))

    def __repr__(self):
        return self.data.__repr__()

    def help_column_names(self):
        """Prints names and description of columns."""
        print(self._colnames)

    def add_iteration(self):
        """
        Adds the next row to the table. New values will be added to this row.
        """
        self.data = self.data.append(pd.Series(dtype=float), ignore_index=True)

    def add_acquisition(self, time, evals):
        """Adds timing and #evals during acquisitions."""
        self.data.iloc[-1]["time_acquire"] = time
        self.data.iloc[-1]["evals_acquire"] = evals

    def add_truth(self, time, evals):
        """Adds timing and #evals during truth evaluations."""
        self.data.iloc[-1]["time_truth"] = time
        self.data.iloc[-1]["evals_truth"] = evals

    def add_fit(self, time, evals):
        """Adds timing and #evals during GP fitting."""
        self.data.iloc[-1]["time_fit"] = time
        self.data.iloc[-1]["evals_fit"] = evals

    def add_convergence(self, time, evals):
        """Adds timing and #evals during convergence computation."""
        self.data.iloc[-1]["time_convergence"] = time
        self.data.iloc[-1]["evals_convergence"] = evals

    def plot_timing(self, truth=True):
        """
        Plots as stacked bars the timing of each part of each iteration.

        Pass ``truth=False`` (default: True) to exclude the computation time of the true
        posterior at training points, for e.g. overhead-only plots.
        """
        import matplotlib.pyplot as plt
        plt.figure()
        # cast x values into list, to prevent finer x ticks
        iters = [str(i) for i in self.data.index.to_numpy(int)]
        bottom = np.zeros(len(self.data.index))
        for col, label in {
                "time_acquire": "Acquisition",
                "time_truth": "Truth",
                "time_fit": "GP fit",
                "time_convergence": "Convergence crit."}.items():
            if not truth and col == "time_truth":
                continue
            plt.bar(iters, self.data[col], label=label, bottom=bottom)
            bottom += self.data[col].to_numpy(dtype=float)
        plt.xlabel("Iteration")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.show(block=False)

    def plot_evals(self, truth=True):
        """
        Plots as stacked bars the number of evaluations of each part of each iteration.

        Pass ``truth=False`` (default: True) to exclude the number of evaluations of the
        true posterior at training points, for e.g. overhead-only plots.
        """
        import matplotlib.pyplot as plt
        plt.figure()
        # cast x values into list, to prevent finer x ticks
        iters = [str(i) for i in self.data.index.to_numpy(int)]
        bottom = np.zeros(len(self.data.index))
        for col, label in {
                "evals_acquire": "Acquisition",
                "evals_truth": "Truth",
                "evals_fit": "GP fit",
                "evals_convergence": "Convergence crit."}.items():
            if not truth and col == "evals_truth":
                continue
            plt.bar(iters, self.data[col], label=label, bottom=bottom)
            bottom += self.data[col].to_numpy(dtype=float)
        plt.xlabel("Iteration")
        plt.ylabel("Number of evaluations")
        plt.legend()
        plt.show(block=False)


class Timer:
    """Class for timing code within ``with`` block."""

    def __enter__(self):
        """Saves initial wallclock time."""
        self.start = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        """Saves final wallclock time and difference."""
        self.end = time.time()
        self.time = self.end - self.start


class TimerCounter(Timer):
    """
    Class for timing code within ``with`` block, and count number of evaluations of a
    given GP model.
    """

    def __init__(self, *gps):
        """Takes the GP's whose evaluations will be counted."""
        self.gps = gps  # save references for use at exit

    def __enter__(self):
        """Saves initial wallclock time and number of evaluations."""
        super().__enter__()
        self.init_eval = np.array([gp.n_eval for gp in self.gps], dtype=int)
        return self

    def __exit__(self, *args, **kwargs):
        """Saves final wallclock time and number of evaluations, and their differences."""
        super().__exit__()
        self.final_eval = np.array([gp.n_eval for gp in self.gps], dtype=int)
        self.evals = sum(self.final_eval - self.init_eval)


class Runner(object):
    def __init__(self):
        self.gp = None
        self.gp_acquisition = None
        self.convergence = None
        self.options = None
        self.paramnames = None
        self.bounds = None
        self.has_run = False
    def run(self, model, gp="RBF", gp_acquisition="Log_exp",
            convergence_criterion="CorrectCounter",
            callback=None,
            convergence_options=None, options={}, checkpoint=None, verbose=3):
        model, gpr, acquisition, convergence, options = run(model, gp=gp, gp_acquisition=gp_acquisition,
            convergence_criterion=convergence_criterion,
            callback=callback,
            convergence_options=convergence_options, options=options, checkpoint=checkpoint, verbose=verbose)
        self.gp = gpr
        self.gp_acquisition = acquisition
        self.convergence = convergence
        self.options = options
        self.paramnames = model.prior.params
        self.bounds = model.prior.bounds
        self.has_run = True
    def generate_mc_sample(sampler="mcmc", output=None, add_options=None, restart=False):
        if not self.has_run:
            raise Exception("You have to first run before you can generate an mc_sample")
        return mc_sample_from_gp(self.gp, bounds=self.bounds, paramnames=self.paramnames, sampler=sampler, convergence=self.convergence, options=self.options, output=output, add_options=add_options, restart=restart)
