import os
import warnings
import numpy as np
from copy import deepcopy
from itertools import chain
from inspect import getfullargspec

from cobaya.model import Model

from gpry.mpi import mpi_comm, mpi_size, mpi_rank, is_main_process, get_random_state, \
    split_number_for_parallel_processes, multiple_processes, sync_processes
from gpry.gpr import GaussianProcessRegressor
from gpry.gp_acquisition import GP_Acquisition
from gpry.svm import SVM
from gpry.preprocessing import Normalize_bounds, Normalize_y
import gpry.convergence as gpryconv
from gpry.progress import Progress, Timer, TimerCounter
from gpry.io import create_path, check_checkpoint, read_checkpoint, save_checkpoint
from gpry.mc import mc_sample_from_gp
from gpry.plots import plot_convergence


class Runner(object):
    r"""
    Class that takes care of constructing the Bayesian quadrature/likelihood
    characterization loop. After initialisation, the algorithm can be launched with
    :func:`Runner.run`, and, optionally after that, an MC process can be launched on the
    surrogate model with :func:`Runner.generate_mc_sample`.

    Parameters
    ----------
    model : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_
        Contains all information about the parameters in the likelihood and
        their priors as well as the likelihood itself. Cobaya is only used here
        as a wrapper to get the logposterior etc.

    gpr : GaussianProcessRegressor, "RBF" or "Matern", optional (default="RBF")
        The GP used for interpolating the posterior. If None or "RBF" is given
        a GP with a constant kernel multiplied with an anisotropic RBF kernel
        and dynamic bounds is generated. The same kernel with a Matern 3/2
        kernel instead of a RBF is generated if "Matern" is passed. This might
        be useful if the posterior is not very smooth.
        Otherwise a custom GP regressor can be created and passed.

    gp_acquisition : GP_Acquisition, optional (default="LogExp")
        The acquisition object. If None is given the LogExp acquisition
        function is used (with the :math:`\zeta` value chosen automatically
        depending on the dimensionality of the prior) and the GP's X-values are
        preprocessed to be in the uniform hypercube before optimizing the
        acquistion function.

    convergence_criterion : Convergence_criterion, optional (default="CorrectCounter")
        The convergence criterion. If None is given the Correct counter convergence
        criterion is used with a relative threshold of 0.01 and an absolute threshold of
        0.05.

    convergence_options : optional parameters passed to the convergence criterion.

    options : dict, optional (default=None)
        A dict containing all options regarding the bayesian optimization loop.
        The available options are:

            * n_initial : Number of initial samples before starting the BO loop
              (default: 3*number of dimensions)
            * n_points_per_acq : Number of points which are aquired with
              Kriging believer for every acquisition step (default: equals the
              number of parallel processes)
            * max_points : Maximum number of attempted sampling points before the run
              fails. This is useful if you e.g. want to restrict the maximum computation
              resources (default: 1000).
            * max_accepted : Maximum number of accepted sampling points before the run
              fails. This might be useful if you use the DontConverge convergence
              criterion, specifying exactly how many points you want to have in your GP
              (default: max_points)
            * max_init : Maximum number of points drawn at initialization
              before the run fails (default: 10 * number of dimensions * n_initial).
              If the run fails repeatadly at initialization try decreasing the volume
              of your prior.

    callback : callable, optional (default=None)
        Function run each iteration after adapting the recently acquired points and
        the computation of the convergence criterion. This function should take arguments
        ``callback(model, current_gpr, gp_acquistion, convergence_criterion, options, progress, previous_gpr, new_X, new_y, pred_y)``, or simply ``callback(runner_instance)``.
        When running in parallel, the function is run by the main process only, unless
        ``callback_is_MPI_aware=True``.

    callback_is_MPI_aware : bool (default: False)
        If True, the callback function is called for every process simultaneously, and
        it is expected to handle parallelisation internally. If false, only the main
        process calls it.

    checkpoint : str, optional (default=None)
        Path for storing checkpointing information from which to resume in case the
        algorithm crashes. If None is given no checkpoint is saved.

    load_checkpoint: "resume" or "overwrite", must be specified if path is not None.
        Whether to resume from the checkpoint files if existing ones are found
        at the location specified by `checkpoint`.

    plots : bool (default: True)
        If True, produces some progress plots.

    verbose : 1, 2, 3, optional (default: 3)
        Level of verbosity. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise. Is passed to the GP, Acquisition and Convergence
        criterion if they are built automatically.

    Attributes
    ----------
    model : Cobaya model
        The model that was used to run the GP on (if running in parallel, needs to be
        passed for all processes).

    gpr : GaussianProcessRegressor
        This can be used to call an MCMC sampler for getting marginalized
        properties. This is the most crucial component.

    gp_acquisition : GP_Acquisition
        The acquisition object that was used for the active sampling procedure.

    convergence_criterion : Convergence_criterion
        The convergence criterion used for determining convergence. Depending
        on the criterion used this also contains the approximate covariance
        matrix of the posterior distribution which can be used by the MCMC
        sampler.

    options : dict
        The options dict used for the active sampling loop.

    progress : Progress
        Object containing per-iteration progress information: number of accepted training
        points, number of GP evaluations, timing of different parts of the algorithm, and
        value of the convergence criterion.
    """

    def __init__(self, model, gpr="RBF", gp_acquisition="LogExp",
                 convergence_criterion="CorrectCounter", callback=None,
                 callback_is_MPI_aware=False, convergence_options=None, options={},
                 checkpoint=None, load_checkpoint=None, plots=True, verbose=3):
        self.model = model
        self.checkpoint = checkpoint
        if self.checkpoint is not None:
            self.plots_path = os.path.join(self.checkpoint, _plots_path)
            if is_main_process:
                create_path(self.checkpoint, verbose=verbose >= 3)
                create_path(self.plots_path, verbose=verbose >= 3)
        else:
            self.plots_path = _plots_path
            if is_main_process:
                create_path(self.plots_path, verbose=verbose >= 3)
        self.options = options
        self.plots = plots
        self.verbose = verbose
        self.rng = get_random_state()
        if is_main_process:
            # Check if a checkpoint exists already and if so resume from there
            self.loaded_from_checkpoint = False
            if checkpoint is not None:
                if load_checkpoint not in ["resume", "overwrite"]:
                    raise ValueError("If a checkpoint location is specified you need to "
                                     "set 'load_checkpoint' to 'resume' or 'overwrite'.")
                if load_checkpoint == "resume":
                    self.log("Checking for checkpoint to resume from...", level=3)
                    checkpoint_files = check_checkpoint(checkpoint)
                    self.loaded_from_checkpoint = np.all(checkpoint_files)
                    if self.loaded_from_checkpoint:
                        self.read_checkpoint()
                        # Overwrite internal parameters by those loaded from checkpoint.
                        model, gpr, gp_acquisition, convergence_criterion, options = \
                            self.model, self.gpr, self.acquisition, self.convergence, \
                            self.options
                        self.log("#########################################\n"
                                 "Checkpoint found. Resuming from there...\n"
                                 "If this behaviour is unintentional either\n"
                                 "turn the checkpoint option off or rename it\n"
                                 "to a file which doesn't exist.\n"
                                 "#########################################\n", level=3)
                    else:
                        if np.any(checkpoint_files):
                            self.log("warning: Found checkpoint files but they were "
                                     "incomplete. Ignoring them...", level=2)
            # Check model
            if not isinstance(model, Model):
                raise TypeError(f"'model' needs to be a Cobaya model. got {model}")
            try:
                prior_bounds = model.prior.bounds(confidence_for_unbounded=0.99995)
            except Exception as excpt:
                raise RuntimeError("There seems to be something wrong with "
                                   f"the model instance: {excpt}")
            # Construct GP if it's not already constructed
            if isinstance(gpr, str):
                if gpr not in ["RBF", "Matern"]:
                    raise ValueError("Supported standard kernels are 'RBF' "
                                     f"and Matern, got {gpr}")
                self.gpr = GaussianProcessRegressor(
                    kernel=gpr,
                    n_restarts_optimizer=10 + 2 * self.d,
                    preprocessing_X=Normalize_bounds(prior_bounds),
                    preprocessing_y=Normalize_y(),
                    bounds=prior_bounds,
                    verbose=verbose
                )
            elif not isinstance(gpr, GaussianProcessRegressor):
                raise TypeError("gpr should be a GP regressor, 'RBF' or 'Matern'"
                                f", got {gpr}")
            else:
                self.gpr = gpr
            # Construct the acquisition object if it's not already constructed
            if isinstance(gp_acquisition, str):
                if gp_acquisition not in ["LogExp", "NonlinearLogExp"]:
                    raise ValueError("Supported acquisition function is 'LogExp', "
                                     f"'NonlinearLogExp', got {gp_acquisition}")
                self.acquisition = GP_Acquisition(
                    prior_bounds, proposer=None, acq_func=gp_acquisition,
                    acq_optimizer="fmin_l_bfgs_b",
                    n_restarts_optimizer=5 * self.d, n_repeats_propose=10,
                    preprocessing_X=Normalize_bounds(prior_bounds),
                    zeta_scaling=options.get("zeta_scaling", 1.1), verbose=verbose)
            elif isinstance(gp_acquisition, GP_Acquisition):
                self.acquisition = gp_acquisition
            else:
                raise TypeError(
                    "gp_acquisition should be an Acquisition object or "
                    f"'LogExp', or 'NonlinearLogExp', got {gp_acquisition}")
            # Construct the convergence criterion
            if isinstance(convergence_criterion, str):
                try:
                    conv_class = getattr(gpryconv, convergence_criterion)
                except AttributeError:
                    raise ValueError(
                        f"Unknown convergence criterion {convergence_criterion}. "
                        f"Available convergence criteria: {gpryconv.builtin_names()}")
                self.convergence = conv_class(model.prior, convergence_options or {})
            elif isinstance(convergence_criterion, gpryconv.ConvergenceCriterion):
                self.convergence = convergence_criterion
            else:
                raise TypeError("convergence_criterion should be a "
                                "Convergence_criterion object or "
                                f"{gpryconv.builtin_names()}, got "
                                f"{convergence_criterion}")
            self.convergence_is_MPI_aware = self.convergence.is_MPI_aware
            # Read in options for the run
            if options is None:
                self.log(
                    "No options dict found. Defaulting to standard parameters.", level=3)
                options = {}
            self.n_initial = options.get("n_initial", 3 * self.d)
            self.max_points = options.get("max_points", 1000)
            self.max_accepted = options.get("max_accepted", self.max_points)
            self.max_init = options.get("max_init", 10 * self.d * self.n_initial)
            self.n_points_per_acq = options.get("n_points_per_acq", mpi_size)
            self.fit_full_every = options.get(
                "fit_full_every", max(int(2 * np.sqrt(self.d)), 1))
            if self.n_points_per_acq < mpi_size:
                self.log("Warning: parallellisation not fully utilised! It is advised to "
                         "make ``n_points_per_acq`` equal to the number of MPI processes "
                         "(default when not specified).", level=2)
            if self.n_points_per_acq > 2 * self.d:
                self.log("Warning: The number kriging believer samples per "
                         "acquisition step is larger than 2x number of dimensions of "
                         "the feature space. This may lead to slow convergence."
                         "Consider running it with less cores or decreasing "
                         "n_points_per_acq manually.", level=2)
            # Sanity checks
            if self.n_initial >= self.max_points:
                raise ValueError("The number of initial samples needs to be "
                                 "smaller than the maximum number of points")
            if self.n_initial <= 0:
                raise ValueError("The number of initial samples needs to be bigger "
                                 "than 0")
            if self.max_accepted > self.max_points:
                raise ValueError("You manually set max_accepted > max_points, but "
                                 " you cannot have more accepted than sampled points")
            # Callback
            self.callback = callback
            self.callback_is_MPI_aware = callback_is_MPI_aware
            self.callback_is_single_arg = (callable(callback) and
                                           len(getfullargspec(callback).args) == 1)
            # Print resume
            self.log("Initialized GPry.", level=3)
        if multiple_processes:
            for attr in ("n_initial", "max_init", "max_points", "max_accepted",
                         "n_points_per_acq", "gpr", "acquisition",
                         "convergence_is_MPI_aware", "callback_is_MPI_aware",
                         "callback_is_single_arg", "loaded_from_checkpoint"):
                setattr(self, attr, mpi_comm.bcast(getattr(self, attr, None)))
            # Only broadcast non-MPI-aware objects if necessary, to save trouble+memory
            if self.convergence_is_MPI_aware or self.callback_is_MPI_aware:
                self.convergence = mpi_comm.bcast(
                    convergence if is_main_process else None)
            if self.callback_is_MPI_aware:
                self.callback = mpi_comm.bcast(
                    callback if is_main_process else None)
            else:  # for check of whether to call it
                callback_func = callback
                self.callback = mpi_comm.bcast(
                    (callback is not None) if is_main_process else None)
                if is_main_process:
                    self.callback = callback_func
        # Prepare progress summary table; the table key is the iteration number
        if not self.loaded_from_checkpoint:
            self.progress = Progress()
        self.has_run = False
        self.has_converged = False

    @property
    def d(self):
        """Dimensionality of the problem."""
        return self.model.prior.d()

    def log(self, msg, level=None):
        """
        Print a message if its verbosity level is equal or lower than the given one (or
        always if ``level=None``.
        """
        if level is None or level <= self.verbose:
            print(msg)

    def read_checkpoint(self):
        """
        Loads checkpoint files to be able to resume a run or save the results for
        further processing.
        """
        self.model, self.gpr, self.acquisition, self.convergence, self.options, \
            self.progress = read_checkpoint(self.checkpoint)

    def save_checkpoint(self):
        """
        Saves checkpoint files to be able to resume a run or save the results for
        further processing.
        """
        if is_main_process:
            save_checkpoint(self.checkpoint, self.model, self.gpr, self.acquisition,
                            self.convergence, self.options, self.progress)

    def run(self):
        r"""
        Runs the acquisition-training-convergence loop until either convergence or
        a stopping condition is reached.
        """
        if self.has_run:
            self.log("The GP fitting has already run. Doing nothing.")
            return
        if not self.loaded_from_checkpoint:
            # Define initial training set
            self.log("Starting by drawing initial samples.", level=3)
            self.do_initial_training()
            if is_main_process:
                # Save checkpoint
                self.save_checkpoint()
                self.log("Initial samples drawn, starting with Bayesian "
                         "optimization loop.", level=3)
        if multiple_processes:
            n_finite = mpi_comm.bcast(len(self.gpr.y_train) if is_main_process else None)
        else:
            n_finite = len(self.gpr.y_train)
        # Run bayesian optimization loop
        n_iterations = int((self.max_points - n_finite) / self.n_points_per_acq)
        n_evals_per_acq_per_process = \
            split_number_for_parallel_processes(self.n_points_per_acq)
        n_evals_this_process = n_evals_per_acq_per_process[mpi_rank]
        i_evals_this_process = sum(n_evals_per_acq_per_process[:mpi_rank])
        it = 0
        n_left = self.max_accepted - n_finite
        for it in range(n_iterations):
            self.current_iteration = it
            self.progress.add_iteration()
            self.progress.add_current_n_truth(self.gpr.n_total, self.gpr.n)
            if is_main_process:
                if self.max_accepted != self.max_points:
                    self.log(f"+++ Iteration {it} "
                             f"(Accepting at most {n_left} more points) +++++++++",
                             level=3)
                else:
                    self.log(f"+++ Iteration {it} "
                             f"(of at most {n_iterations} iterations) +++++++++",
                             level=3)
                # Save old gp for convergence criterion
                old_gpr = deepcopy(self.gpr)
            self.gpr = mpi_comm.bcast(self.gpr if is_main_process else None)
            # Acquire new points in parallel
            with TimerCounter(self.gpr) as timer_acq:
                new_X, y_pred, acq_vals = self.acquisition.multi_add(
                    self.gpr, n_points=self.n_points_per_acq, random_state=self.rng)
            self.progress.add_acquisition(timer_acq.time, timer_acq.evals)
            if is_main_process:
                self.log(f"New X {new_X} ; y_lie {y_pred} ; acq {acq_vals}", level=3)
            # Get logposterior value(s) for the acquired points (in parallel)
            if multiple_processes:
                new_X = mpi_comm.bcast(new_X if is_main_process else None)
            new_X_this_process = new_X[
                i_evals_this_process: i_evals_this_process + n_evals_this_process]
            new_y_this_process = np.empty(0)
            with Timer() as timer_truth:
                for x in new_X_this_process:
                    new_y_this_process = np.append(
                        new_y_this_process, self.model.logpost(x))
            self.progress.add_truth(timer_truth.time, len(new_X))
            # Collect (if parallel) and append to the current model
            if multiple_processes:
                # Send together X's and y's, in order to avoid race-cond changes in order
                new_Xy_pairs = mpi_comm.gather((new_X_this_process, new_y_this_process))
                if is_main_process:
                    # Transpose+concatenate the pairs
                    new_X, new_y = list(
                        list(chain(*X_or_y)) for X_or_y in chain(zip(*new_Xy_pairs)))
                new_X, new_y = mpi_comm.bcast((np.array(new_X), np.array(new_y))
                                              if is_main_process else (None, None))
            else:
                new_y = new_y_this_process
            if is_main_process:
                do_simplified_fit = (it % self.fit_full_every != self.fit_full_every - 1)
                with TimerCounter(self.gpr) as timer_fit:
                    self.gpr.append_to_data(new_X, new_y,
                                            fit=True, simplified_fit=do_simplified_fit)
                self.progress.add_fit(timer_fit.time, timer_fit.evals_loglike)
            if multiple_processes:
                self.gpr = mpi_comm.bcast(self.gpr)
            n_left = self.max_accepted - self.gpr.n
            if multiple_processes:
                self.gpr, old_gpr, new_X, new_y, y_pred = mpi_comm.bcast(
                    (self.gpr, old_gpr, new_X, new_y, y_pred)
                    if is_main_process else None)
            # TODO: better failsafes for MPI_aware=False BUT actually using MPI
            # Use a with statement to pass an MPI communicator (dummy if MPI_aware=False)
            if self.callback:
                if self.callback_is_MPI_aware or is_main_process:
                    # TODO: unify order of arguments with read/save_checkpoint.
                    #       maybe even pass a runner object?
                    if self.callback_is_single_arg:
                        args = [self]
                    else:
                        args = [self.model, self.gpr, self.acquisition,
                                self.convergence, self.options, self.progress,
                                old_gpr, new_X, new_y, y_pred]
                    self.callback(*args)
                mpi_comm.barrier()
            # Calculate convergence and break if the run has converged
            if not self.convergence_is_MPI_aware:
                if is_main_process:
                    try:
                        with TimerCounter(self.gpr, old_gpr) as timer_convergence:
                            is_converged = self.convergence.is_converged(
                                self.gpr, old_gpr, new_X, new_y, y_pred)
                        self.progress.add_convergence(
                            timer_convergence.time, timer_convergence.evals,
                            self.convergence.last_value)
                    except gpryconv.ConvergenceCheckError:
                        self.progress.add_convergence(
                            timer_convergence.time, timer_convergence.evals,
                            np.nan)
                        is_converged = False
                if multiple_processes:
                    is_converged = mpi_comm.bcast(
                        is_converged if is_main_process else None)
            else:  # run by all processes
                # NB: this assumes that when the criterion fails,
                #     ALL processes raise ConvergenceCheckerror, not just rank 0
                try:
                    with TimerCounter(self.gpr, old_gpr) as timer_convergence:
                        is_converged = self.convergence.is_converged(
                            self.gpr, old_gpr, new_X, new_y, y_pred)
                    self.progress.add_convergence(
                        timer_convergence.time, timer_convergence.evals,
                        self.convergence.last_value)
                except gpryconv.ConvergenceCheckError:
                    self.progress.add_convergence(
                        timer_convergence.time, timer_convergence.evals,
                        np.nan)
                    is_converged = False
            sync_processes()
            self.progress.mpi_sync()
            if is_main_process:
                self.log(f"run - tot: {self.gpr.n_total}, "
                         f"acc: {self.gpr.n}, "
                         f"con: {self.convergence.values[-1]}, "
                         f"lim: {self.convergence.thres[-1]}", level=3)
            if is_converged:
                self.has_converged = True
                break
            # If the loop reaches n_left <= 0, then all processes need to break,
            # not just the main process
            n_left = mpi_comm.bcast(n_left if is_main_process else None)
            if n_left <= 0:
                break
            self.save_checkpoint()
            if is_main_process and self.plots:
                self.plot_progress()
        # Save
        self.save_checkpoint()
        if is_main_process and self.plots:
            self.plot_progress()
        if n_left <= 0 and is_main_process \
            and not isinstance(self.convergence, gpryconv.DontConverge) and self.verbose > 1:
            warnings.warn("The maximum number of accepted points was reached before "
                          "convergence. Either increase max_accepted or try to "
                          "choose a smaller prior.")
        if it == n_iterations and is_main_process and self.verbose > 1:
            warnings.warn("Not enough points were accepted before "
                          "reaching convergence/reaching the specified max_points.")
        # Now that the run has converged we can return the gp and all other
        # relevant quantities which can then be processed with an MCMC or other
        # sampler
        if multiple_processes:
            self.gpr, self.acquisition, self.convergence, self.progress, self.options = \
                mpi_comm.bcast(
                    (self.gpr, self.acquisition, self.convergence, self.progress,
                     self.options)
                    if is_main_process else None)
        self.has_run = True

    def do_initial_training(self, max_init=None):
        """
        Draws an initial sample for the `gpr` GP model until it has a training set of size
        `n_initial`, counting only finite-target points ("finite" here meaning over the
        threshold of the SVM classifier, if present).

        Parameters
        ----------
        This function is MPI-aware.

        max_init : int, optional (default=None)
            Will fail is it needed to evaluate the target more than `max_init`
            times (defaults to 10 times the dimension of the problem times the
            number of initial samples requested).
        """
        max_init = max_init or 10 * self.d * self.n_initial
        if self.progress:
            self.progress.add_iteration()
            self.progress.add_current_n_truth(0, 0)
            self.progress.add_acquisition(0, 0)
            self.progress.add_convergence(0, 0, np.nan)
        # Check if there's an SVM and if so read out it's threshold value
        # We will compare it against y - max(y)
        if isinstance(self.gpr.account_for_inf, SVM):
            # Grab the threshold from the internal SVM (the non-preprocessed one)
            self.gpr.account_for_inf.update_threshold(dimension=self.d)
            is_finite = lambda y: self.gpr.account_for_inf.is_finite(
                y, y_is_preprocessed=False)
        else:
            is_finite = lambda y: np.isfinite(y)
        if is_main_process:
            # Check if the GP already contains points. If so they are reused.
            pretrained = 0
            if hasattr(self.gpr, "y_train"):
                if len(self.gpr.y_train) > 0:
                    pretrained = len(self.gpr.y_train)
            n_still_needed = self.n_initial - pretrained
            n_to_sample_per_process = int(np.ceil(n_still_needed / mpi_size))
            # Arrays to store the initial sample
            X_init = np.empty((0, self.d))
            y_init = np.empty(0)
        if multiple_processes:
            n_to_sample_per_process = mpi_comm.bcast(
                n_to_sample_per_process if is_main_process else None)
        if n_to_sample_per_process == 0 and self.verbose > 1:  # Enough pre-training
            warnings.warn("The number of pretrained points exceeds the number of "
                          "initial samples")
            return
        n_iterations_before_giving_up = int(np.ceil(max_init / n_to_sample_per_process))
        # Initial samples loop. The initial samples are drawn from the prior
        # and according to the distribution of the prior.
        with Timer() as timer_truth:
            for i in range(n_iterations_before_giving_up):
                X_init_loop = np.empty((0, self.d))
                y_init_loop = np.empty(0)
                for j in range(n_to_sample_per_process):
                    # Draw point from prior and evaluate logposterior at that point
                    X = self.model.prior.reference(warn_if_no_ref=False)
                    self.log(f"Evaluating true posterior at {X}", level=3)
                    y = self.model.logpost(X)
                    self.log(f"Got {y}", level=3)
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
        if self.progress and is_main_process:
            self.progress.add_truth(timer_truth.time, len(X_init))
        self.log("Done getting initial training samples.", level=3)
        if is_main_process:
            # Append the initial samples to the gpr
            with TimerCounter(self.gpr) as timer_fit:
                self.gpr.append_to_data(X_init, y_init)
            if self.progress:
                self.progress.add_fit(timer_fit.time, timer_fit.evals_loglike)
            # Raise error if the number of initial samples hasn't been reached
            if not finished:
                raise RuntimeError("The desired number of finite initial "
                                   "samples hasn't been reached. Try "
                                   "increasing max_init or decreasing the "
                                   "volume of the prior")
        if self.progress:
            self.progress.mpi_sync()

    def plot_progress(self):
        """
        Creates some progress plots and saves them at path (assumes path exists).
        """
        self.progress.plot_timing(
            truth=False, save=os.path.join(self.plots_path, "timing.svg"))
        self.progress.plot_evals(save=os.path.join(self.plots_path, "evals.svg"))
        fig, ax = plot_convergence(self.convergence)
        fig.savefig(os.path.join(self.plots_path, "convergence.svg"))
        import matplotlib.pyplot as plt
        plt.close(fig)

    def generate_mc_sample(self, sampler="mcmc", output=None, add_options=None,
                           resume=False):
        """
        Runs an MC process using `Cobaya <https://cobaya.readthedocs.io/en/latest/sampler.html>`_.

        Parameters
        ----------
        sampler : string (default `"mcmc"`) or dict
            Sampler to be initialised. If a string, it must be `"mcmc"` or `"polychord"`.
            It can also be a dict as ``{sampler: {option: value, ...}}``, containing a
            full sampler definition, see `here
            <https://cobaya.readthedocs.io/en/latest/sampler.html>`_. In this case, any
            sampler understood by Cobaya can be used.

        add_options : dict, optional
            Dict of additional options to be passed to the sampler.

        output: path, optional (default: ``checkpoint/chains``, if ``checkpoint != None``)
            The path where the resulting Monte Carlo sample shall be stored. If passed
            explicitly ``False``, produces no output.

        resume: bool, optional (default=False)
            Whether to resume from existing output files (True) or force overwrite (False)

        Returns
        -------
        surr_info : dict
            The dictionary that was used to run (or initialized) the sampler,
            corresponding to the surrogate model, and populated with the sampler input
            specification.

        sampler : Sampler instance
            The sampler instance that has been run (or just initialised). The sampler
            products can be retrieved with the `Sampler.products()` method.
        """
        if not self.gpr.fitted:
            raise Exception("You have to have added points to the GPR "
                            "before you can generate an mc_sample")
        if output is None and self.checkpoint is not None:
            output = os.path.join(self.checkpoint, "chains/")
        return mc_sample_from_gp(self.gpr, true_model=self.model, sampler=sampler,
                                 convergence=self.convergence, output=output,
                                 add_options=add_options, resume=resume)

    def plot_mc(self, surr_info, sampler, add_training=True, add_samples=None):
        """
        Creates some progress plots and saves them at path (assumes path exists).

        .. warning::
            This method requires GetDist to be installed. It is neither a requirement
            for GPry nor Cobaya so you might have to install it manually if you want to
            use it (highly encouraged).

        Parameters
        ----------
        surr_info, sampler : dict, Cobaya.sampler
            Return values of method :func:`generate_mc_sample`

        add_training : bool, optional (default=True)
            Whether the training locations are plotted on top of the contours.

        add_samples : dict(label, getdist.MCSamples), optional (default=None)
            Whether the training locations are plotted on top of the contours.
        """
        if is_main_process:
            from getdist.mcsamples import MCSamplesFromCobaya
            import getdist.plots as gdplt
            from gpry.plots import getdist_add_training
            import matplotlib.pyplot as plt
            gdsamples_gp = MCSamplesFromCobaya(surr_info, sampler.products()["sample"])
            gdplot = gdplt.get_subplot_plotter(width_inch=5)
            to_plot = [gdsamples_gp]
            if add_samples:
                to_plot += list(add_samples.values())
            gdplot.triangle_plot(
                to_plot, self.model.parameterization.sampled_params(), filled=True)
            if add_training and self.d > 1:
                getdist_add_training(gdplot, self.model, self.gpr)
            plt.savefig(os.path.join(self.plots_path, "Surrogate_triangle.png"), dpi=300)


def run(model, gpr="RBF", gp_acquisition="LogExp",
        convergence_criterion="CorrectCounter",
        callback=None, callback_is_MPI_aware=False,
        convergence_options=None, options={}, checkpoint=None,
        load_checkpoint=None, verbose=3):
    r"""
    This function is just a wrapper which internally creates a runner instance and runs
    the bayesian optimization loop. This function will probably be depreciated in a few
    versions.

    Parameters
    ----------
    model : Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_
        Contains all information about the parameters in the likelihood and
        their priors as well as the likelihood itself. Cobaya is only used here
        as a wrapper to get the logposterior etc.

    gpr : GaussianProcessRegressor, "RBF" or "Matern", optional (default="RBF")
        The GP used for interpolating the posterior. If None or "RBF" is given
        a GP with a constant kernel multiplied with an anisotropic RBF kernel
        and dynamic bounds is generated. The same kernel with a Matern 3/2
        kernel instead of a RBF is generated if "Matern" is passed. This might
        be useful if the posterior is not very smooth.
        Otherwise a custom GP regressor can be created and passed.

    gp_acquisition : GP_Acquisition, optional (default="LogExp")
        The acquisition object. If None is given the LogExp acquisition
        function is used (with the :math:`\zeta` value chosen automatically
        depending on the dimensionality of the prior) and the GP's X-values are
        preprocessed to be in the uniform hypercube before optimizing the
        acquistion function.

    convergence_criterion : Convergence_criterion, optional (default="KL")
        The convergence criterion. If None is given the KL-divergence between
        consecutive runs is calculated with an MCMC run and the run converges
        if KL<0.02 for two consecutive steps.

    convergence_options: optional parameters passed to the convergence criterion.

    options : dict, optional (default=None)
        A dict containing all options regarding the bayesian optimization loop.
        The available options are:

            * n_initial : Number of initial samples before starting the BO loop
              (default: 3*number of dimensions)
            * n_points_per_acq : Number of points which are aquired with
              Kriging believer for every acquisition step (default: equals the
              number of parallel processes)
            * max_points : Maximum number of attempted sampling points before the run
              fails. This is useful if you e.g. want to restrict the maximum computation
              resources (default: 1000).
            * max_accepted : Maximum number of accepted sampling points before the run
              fails. This might be useful if you use the DontConverge convergence
              criterion, specifying exactly how many points you want to have in your GP
              (default: max_points)
            * max_init : Maximum number of points drawn at initialization
              before the run fails (default: 10 * number of dimensions * n_initial).
              If the run fails repeatadly at initialization try decreasing the volume
              of your prior.

    callback: callable, optional (default=None)
        Function run each iteration after adapting the recently acquired points and
        the computation of the convergence criterion. This function should take arguments
        ``callback(model, current_gpr, gp_acquistion, convergence_criterion, options,
                   progress, previous_gpr, new_X, new_y, pred_y)``.
        When running in parallel, the function is run by the main process only.

    callback_is_MPI_aware: bool (default: False)
        If True, the callback function is called for every process simultaneously, and
        it is expected to handle parallelisation internally. If false, only the main
        process calls it.

    checkpoint : str, optional (default=None)
        Path for storing checkpointing information from which to resume in case the
        algorithm crashes. If None is given no checkpoint is saved.

    load_checkpoint: "resume" or "overwrite", must be specified if path is not None.
        Whether to resume from the checkpoint files if existing ones are found
        at the location specified by ´checkpoint´.

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

    gpr : GaussianProcessRegressor
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

    progress : Progress
        Object containing per-iteration progress information: number of accepted training
        points, number of GP evaluations, timing of different parts of the algorithm, and
        value of the convergence criterion.
    """
    runner = Runner(model, gpr=gpr, gp_acquisition=gp_acquisition,
                    convergence_criterion=convergence_criterion, callback=callback,
                    callback_is_MPI_aware=callback_is_MPI_aware,
                    convergence_options=convergence_options, options=options,
                    checkpoint=checkpoint, load_checkpoint=load_checkpoint,
                    verbose=verbose)
    runner.run()
    return (runner.model, runner.gpr, runner.acquisition, runner.convergence,
            runner.options, runner.progress)
