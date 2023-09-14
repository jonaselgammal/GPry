import os
import warnings
from copy import deepcopy
from inspect import getfullargspec
from typing import Mapping
import numpy as np

from cobaya.model import Model

from gpry.mpi import mpi_comm, mpi_size, mpi_rank, is_main_process, get_random_state, \
    split_number_for_parallel_processes, multiple_processes, sync_processes, share_attr
from gpry.proposal import InitialPointProposer, ReferenceProposer, PriorProposer, \
    UniformProposer
from gpry.gpr import GaussianProcessRegressor
from gpry.gp_acquisition import GenericGPAcquisition
import gpry.gp_acquisition as gprygpacqs
import gpry.acquisition_functions as gpryacqfuncs
from gpry.svm import SVM
from gpry.preprocessing import Normalize_bounds, Normalize_y
import gpry.convergence as gpryconv
from gpry.progress import Progress, Timer, TimerCounter
from gpry.io import create_path, check_checkpoint, read_checkpoint, save_checkpoint
from gpry.mc import mc_sample_from_gp
from gpry.plots import plot_convergence, plot_distance_distribution
from gpry.tools import create_cobaya_model


_plots_path = "images"


class Runner():
    r"""
    Class that takes care of constructing the Bayesian quadrature/likelihood
    characterization loop. After initialisation, the algorithm can be launched with
    :func:`Runner.run`, and, optionally after that, an MC process can be launched on the
    surrogate model with :func:`Runner.generate_mc_sample`.

    Parameters
    ----------
    model : callable or Cobaya `model object <https://cobaya.readthedocs.io/en/latest/cosmo_model.html>`_
        Likelihood function (returning log-likelihood; requires additional argument
        ``bounds``) or Cobaya Model instance (which contains all information about the
        parameters in the likelihood and their priors as well as the likelihood itself).
        It must not be specified if 'resuming' from a checkpoint (see ``load_checkpoint``
        below).

    bounds: List of [min, max], or Dict {name: [min, max],...}
        List or dictionary of parameter bounds. If it is a dictionary, the keys need to
        correspond to the argument names of the ``likelihood`` function, and the values
        can be either bounds specified as ``[min, max]``, or bounds and labels, as
        ``{"prior": [min, max], "latex": [label]}``. It does not need to be defined (will
        be ignored) if a Cobaya ``Model`` instance is passed as ``model``.

    gpr : GaussianProcessRegressor, str, dict, optional (default="RBF")
        The GP used for interpolating the posterior. If None or "RBF" is given
        a GP with a constant kernel multiplied with an anisotropic RBF kernel
        and dynamic bounds is generated. The same kernel with a Matern 3/2
        kernel instead of a RBF is generated if "Matern" is passed. This might
        be useful if the posterior is not very smooth. Otherwise a custom GP regressor can
        be defined as a dict containing the arguments of ``GaussianProcessRegressor``, or
        passing an already initialized instance.

    gp_acquisition : GenericGPAcquisition, optional (default="LogExp")
        The acquisition object. If None is given the BatchOptimizer with a LogExp
        acquisition function is used (with the :math:`\zeta` value chosen automatically
        depending on the dimensionality of the prior) and the GP's X-values are
        preprocessed to be in the uniform hypercube before optimizing the
        acquistion function. It can also be passed an initialized instance, or a dict with
        arguments with which to initialize one.

    initial_proposer : InitialPointProposer, str, dict, optional (default="reference")
        Proposer used for drawing the initial training samples before running the
        Bayesian optimisation loop. As standard the samples are drawn from the model
        reference (prior if no reference is specified). Alternative options which can be
        passed as strings are ``"prior", "uniform"``. The ``"reference"`` proposer
        defaults to the prior if no reference distribution is provided. If defined as a
        dict with the proposer name as single key, the values will be passed as kwargs to
        the proposer.

    convergence_criterion : ConvergenceCriterion, str, dict, optional (default="CorrectCounter")
        The convergence criterion. If None is given the Correct counter convergence
        criterion is used with adaptive relative and absoluter thresholds. Can be
        specified as a dict to initialize a ConvergenceCriterion class with some
        arguments, or directly as an instance of ConvergenceCriterion.

    options : dict, optional (default=None)
        A dict containing all options regarding the bayesian optimization loop.
        The available options are:

            * n_initial : Number of finite initial truth evaluations before starting the
              BO loop (default: 3*number of dimensions)
            * max_initial : Maximum number of truth evaluations at initialization. If it
              is reached before `n_initial` finite points have been found, the run will
              fail. To avoid that, try decreasing the volume of your prior
              (default: 10 * number of dimensions * n_initial).
            * n_points_per_acq : Number of points which are aquired with
              Kriging believer for every acquisition step (default: equals the
              number of parallel processes)
            * max_total : Maximum number of attempted sampling points before the run
              fails. This is useful if you e.g. want to restrict the maximum computation
              resources (default: 70 * (number of dimensions)**1.5)).
            * max_finite : Maximum number of sampling points accepted into the GP training
              set before the run fails. This might be useful if you use the DontConverge
              convergence criterion, specifying exactly how many points you want to have
              in your GP. If you set this limit by hand and find that it is easily
              saturated, try decreasing the volume of your prior (default: max_total).

    callback : callable, optional (default=None)
        Function run each iteration after adapting the recently acquired points and
        the computation of the convergence criterion. This function should take the
        runner as argument: ``callback(runner_instance)``.
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

    seed: int, optional
        Seed for the random number generator. Allows for reproducible runs.

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

    gp_acquisition : GenericGPAcquisition
        The acquisition object that was used for the active sampling procedure.

    convergence_criterion : Convergence_criterion
        The convergence criterion used for determining convergence. Depending
        on the criterion used this also contains the approximate covariance
        matrix of the posterior distribution which can be used by the MCMC
        sampler.

    options : dict
        The options dict used for the active sampling loop.

    progress : Progress
        Object containing per-iteration progress information: number of finite training
        points, number of GP evaluations, timing of different parts of the algorithm, and
        value of the convergence criterion.
    """

    def __init__(self,
                 model=None,
                 bounds=None,
                 gpr="RBF",
                 gp_acquisition="LogExp",
                 initial_proposer="reference",
                 convergence_criterion="CorrectCounter",
                 callback=None,
                 callback_is_MPI_aware=False,
                 options=None,
                 checkpoint=None,
                 load_checkpoint=None,
                 seed=None,
                 plots=False,
                 verbose=3,
                 # DEPRECATED ON 13-09-2023:
                 convergence_options=None,
                 ):
        self.verbose = verbose
        if model is None:
            if not (checkpoint is not None and str(load_checkpoint).lower() == "resume"):
                raise ValueError(
                    "'model' must be specified unless resuming from a checkpoint.")
        elif isinstance(model, Model):
            self.model = model
        elif callable(model):
            if bounds is None:
                raise ValueError("'bounds' need to be defined if a likelihood "
                                 "function is passed.")
            self.model = create_cobaya_model(model, bounds)
        self.checkpoint = checkpoint
        if self.checkpoint is not None:
            self.plots_path = os.path.join(self.checkpoint, _plots_path)
            if is_main_process:
                create_path(self.checkpoint, verbose=self.verbose >= 3)
                if plots:
                    create_path(self.plots_path, verbose=self.verbose >= 3)
        else:
            self.plots_path = _plots_path
            if plots and is_main_process:
                create_path(self.plots_path, verbose=self.verbose >= 3)
        self.plots = plots
        self.ensure_paths(plots=self.plots)
        self.random_state = get_random_state(seed)
        if is_main_process:
            self.options = options or {}
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
                        self.read_checkpoint(model=model)
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
            if not isinstance(model, Model) and not callable(model):
                if load_checkpoint == "resume":
                    raise ValueError(f"Resuming from checkpoint {checkpoint} failed. "
                                     "In this case, a 'model' needs to be specified.")
                else:
                    raise TypeError("'model' needs to be a likelihood function or a "
                                    f"Cobaya model. got {model!r}")
            try:
                self.prior_bounds = self.model.prior.bounds(
                    confidence_for_unbounded=0.99995)
            except Exception as excpt:
                raise RuntimeError("There seems to be something wrong with "
                                   f"the model instance: {excpt}") from excpt
            # Construct the main loop elements:
            # GPR, GPAcquisition, InitialProposer and ConvergenceCriterion
            # DEPRECATED ON 13-09-2023:
            if convergence_options is not None and isinstance(convergence_criterion, str):
                self.log(
                    "*Warning*: 'convergence_options' has been deprecated. You can "
                    "now speficy arguments for the convergence criterion passing it "
                    "as a dict, e.g. 'convergence_criterion={\"CorrectCounter\": "
                    "{\"kwarg\": value}'. Your arguments have been passed, but this "
                    "will fail in the future."
                )
                convergence_criterion = {convergence_criterion: convergence_options}
            zeta_scaling = (options or {}).pop("zeta_scaling", None)
            if zeta_scaling is not None:
                self.log(
                    "*Warning*: Passing 'zeta_scaling' as part of the 'options' has been "
                    "deprecated. It should be passed as a key of the 'acq_func' dict "
                    "the GPAcquisition specification, e.g. '{\"BatchOptimizer\": "
                    "{\"acq_func\": {\"zeta_scaling\": 0.85}}}'. The given 'zeta_scaling'"
                    " is being used, but this will fail in the future."
                )
                if isinstance(gp_acquisition, str):
                    gp_acquisition = {gp_acquisition: {"zeta_scaling": zeta_scaling}}
            # END OF DEPRECATION BLOCK
            self._construct_gpr(gpr)
            self._construct_gp_acquisition(gp_acquisition)
            self._construct_initial_proposer(initial_proposer)
            self._construct_convergence_criterion(convergence_criterion)




            # Read in options for the run
            if options is None:
                self.log(
                    "No options dict found. Defaulting to standard parameters.", level=3)
                options = {}
            self.n_initial = options.get("n_initial", 3 * self.d)
            self.max_initial = options.get("max_initial", 10 * self.d * self.n_initial)
            self.max_total = options.get("max_total", int(70 * self.d**1.5))
            self.max_finite = options.get("max_finite", self.max_total)
            self.n_points_per_acq = options.get("n_points_per_acq", min(mpi_size, self.d))
            self.fit_full_every = options.get(
                "fit_full_every", max(int(2 * np.sqrt(self.d)), 1))
            if self.n_points_per_acq > self.d:
                self.log("Warning: The number kriging believer samples per "
                         "acquisition step is larger than the number of dimensions of "
                         "the feature space. This may lead to slow convergence."
                         "Consider running it with less cores or decreasing "
                         "n_points_per_acq manually.", level=2)
            elif (self.n_points_per_acq < mpi_size and self.n_points_per_acq < mpi_size and
                  self.n_points_per_acq < self.d):
                self.log("Warning: parallellisation not fully utilised! It is advised to "
                         "make ``n_points_per_acq`` equal to the number of MPI processes "
                         "(default when not specified).", level=2)
            # Sanity checks
            if self.n_initial <= 0:
                raise ValueError("The number of initial samples needs to be bigger "
                                 "than 0")
            for attr in ["n_initial", "max_initial", "max_finite",
                         "max_total", "n_points_per_acq"]:
                if getattr(self, attr) < 0 or \
                   getattr(self, attr) != int(getattr(self, attr)):
                    raise ValueError(f"'{attr}' must be a positive integer.")
            if self.n_initial >= self.max_finite:
                raise ValueError("The number of initial samples needs to be smaller than "
                                 "the maximum number of finite and total points.")
            if self.max_finite > self.max_total:
                raise ValueError("The maximum number of initial truth evaluations needs "
                                 "to be smaller than the maximum total number of "
                                 "evaluations.")
            # Diagnosis
            self.diagnosis = options.get("diagnosis", None)
            # Callback
            self.callback = callback
            self.callback_is_MPI_aware = callback_is_MPI_aware
            # Print resume
            self.log("Initialized GPry.", level=3)
        if multiple_processes:
            for attr in ("n_initial", "max_initial", "max_total", "max_finite",
                         "n_points_per_acq", "options", "acquisition",
                         "convergence_is_MPI_aware", "callback_is_MPI_aware",
                         "loaded_from_checkpoint", "initial_proposer", "progress",
                         "diagnosis"):
                share_attr(self, attr)
            self._share_gpr_from_main()
            # Only broadcast non-MPI-aware objects if necessary, to save trouble+memory
            if self.convergence_is_MPI_aware:
                share_attr(self, "convergence")
            elif not is_main_process:
                self.convergence = None
            if self.callback_is_MPI_aware:
                share_attr(self, "callback")
            else:  # for check of whether to call it
                callback_func = callback
                self.callback = mpi_comm.bcast(
                    (callback is not None) if is_main_process else None)
                if is_main_process:
                    self.callback = callback_func
        # Prepare progress summary table; the table key is the iteration number
        if not self.loaded_from_checkpoint:
            self.progress = Progress()
        self.current_iteration = 0
        self.has_run = False
        self.has_converged = False
        self.old_gpr, self.new_X, self.new_y, self.y_pred = None, None, None, None
        self.mean, self.cov = None, None

    def _construct_gpr(self, gpr):
        """Constructs or passes the GPR."""
        if isinstance(gpr, GaussianProcessRegressor):
            self.gpr = gpr
        elif isinstance(gpr, (Mapping, str)):
            if isinstance(gpr, str):
                gpr = {"kernel": gpr}
            gpr_defaults = {
                "kernel": "RBF",
                "n_restarts_optimizer": 10 + 2 * self.d,
                "preprocessing_X": Normalize_bounds(self.prior_bounds),
                "preprocessing_y": Normalize_y(),
                "bounds": self.prior_bounds,
                "random_state": self.random_state,
                "verbose": self.verbose,
            }
            for k, value in gpr_defaults.items():
                if gpr.get(k) is None:
                    gpr[k] = value
            try:
                self.gpr = GaussianProcessRegressor(**gpr)
            except ValueError as excpt:
                raise ValueError(
                    f"Error when initializing the GP regressor: {str(excpt)}"
                ) from excpt
        else:
            raise TypeError(
                "'gpr' should be a GP regressor, a dict of arguments for the GPR, "
                "or a string specifying the kernel ('RBF' or 'Matern'). Got {gpr}"
            )

    def _construct_gp_acquisition(self, gp_acquisition):
        """Constructs or passes the GPAcquisition instance."""
        default_gq_acquisition = "BatchOptimizer"
        if isinstance(gp_acquisition, GenericGPAcquisition):
            self.acquisition = gp_acquisition
        elif isinstance(gp_acquisition, (Mapping, str)) or gp_acquisition is None:
            if gp_acquisition is None:
                gp_acquisition = {default_gq_acquisition: {}}
            elif isinstance(gp_acquisition, str):
                gp_acquisition = {gp_acquisition: {}}
            # If an acq_func name was passed, use the standard batch-optimization one
            if list(gp_acquisition)[0] in gpryacqfuncs.builtin_names():
                gp_acquisition = {
                    default_gq_acquisition: {"acq_func": {list(gp_acquisition)[0]: {}}}}
            gp_acquisition_name = list(gp_acquisition)[0]
            gp_acquisition_args = gp_acquisition[gp_acquisition_name] or {}
            gp_acquisition_defaults = {
                "bounds": self.prior_bounds,
                "preprocessing_X": Normalize_bounds(self.prior_bounds),
                "random_state": self.random_state,
                "acq_func": {"LogExp": {"zeta_scaling": 0.85}},
                "verbose": self.verbose,
            }
            for k, value in gp_acquisition_defaults.items():
                if gp_acquisition_args.get(k) is None:
                    gp_acquisition_args[k] = value
            try:
                gp_acquisition_class = getattr(gprygpacqs, gp_acquisition_name)
            except AttributeError as excpt:
                raise ValueError(
                    f"Unknown GPAcquisiton class {gp_acquisition_name}. "
                    f"Available GPAcquisition classes: {gprygpacqs.builtin_names()}"
                ) from excpt
            try:
                self.acquisition = gp_acquisition_class(**gp_acquisition_args)
            except Exception as excpt:
                raise ValueError(
                    "Error when initialising the GPAcquisition object "
                    f"{gp_acquisition_name} with arguments {gp_acquisition_args}: "
                    f"{str(excpt)}"
                ) from excpt
        else:
            raise TypeError(
                "'gp_acquisition' should be a GPAcquisition object, "
                "or a dict or string specification for one of "
                f"{gprygpacqs.builtin_names()}. Got {gp_acquisition}"
            )

    def _construct_initial_proposer(self, initial_proposer):
        """Constructs or passes the initial proposer."""
        if isinstance(initial_proposer, InitialPointProposer):
            self.intial_proposer = initial_proposer
        elif isinstance(initial_proposer, (Mapping, str)):
            if isinstance(initial_proposer, str):
                initial_proposer = {initial_proposer: {}}
            initial_proposer_name = list(initial_proposer)[0]
            initial_proposer_args = initial_proposer[initial_proposer_name]
            if initial_proposer_name.lower() == "reference":
                self.initial_proposer = ReferenceProposer(
                    self.model, **initial_proposer_args)
            elif initial_proposer_name.lower() == "prior":
                self.initial_proposer = PriorProposer(
                    self.model, **initial_proposer_args)
            elif initial_proposer_name.lower() == "uniform":
                self.initial_proposer = UniformProposer(
                    self.prior_bounds, **initial_proposer_args)
            else:
                raise ValueError(
                    "Supported standard initial point proposers are "
                    f"'reference', 'prior', 'uniform'. Got {initial_proposer}")
        else:
            raise TypeError(
                "'initial_proposer' should be an InitialPointProposer instance, a "
                "dict specification, or one of 'reference', 'prior' or 'uniform'. "
                f" Got {initial_proposer}"
            )

    def _construct_convergence_criterion(self, convergence_criterion):
        """Constructs or passes the convergence criterion."""
        if isinstance(convergence_criterion, gpryconv.ConvergenceCriterion):
            self.convergence = convergence_criterion
        elif isinstance(convergence_criterion, (Mapping, str)):
            if isinstance(convergence_criterion, str):
                convergence_criterion = {convergence_criterion: {}}
            convergence_name = list(convergence_criterion)[0]
            convergence_args = convergence_criterion[convergence_name] or {}
            try:
                convergence_class = getattr(gpryconv, convergence_name)
            except AttributeError as excpt:
                raise ValueError(
                    f"Unknown convergence criterion {convergence_name}. "
                    f"Available convergence criteria: {gpryconv.builtin_names()}"
                ) from excpt
            try:
                self.convergence = convergence_class(
                    self.model.prior, convergence_args)
            except Exception as excpt:
                raise ValueError(
                    "Error when initialising the convergence criterion "
                    f"{convergence_name} with arguments {convergence_args}: "
                    f"{str(excpt)}"
                ) from excpt
        else:
            raise TypeError(
                "'convergence_criterion' should be a ConvergenceCriterion instance, "
                "or a dict or string specification for one of "
                f"{gpryconv.builtin_names()}. Got {convergence_criterion}"
            )
        # This attr allows *not* to have to share the convergence criterion
        self.convergence_is_MPI_aware = self.convergence.is_MPI_aware

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

    def ensure_paths(self, plots=True):
        """
        Creates paths for checkpoint and plots.
        """
        if is_main_process:
            if self.checkpoint:
                create_path(self.checkpoint, verbose=self.verbose >= 3)
            if plots:
                create_path(self.plots_path, verbose=self.verbose >= 3)

    @property
    def n_total_left(self):
        """Number of truth evaluations before stopping."""
        return self.max_total - self.gpr.n_total

    @property
    def n_finite_left(self):
        """Number of truth evaluations with finite return value before stopping."""
        return self.max_finite - self.gpr.n_finite

    def banner(self, text, max_line_length=79, prefix="| ", suffix=" |",
               header="=", footer="=", level=3):
        """Creates an iteration banner."""
        default_header_footer = "="
        if header:
            if not isinstance(header, str):
                header = default_header_footer
            self.log(max_line_length * str(header), level=level)
        text = text.strip("\n")
        lines = text.split("\n")
        for line in lines:
            line = prefix + line
            left_before_suffix = max_line_length - len(line) - len(suffix)
            if left_before_suffix >= 0:
                line += " " * left_before_suffix + suffix
            self.log(line, level=level)
        if footer:
            if not isinstance(footer, str):
                footer = default_header_footer
            self.log(max_line_length * str(footer), level=level)

    def read_checkpoint(self, model=None):
        """
        Loads checkpoint files to be able to resume a run or save the results for
        further processing.

        Parameters
        ----------
        model : cobaya.model.Model, optional
            If passed, it will be used instead of the loaded one.
        """
        self.model, self.gpr, self.acquisition, self.convergence, self.options, \
            self.progress = read_checkpoint(self.checkpoint, model=model)

    def save_checkpoint(self):
        """
        Saves checkpoint files to be able to resume a run or save the results for
        further processing.
        """
        if is_main_process:
            save_checkpoint(self.checkpoint, self.model, self.gpr, self.acquisition,
                            self.convergence, self.options, self.progress)

    def _share_gpr_from_main(self):
        """
        Shares the GPR of the main process, restoring each process' RNG.
        """
        if not multiple_processes:
            return
        share_attr(self, "gpr")
        self.gpr.set_random_state(self.random_state)

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
            if is_main_process:
                self.banner("Drawing initial samples.")
            self.do_initial_training()
            if is_main_process:
                # Save checkpoint
                self.save_checkpoint()
        # Run bayesian optimization loop
        n_evals_per_acq_per_process = \
            split_number_for_parallel_processes(self.n_points_per_acq)
        n_evals_this_process = n_evals_per_acq_per_process[mpi_rank]
        i_evals_this_process = sum(n_evals_per_acq_per_process[:mpi_rank])
        self.has_converged = False
        if is_main_process:
            maybe_stop_before_max_total = (
                (self.max_finite < self.max_total) or
                not isinstance(self.convergence, gpryconv.DontConverge))
            at_most_str = "at most " if maybe_stop_before_max_total else ""
        while (self.n_total_left > 0 and self.n_finite_left > 0 and
               not self.has_converged):
            self.current_iteration += 1
            self.progress.add_iteration()
            if is_main_process:
                n_iter_left = int(np.ceil(self.n_total_left / self.n_points_per_acq))
                self.banner(f"Iteration {self.current_iteration} "
                            f"({at_most_str}{n_iter_left} left)\n"
                            f"Total truth evals: {self.gpr.n_total} "
                            f"({self.gpr.n_finite} finite) of {self.max_total}" +
                            (f" (or {self.max_finite} finite)"
                             if self.max_finite < self.max_total else ""))
            self.old_gpr = deepcopy(self.gpr)
            self.progress.add_current_n_truth(self.gpr.n_total, self.gpr.n_finite)
            # Acquire new points in parallel
            sync_processes()  # to sync the timer
            with TimerCounter(self.gpr) as timer_acq:
                new_X, y_pred, acq_vals = self.acquisition.multi_add(
                    self.gpr, n_points=self.n_points_per_acq, random_state=self.random_state)
            self.progress.add_acquisition(timer_acq.time, timer_acq.evals)
            if is_main_process:
                self.log(f"[ACQUISITION] ({timer_acq.time:.2g} sec) Proposed {len(new_X)}"
                         " point(s) for truth evaluation.", level=3)
                self.log("New location(s) proposed, as [X, logp_gp(X), acq(X)]:", level=4)
                for X, y, acq in zip(new_X, y_pred, acq_vals):
                    self.log(f"   {X} {y} {acq}", level=4)
            # Get logposterior value(s) for the acquired points (in parallel)
            new_X_this_process = new_X[
                i_evals_this_process: i_evals_this_process + n_evals_this_process]
            new_y_this_process = np.empty(0)
            sync_processes()  # to sync the timer
            with Timer() as timer_truth:
                for x in new_X_this_process:
                    self.log(f"[{mpi_rank}] Evaluating true posterior at {x}", level=4)
                    new_y_this_process = np.append(
                        new_y_this_process, self.model.logpost(x))
                    self.log(f"[{mpi_rank}] Got true log-posterior {new_y_this_process} "
                             f"at {x}", level=4)
            self.progress.add_truth(timer_truth.time, len(new_X))
            # Collect (if parallel) and append to the current model
            if multiple_processes:
                # GATHER keeps rank order (MPI standard): we can do X and y separately
                new_Xs = mpi_comm.gather(new_X_this_process)
                new_ys = mpi_comm.gather(new_y_this_process)
                if is_main_process:
                    new_X = np.concatenate(new_Xs)
                    new_y = np.concatenate(new_ys)
                new_X, new_y = mpi_comm.bcast(
                    (new_X, new_y) if is_main_process else (None, None))
            else:
                new_y = new_y_this_process
            if is_main_process:
                self.log(f"[EVALUATION] ({timer_truth.time:.2g} sec) Evaluated the true "
                         f"model at {len(new_X)} location(s)" +
                         (f" (at most {len(new_X_this_process)} per MPI process)"
                          if multiple_processes else "") +
                         f", of which {sum(np.isfinite(new_y))} returned a finite value.",
                         level=3)
            sync_processes()
            # Add the newly evaluated truths to the GPR, and maye refit hyperparameters
            if is_main_process:
                kwargs_append = {}
                kwargs_append["simplified_fit"] = \
                    (self.current_iteration % self.fit_full_every !=
                     self.fit_full_every - 1)
                if self.cov is not None:
                    stds = np.sqrt(np.diag(self.cov))
                    prior_bounds = self.model.prior.bounds(confidence_for_unbounded=0.99995)
                    relative_stds = stds / (prior_bounds[:, 1] - prior_bounds[:, 0])
                    new_bounds = np.array([relative_stds / 2,  relative_stds * 2]).T
                    kwargs_append["hyperparameter_bounds"] = self.gpr.kernel_.bounds.copy()
                    kwargs_append["hyperparameter_bounds"][1:] = np.log(new_bounds)
                with TimerCounter(self.gpr) as timer_fit:
                    self.gpr.append_to_data(new_X, new_y,
                                            fit=True, **kwargs_append)
                self.progress.add_fit(timer_fit.time, timer_fit.evals_loglike)
            if is_main_process:
                hyperparams_or_not = "*not* " if kwargs_append["simplified_fit"] else ""
                self.log(f"[FIT] ({timer_fit.time:.2g} sec) Fitted GP model with new "
                         "acquired points, "
                         f"{hyperparams_or_not}including GPR hyperparameters. "
                         f"{(self.gpr.n_last_appended_finite if sum(np.isfinite(new_y)) else 0)} finite points were added to "
                         "the GPR.", level=3)
                self.log(f"Current GPR kernel: {self.gpr.kernel_}", level=2)
            self._share_gpr_from_main()
            sync_processes()
            # share new_X, new_y and y_pred to the runner instance
            self.new_X, self.new_y, self.y_pred = mpi_comm.bcast(
                (new_X, new_y, y_pred) if is_main_process else (None, None, None))
            # We *could* check the max_total/finite condition and stop now, but it is
            # good to run the convergence criterion anyway, in case it has converged
            # Run the `callback` function
            # TODO: better failsafes for MPI_aware=False BUT actually using MPI
            # Use a with statement to pass an MPI communicator (dummy if MPI_aware=False)
            if self.callback:
                if self.callback_is_MPI_aware or is_main_process:
                    with Timer() as timer_callback:
                        self.callback(self)
                    if is_main_process:
                        self.log(f"[CALLBACK] ({timer_callback.time:.2g} sec) Evaluated "
                                 "the callback function.", level=3)
                sync_processes()
            # Calculate convergence and break if the run has converged
            if not self.convergence_is_MPI_aware:
                if is_main_process:
                    try:
                        with TimerCounter(self.gpr, self.old_gpr) as timer_convergence:
                            self.has_converged = self.convergence.is_converged(
                                self.gpr, self.old_gpr, new_X, new_y, y_pred)
                        self.progress.add_convergence(
                            timer_convergence.time, timer_convergence.evals,
                            self.convergence.last_value)
                    except gpryconv.ConvergenceCheckError:
                        self.progress.add_convergence(
                            timer_convergence.time, timer_convergence.evals,
                            np.nan)
                        self.has_converged = False
            else:  # run by all processes
                # NB: this assumes that when the criterion fails,
                #     ALL processes raise ConvergenceCheckerror, not just rank 0
                try:
                    with TimerCounter(self.gpr, self.old_gpr) as timer_convergence:
                        self.has_converged = self.convergence.is_converged(
                            self.gpr, self.old_gpr, new_X, new_y, y_pred)
                    self.progress.add_convergence(
                        timer_convergence.time, timer_convergence.evals,
                        self.convergence.last_value)
                except gpryconv.ConvergenceCheckError:
                    self.progress.add_convergence(
                        timer_convergence.time, timer_convergence.evals,
                        np.nan)
                    self.has_converged = False
            share_attr(self, "has_converged")
            if is_main_process:
                self.log(f"[CONVERGENCE] ({timer_convergence.time:.2g} sec) "
                         "Evaluated convergence criterion to "
                         f"{self.convergence.last_value:.2g} (limit "
                         f"{self.convergence.thres[-1]:.2g}).", level=3)
            sync_processes()
            # TODO: uncomment for mean and cov updates (cov would be used for corr.length)
            # self.update_mean_cov()
            self.progress.mpi_sync()
            self.save_checkpoint()
            if is_main_process and self.plots:
                self.plot_progress()
        else:  # check "while" ending condition
            sync_processes()
            if is_main_process:
                lines = "Finished!\n"
                if self.has_converged:
                    lines += "- The run has converged.\n"
                if self.n_total_left <= 0:
                    lines += ("- The maximum number of truth evaluations "
                              f"({self.max_total}) has been reached.\n")
                if self.max_finite < self.max_total and self.n_finite_left <= 0:
                    lines += ("- The maximum number of finite truth evaluations "
                              f"({self.max_finite}) has been reached.")
                self.banner(lines)
            if self.diagnosis:
                self.diagnose()
        self.has_run = True

    def do_initial_training(self):
        """
        Draws an initial sample for the `gpr` GP model until it has a training set of size
        `n_initial`, counting only finite-target points ("finite" here meaning over the
        threshold of the SVM classifier, if present).

        This function is MPI-aware and broadcasts the initialized GPR to all processes.
        """
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
        n_iterations_before_giving_up = int(
            np.ceil(self.max_initial / n_to_sample_per_process))
        # Initial samples loop. The initial samples are drawn from the prior
        # and according to the distribution of the prior.
        sync_processes()  # to sync the timer
        with Timer() as timer_truth:
            for i in range(n_iterations_before_giving_up):
                X_init_loop = np.empty((0, self.d))
                y_init_loop = np.empty(0)
                for j in range(n_to_sample_per_process):
                    # Draw point from prior and evaluate logposterior at that point
                    X = self.initial_proposer.get(random_state=self.random_state)
                    self.log(f"[{mpi_rank}] Evaluating true posterior at {X}", level=4)
                    y = self.model.logpost(X)
                    self.log(f"[{mpi_rank}] Got true log-posterior {y} at {X}", level=4)
                    X_init_loop = np.append(X_init_loop, np.atleast_2d(X), axis=0)
                    y_init_loop = np.append(y_init_loop, y)
                # Gather points and decide whether to break.
                if multiple_processes:
                    # GATHER keeps rank order (MPI standard): we can do X and y separately
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
                    finished = n_finite_new >= n_still_needed
                if multiple_processes:
                    finished = mpi_comm.bcast(finished if is_main_process else None)
                if finished:
                    break
                else:
                    # TODO: maybe re-fit SVM to shrink initial sample region
                    pass
        if self.progress and is_main_process:
            self.progress.add_truth(timer_truth.time, len(X_init))
        if is_main_process:
            self.log(f"[EVALUATION] ({timer_truth.time:.2g} sec) "
                     f"Evaluated the true model at {len(X_init)} location(s)"
                     f", of which {sum(is_finite(y_init - max(y_init)))} returned a "
                     f"finite value." +
                     (" Each MPI process evaluated at most "
                      f"{max(len(p) for p in all_points)} locations."
                      if multiple_processes else ""), level=3)
        if is_main_process:
            # Raise error if the number of initial samples hasn't been reached
            if not finished:
                raise RuntimeError("The desired number of finite initial "
                                   "samples hasn't been reached. Try "
                                   "increasing max_initial or decreasing the "
                                   "volume of the prior")
            # Append the initial samples to the gpr
            with TimerCounter(self.gpr) as timer_fit:
                self.gpr.append_to_data(X_init, y_init)
            self.progress.add_fit(timer_fit.time, timer_fit.evals_loglike)
            self.log(f"[FIT] ({timer_fit.time:.2g} sec) Fitted GP model with new acquired"
                     " points, including GPR hyperparameters. "
                     f"{self.gpr.n_last_appended_finite} finite points were added to the "
                     "GPR.", level=3)
            self.log(f"Current GPR kernel: {self.gpr.kernel_}", level=4)
        # Broadcast results
        self._share_gpr_from_main()
        self.progress.mpi_sync()

    def update_mean_cov(self):
        """
        Updates and shares mean and cov if available, checking GPAcquisition first, and
        Convergence second if not present in GPAcquisition.
        """
        for attr in ["mean", "cov"]:
            if is_main_process:
                value = getattr(self.acquisition, attr, None)
                if value is None:
                    value = getattr(self.convergence, attr, None)
                setattr(self, attr, value)
            share_attr(self, attr)

    def plot_progress(self):
        """
        Creates some progress plots and saves them at path (assumes path exists).
        """
        if not is_main_process:
            return
        self.ensure_paths(plots=True)
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        self.progress.plot_timing(
            truth=False, save=os.path.join(self.plots_path, "timing.svg"))
        self.progress.plot_evals(save=os.path.join(self.plots_path, "evals.svg"))
        fig, ax = plot_convergence(self.convergence)
        fig.savefig(os.path.join(self.plots_path, "convergence.svg"))
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
            output = os.path.join(self.checkpoint, "chains/mc_samples")
        return mc_sample_from_gp(self.gpr, true_model=self.model, sampler=sampler,
                                 convergence=self.convergence, output=output,
                                 add_options=add_options, resume=resume,
                                 verbose=self.verbose)

    def diagnose(self):
        if is_main_process:
            lines = "Starting diagnosis\n"
            lines += "- Evaluating corners"
            self.log(lines)
            bounds = self.model.prior.bounds()
            ndim = len(bounds)
            mesh = np.meshgrid(*bounds)
            corners = np.stack(mesh, axis=-1).reshape(-1, ndim)
            # Evaluate GP at all corners
            vals_in_corners = self.gpr.predict(corners, validate=False)
            # Check if at any point it's overshooting
            higher_than_max = vals_in_corners > self.gpr.y_max
            if np.sum(higher_than_max) > 0:
                lines = f"WARNING: found {np.sum(higher_than_max)} corners\n"
                lines += "where the GP predicts a higher value than its\n"
                lines += "maximum. Reevaluating those corners..."
                self.log(lines)
                # Filter the points where the high values are predicted and
                # evaluate the posterior distribution there
                points_to_evaluate = np.atleast_2d(corners[higher_than_max])
                new_vals = np.empty(len(points_to_evaluate))
                for i, p in enumerate(points_to_evaluate):
                    new_vals[i] = self.model.logpost(p)
                    self.gpr.append_to_data(points_to_evaluate, new_vals,
                            fit=True)
                self._share_gpr_from_main()
                # self.save_checkpoint()
                self.log("...done.")

    # pylint: disable=import-outside-toplevel
    def plot_mc(self, surr_info_or_sample_folder, sampler=None, add_training=True,
                add_samples=None, output=None):
        """
        Creates a triangle plot of an MC sample of the surrogate model, and optionally
        shows some evaluation locations.

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
            Extra getdist.MCSamples objects to be added to the plot.

        output : str or os.path, optional (default=None)
            The location to save the generated plot in. If ``None`` it will be saved in
            ``checkpoint_path/images/Surrogate_triangle.pdf`` or
            ``./images/Surrogate_triangle.png`` if ``checkpoint_path`` is ``None``
        """
        if not is_main_process:
            return None
        self.ensure_paths(plots=True)
        if isinstance(surr_info_or_sample_folder, str):
            root = os.path.abspath(surr_info_or_sample_folder)
            if os.path.isdir(root):
                root += "/"  # to force GetDist to treat it as folder, not prefix
            from getdist.mcsamples import loadMCSamples
            gdsamples_gp = loadMCSamples(root)
        else:  # passed surr_info, sampler
            gdsamples_gp = sampler.products(
                to_getdist=True, combined=True, skip_samples=0.33)["sample"]
        import getdist.plots as gdplt
        from gpry.plots import getdist_add_training
        import matplotlib.pyplot as plt
        gdplot = gdplt.get_subplot_plotter(width_inch=5)
        to_plot = [gdsamples_gp]
        if add_samples:
            to_plot += list(add_samples.values())
        gdplot.triangle_plot(
            to_plot, self.model.parameterization.sampled_params(), filled=True)
        if add_training and self.d > 1:
            getdist_add_training(gdplot, self.model, self.gpr)
        if output is None:
            plt.savefig(os.path.join(self.plots_path, "Surrogate_triangle.png"),
                        dpi=300)
        else:
            plt.savefig(output)
        return gdplot

    def plot_distance_distribution(
            self, surr_info, sampler, show_added=True, output=None):
        """
        Creates a triangle plot of an MC sample of the surrogate model, and optionally
        shows some evaluation locations.

        Parameters
        ----------
        surr_info, sampler : dict, Cobaya.sampler
            Return values of method :func:`generate_mc_sample`
        show_added: bool (default True)
            Colours the stacks depending on how early or late the corresponding points
            were added (bluer stacks represent newer points).
        output : str or os.path, optional (default=None)
            The location to save the generated plot in. If ``None`` it will be saved in
            ``.png`` format at ``checkpoint_path/images/``, or ``./images/`` if
            ``checkpoint_path`` was ``None``.
        """
        if not is_main_process:
            return
        self.ensure_paths(plots=True)
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        mean = sampler.products()["sample"].mean()
        covmat = sampler.products()["sample"].cov()
        fig, ax = plot_distance_distribution(
            self.gpr, mean, covmat, density=False, show_added=show_added)
        if output is None:
            plt.savefig(os.path.join(self.plots_path, "Distance_distribution.png"),
                        dpi=300)
        else:
            plt.savefig(output)
        fig, ax = plot_distance_distribution(
            self.gpr, mean, covmat, density=True, show_added=show_added)
        if output is None:
            plt.savefig(os.path.join(self.plots_path,
                                     "Distance_distribution_density.png"),
                        dpi=300)
        else:
            plt.savefig(output)
        plt.close(fig)
