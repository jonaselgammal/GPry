"""
Interface for Cobaya: wrapper using the Cobaya.sampler.Sampler class.
"""

import os
import logging
from tempfile import gettempdir

from cobaya.sampler import Sampler

from gpry.run import Runner

class GPrySampler(Sampler):
    # The GP used for interpolating the posterior.
    gpr = "RBF"  # or Matern, or GaussianProcessRegressor instance
    # TODO: option for kernel args

    # Acquisition class. Can pass an instance too
    gp_acquisition = "LogExp"
    # TODO: acquisition object args (e.g. zeta)

    # Proposer used for drawing the initial training samples before running the aquisition
    # loop. Can pass instance too.
    initial_proposer = "reference"  # or prior, or uniform (within prior bounds)
    # TODO: initial proposer args

    # The convergence criterion. Can pass instance too
    convergence_criterion = "CorrectCounter"

    # Convergence criterion args
    convergence_options = None
    # TODO: maybe rename to _args

    # Options regarding the bayesian optimization loop.
    options = {
        # Number of inite initial truth evaluations before starting the acq loop.
        ## "n_initial": None,  # default: 3 times the dimensionality.
        # Maximum number of truth evaluations at initialization. If it
        # is reached before `n_initial` finite points have been found, the run will fail.
        ##"max_initial": None,  # default: 10 times the dimensionality times n_initial
        # Number of points aquired with Kriging believer at every acquisition step
        ##"n_points_per_acq": None,  # default: number of parallel MPI processes
        # Maximum number of attempted sampling points before the run fails.
        ##"max_total": None,  # default: 70 times the dimentionality ^1.5
        # Maximum number of points accepted into the GP training set before the run fails.
        ##"max_finite": None,  # default: value of max_total
        # Scaling of the :math:`\zeta` parameter in the exponential acquisition function
        # with the number of dimensions :math:`\zeta=1/d^-scaling`
        ##"zeta_scaling": None,  # default: 0.85
    }

    # Cobaya sampler used to generate the final sample from the surrogate model
    mc_sampler = "mcmc"  # default: mcmc with Cobaya defaults

    # Produce progress plots (inside the gpry_output dir).
    # Adds overhead for very fast likelihoods.
    plots = False

    # Function run each iteration after adapting the recently acquired points and
    # the computation of the convergence criterion. See docs for implementation.
    callback = None

    # Whether the callback function handles MPI-parallelization internally.
    # Otherwise run only by the rank-0 process
    callback_is_MPI_aware = None

    # Change to increase or reduce verbosity. If None, it is handled by Cobaya.
    # '3' produces general progress output (default for Cobaya if None),
    # and '4' debug-level output
    verbose = None

    # Other options
    _gpry_output_dir = "gpry_output"

    def initialize(self):
        """
        Initializes GPry.
        """
        # Set some args for the Runner that are derived from Cobaya
        if self.verbose is None:
            if self.log.getEffectiveLevel() == logging.NOTSET:
                self.verbose = 3
            elif self.log.getEffectiveLevel() <= logging.DEBUG:
                self.verbose = 4
            elif self.log.getEffectiveLevel() <= logging.INFO:
                self.verbose = 3
            else:
                self.verbose = 2
        # Prepare output
        self.path_checkpoint = self.get_base_dir(self.output)
        self.mc_sample = None
        # Initialize the runner
        self.gpry_runner = Runner(
            model=self.model,
            gpr=self.gpr,
            gp_acquisition=self.gp_acquisition,
            convergence_criterion=self.convergence_criterion,
            callback=self.callback,
            callback_is_MPI_aware=self.callback_is_MPI_aware,
            convergence_options=self.convergence_options,
            options=self.options,
            initial_proposer=self.initial_proposer,
            checkpoint=self.path_checkpoint,
            # TODO: handle resuming/overwriting properly
            load_checkpoint="overwrite",
            seed=self._rng,
            plots=self.plots,
            verbose=self.verbose,
        )

    def run(self):
        """
        Gets the initial training points and starts the acquistion loop.
        """
        self.gpry_runner.run()
        if self.output:
            # TODO: do not overwrite updated.yaml if writing to root folder of sample!
            output_path = os.path.realpath(
                os.path.join(self.path_checkpoint, "..", self.output.prefix)
            )
        else:
            output_path = os.path.realpath(
                os.path.join(self.path_checkpoint, self.output.prefix)
            )
        # TODO: option to re-do final sample
        self.mc_sample = self.gpry_runner.generate_mc_sample(
            sampler=self.mc_sampler, output=output_path, resume=False
        )

    def products(
        self,
        combined: bool = False,
        skip_samples: float = 0,
        to_getdist: bool = False,
    ) -> dict:
        """
        Returns the products of the run: an MC sample of the surrogate posterior under
        ``sample``, and the GPRy ``Runner`` object under ``runner``.
        """
        # TODO: MPI interactions -- look at cobaya.mcmc
        return {
            "runner": self.gpry_runner,
            "sample": self.mc_sample,
        }

    @classmethod
    def get_base_dir(cls, output):
        if output:
            return output.add_suffix(cls._gpry_output_dir, separator="_")
        return os.path.join(gettempdir(), cls._gpry_output_dir)