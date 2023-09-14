"""
Interface for Cobaya: wrapper using the Cobaya.sampler.Sampler class.

To use with Cobaya, once GPry has been installed with pip, add it to the sampler block
as ``gpry.CobayaSampler`.

For input arguments and options, see ``CobayaSampler.yaml`` in this folder, or run
``cobaya-doc gpry.CobayaSampler`` in a shell.
"""

import os
import re
import logging
from tempfile import gettempdir
from inspect import cleandoc

from cobaya.sampler import Sampler
from cobaya.log import LoggedError

from gpry.run import Runner


# pylint: disable=no-member,access-member-before-definition
# pylint: disable=attribute-defined-outside-init
class CobayaSampler(Sampler):
    """GPry: a package for Bayesian inference of expensive likelihoods using GPs."""

    # Other options
    _gpry_output_dir = "gpry_output"
    _surrogate_suffix = "gpr"

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
        self.output_strategy = "resume" if self.output.is_resuming() else "overwrite"
        # Initialize the runner
        try:
            self.gpry_runner = Runner(
                model=self.model,
                gpr=self.gpr,
                gp_acquisition=self.gp_acquisition,
                initial_proposer=self.initial_proposer,
                convergence_criterion=self.convergence_criterion,
                options=self.options,
                callback=self.callback,
                callback_is_MPI_aware=self.callback_is_MPI_aware,
                checkpoint=self.path_checkpoint,
                load_checkpoint=self.output_strategy,
                seed=self._rng,
                plots=self.plots,
                verbose=self.verbose,
            )
        except (ValueError, TypeError) as excpt:
            raise LoggedError(
                self.log,
                f"Error when initializing GPry: {str(excpt)}"
            ) from excpt

    def run(self):
        """
        Gets the initial training points and starts the acquistion loop.
        """
        self.gpry_runner.run()
        self.do_surrogate_sample(resume=self.output.is_resuming())

    def do_surrogate_sample(self, resume=False):
        """
        Perform an MC sample of the surrogate model. Can be called by hand if the initial
        one did not converge.
        """
        if self.output:
            output_path = os.path.realpath(
                os.path.join(self.path_checkpoint, "..", self.surrogate_prefix)
            )
        else:
            output_path = os.path.realpath(
                os.path.join(self.path_checkpoint, self.surrogate_prefix)
            )
        self.mc_sample = self.gpry_runner.generate_mc_sample(
            sampler=self.mc_sampler, output=output_path, resume=resume
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

    @property
    def surrogate_prefix(self):
        """
        Prefix for the MC sample of the surrogate model.
        """
        return self.output.prefix + ("_" if self.output else "") + self._surrogate_suffix

    @classmethod
    def get_base_dir(cls, output):
        if output:
            return output.add_suffix(cls._gpry_output_dir, separator="_")
        return os.path.join(gettempdir(), cls._gpry_output_dir)

    @classmethod
    def output_files_regexps(cls, output, info=None, minimal=False):
        """
        Returns a list of tuples `(regexp, root)` of output files potentially produced.
        If `root` in the tuple is `None`, `output.folder` is used.

        If `minimal=True`, returns regexp's for the files that should really not be there
        when we are not resuming.
        """
        # GPry checkpoint files
        regexps_tuples = [
            (re.compile(re.escape(name + ".pkl")), cls.get_base_dir(output))
            for name in ["acq", "con", "gpr", "mod", "opt", "pro"]
        ]
        if minimal:
            return regexps_tuples
        return regexps_tuples + [
            # Raw products base dir
            (None, cls.get_base_dir(output)),
            # Main sample
            (output.collection_regexp(name=None), None),
        ]

    @classmethod
    def get_desc(cls, info=None):
        return ("GPry: a package for Bayesian inference of expensive likelihoods "
                r"with Gaussian Processes \cite{Gammal:2022eob}.")

    @classmethod
    def get_bibtex(cls):
        return cleandoc(r"""
            @article{Gammal:2022eob,
                author = {{El Gammal}, Jonas and Sch\"oneberg, Nils and Torrado, Jes\'us and Fidler, Christian},
                title = "{Fast and robust Bayesian Inference using Gaussian Processes with GPry}",
                eprint = "2211.02045",
                archivePrefix = "arXiv",
                primaryClass = "astro-ph.CO",
                month = "11",
                year = "2022"
            }""")
