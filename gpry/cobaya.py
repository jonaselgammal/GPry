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
from cobaya.component import get_component_class
from cobaya.log import LoggedError
from cobaya.output import get_output

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
        self.path_checkpoint, self.surrogate_prefix = \
            self.get_checkpoint_dir_and_surr_prefix(self.output)
        self.mc_sampler_upd_info = None
        self.mc_sampler_instance = None
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
        self.log.info("Starting learning stage...")
        try:
            self.gpry_runner.run()
        except Exception as excpt:
            raise LoggedError(self.log, "GPry failed during learning: %s", str(excpt))
        self.log.info("Learning stage finished successfully!")
        self.log.info("Starting MC-sampling stage...")
        try:
            self.mc_sampler_upd_info, self.mc_sampler_instance = \
                self.do_surrogate_sample(resume=self.output.is_resuming())
        except Exception as excpt:
            raise LoggedError(
                self.log,
                "GPry failed during MC sampling of the surrogate model: %s",
                str(excpt)
            )
        self.log.info("MC-sampling finished successfully!")

    def do_surrogate_sample(self, resume=False, prefix=None):
        """
        Perform an MC sample of the surrogate model.

        This function is called automatically at the end of a run, but it can be called by
        hand too e.g. if the initial one did not converge.

        Parameters
        ----------
        resume: bool (default: False)
            Whether to try to resume a previous run
        prefix: str, optional
            An alternative path where to save the sample. If not given, the sample will
            use the default one with suffix ``(_)gpr``.

        Resume
        ------
        surr_info : dict
            The dictionary that was used to run (or initialized) the sampler,
            corresponding to the surrogate model, and populated with the sampler input
            specification.

        sampler : Sampler instance
            The sampler instance that has been run (or just initialised). The sampler
            products can be retrieved with the `Sampler.products()` method.
        """
        if prefix is None:
            prefix = self.surrogate_prefix
        return self.gpry_runner.generate_mc_sample(
            sampler=self.mc_sampler, output=prefix, resume=resume
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
        return {
            "sample": self.mc_sampler_instance.products(
                combined=combined,
                skip_samples=skip_samples,
                to_getdist=to_getdist,
            ),
            "runner": self.gpry_runner,
        }

    @classmethod
    def get_checkpoint_dir_and_surr_prefix(cls, output=None):
        """
        Folder where the checkpoint output of GPry is going to be saved, and prefix for
        the output object of the MC sample of the surrogate model.given a Cobaya
        ``Output`` instance.

        These two are wrapped into a single classmethod in order to use the same temp
        folder if called with dummy output.

        Parameters
        ----------
        output: cobaya.output.Output, cobaya.output.DummyOutput, optional
            Cobaya output instance. Can be a dummy one or None, in which case a temporary
            folder will be created.

        Returns
        -------
        checkpoint_dir: str
            Relative folder where the GPry checkpoint will be saved.
        surrogate_prefix: str
            Prefix for surrogate MC chains for a Cobaya ``Output`` with relative path.

        Examples
        --------
        Assuming that ``cls._gpry_output_dir = "gpry_output"`` and
        ``cls._surrogate_suffix = "gpr"``:

        >>> from cobaya.output import get_output
        >>> cls.get_checkpoint_dir_and_surr_prefix(get_output("folder/"))
        'folder/gpry_output', 'folder/gpr'
        >>> cls.get_checkpoint_dir_and_surr_prefix(get_output("folder/prefix"))
        'folder/prefix_gpry_output', 'folder/prefix_gpr'
        >>> cls.get_checkpoint_dir_and_surr_prefix(get_output())  # dummy output
        '[tmp_folder]/gpry_output', '[tmp_folder]/gpr'
        """
        if output:
            return (output.add_suffix(cls._gpry_output_dir, separator="_"),
                    output.add_suffix(cls._surrogate_suffix, separator="_"))
        tmpdir = gettempdir()
        return (os.path.join(tmpdir, cls._gpry_output_dir),
                os.path.join(tmpdir, cls._surrogate_suffix))

    @classmethod
    def output_files_regexps(cls, output, info=None, minimal=False):
        """
        Returns a list of tuples `(regexp, root)` of output files potentially produced.
        If `root` in the tuple is `None`, `output.folder` is used.

        If `minimal=True`, returns regexp's for the files that should really not be there
        when we are not resuming: GPry checkpoint products and the MC sample from the
        surrogate.
        """
        path_checkpoint, surrogate_prefix = \
            cls.get_checkpoint_dir_and_surr_prefix(output)
        # GPry checkpoint files
        regexps_tuples = [
            (re.compile(re.escape(name + ".pkl")), path_checkpoint)
            for name in ["acq", "con", "gpr", "mod", "opt", "pro"]
        ]
        # MC sample from surrogate -- more precise if we know the sampler
        surr_mc_output = get_output(prefix=surrogate_prefix)
        surr_mc_sampler = (info or {}).get("mc_sampler")
        if surr_mc_sampler:
            sampler = get_component_class(surr_mc_sampler, kind="sampler")
            regexps_tuples += [
                (regexp[0], os.path.join(surr_mc_output.folder, regexp[1] or ""))
                 for regexp in sampler.output_files_regexps(
                         output=surr_mc_output, minimal=minimal)
            ]
        else:
            regexps_tuples += \
                [(surr_mc_output.collection_regexp(name=None), surr_mc_output.folder)]
        return regexps_tuples

    @staticmethod
    def is_nora(info):
        """Returns True if NORA is being used."""
        acq_method = list(((info or {}).get("gp_acquisition", {}) or {}).keys())
        print("a", (acq_method and isinstance(acq_method[0], str) and
                    acq_method[0].lower() == "nora"), acq_method)
        return (len(acq_method) > 0 and isinstance(acq_method[0], str) and
                acq_method[0].lower() == "nora")

    @classmethod
    def get_desc(cls, info=None):
        return ("GPry: a package for Bayesian inference of expensive likelihoods "
                r"with Gaussian Processes \cite{Gammal:2022eob}" +
                (", using the NORA acquisition approach \cite{Torrado:2023cbj}."
                 if cls.is_nora(info) else "."))

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
            @article{Torrado:2023cbj,
                author = {Torrado, Jes\'us and Sch\"oneberg, Nils and Gammal, Jonas El},
                title = "{Parallelized Acquisition for Active Learning using Monte Carlo Sampling}",
                eprint = "2305.19267",
                archivePrefix = "arXiv",
                primaryClass = "stat.ML",
                month = "5",
                year = "2023"
           }""")
