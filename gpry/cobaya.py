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
from copy import deepcopy
from tempfile import gettempdir
from inspect import cleandoc
from typing import Union

from cobaya.sampler import Sampler
from cobaya.component import get_component_class
from cobaya.log import LoggedError
from cobaya.output import get_output, split_prefix, OutputReadOnly
from cobaya.tools import get_external_function
from cobaya.collection import SampleCollection

from gpry import mpi
from gpry.run import Runner

# TODO: resuming may not work in cases where internal gp_acq or gpr options aren't changed


# pylint: disable=no-member,access-member-before-definition
# pylint: disable=attribute-defined-outside-init
class CobayaWrapper(Sampler):
    """GPry: a package for Bayesian inference of expensive likelihoods using GPs."""

    # Resume:
    _at_resume_prefer_new = ["plots", "callback", "callback_is_MPI_aware", "verbose"]

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
        # Default to Runner & Acq defaults: (recusively) remove keys with None value:
        for k, v in list(self.gpr.items()):
            if v is None:
                self.gpr.pop(k)
        for k, v in list(self.gp_acquisition.items()):
            if v is None:
                self.gp_acquisition.pop(k)
        gq_acq_input = deepcopy(self.gp_acquisition)
        gp_acq_engine = gq_acq_input.pop("engine", "BatchOptimizer")
        # Grab the relevant acq options, merge them, and kick out the unused ones
        gp_acq_engine_options = None
        for k in list(gq_acq_input):
            if k.startswith("options_"):
                gp_acq_engine_options = gq_acq_input.pop(k)
                if k.lower().endswith(gp_acq_engine.lower()):
                    gq_acq_input.update(gp_acq_engine_options or {})
        gp_acq_input = {gp_acq_engine: gq_acq_input}
        # Initialize the runner
        try:
            self.gpry_runner = Runner(
                model=self.model,
                gpr=self.gpr,
                gp_acquisition=gp_acq_input,
                initial_proposer=self.initial_proposer,
                convergence_criterion=self.convergence_criterion,
                options=self.options,
                callback=get_external_function(self.callback) if self.callback else None,
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
        if mpi.is_main_process:
            self.log.info("Starting learning stage...")
        try:
            self.gpry_runner.run()
        except Exception as excpt:
            raise LoggedError(
                self.log,
                "GPry failed during learning: %s",
                str(excpt)
            ) from excpt
        if mpi.is_main_process:
            if self.gpry_runner.has_converged:
                self.log.info("Learning stage finished successfully!")
            else:
                self.log.info("Learning stage failed to converge! Will MC sample anyway.")
            self.log.info("Starting MC-sampling stage...")
        try:
            self.do_surrogate_sample(resume=self.output.is_resuming())
        except Exception as excpt:
            raise LoggedError(
                self.log,
                "GPry failed during MC sampling of the surrogate model: %s",
                str(excpt)
            ) from excpt
        if mpi.is_main_process:
            if self.gpry_runner.has_converged:
                self.log.info("MC-sampling finished successfully!")
            else:
                self.log.info("MC-sampling finished, but model *DID NOT CONVERGE*!")
            if self.plots:
                self.log.info("Doing some plots...")
                self.do_plots()
        return self.mc_sampler_upd_info, self.mc_sampler_instance

    def do_surrogate_sample(
            self,
            sampler=None,
            add_options=None,
            resume=False,
            prefix=None,
    ):
        """
        Perform an MC sample of the surrogate model.

        This function is called automatically at the end of a run, but it can be called by
        hand too e.g. if the initial one did not converge.

        Parameters
        ----------
        sampler: str
            An anternative sampler, if different from the one specified at initialisation
        add_options: dict
            Configuration to be passed to the sampler.
        resume: bool (default: False)
            Whether to try to resume a previous run
        prefix: str, optional
            An alternative path where to save the sample. If not given, the sample will
            use the default one with suffix ``(_)gpr``.

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
        if prefix is None:
            prefix = self.surrogate_prefix
        self.gpry_runner.generate_mc_sample(
            sampler=self.mc_sampler if sampler is None else sampler,
            add_option=add_options,
            output=prefix,
            resume=resume,
        )
        self.mc_sampler_upd_info = self.gpry_runner.last_mc_surr_info
        self.mc_sampler_instance = self.gpry_runner.last_mc_sampler

    @property
    def is_mc_sampled(self):
        """
        Returns True if the MC sampling of the surrogate process has run and converged.
        """
        return bool(getattr(self.gpry_runner, "_last_mc_samples", False))

    def do_plots(self, format="svg"):
        """
        Produces some results and diagnosis plots.
        """
        self.gpry_runner.plot_progress(format=format)
        self.gpry_runner.plot_distance_distribution(format=format)
        if self.is_mc_sampled:
            self.gpry_runner.plot_mc(format=format)

    def samples(
            self,
            combined: bool = False,
            skip_samples: float = 0,
            to_getdist: bool = False,
    ) -> Union[SampleCollection, "MCSamples"]:
        """
        Returns the last sample from the surrogate model.
        """
        return self.mc_sampler_instance.samples(
            combined=combined,
            skip_samples=skip_samples,
            to_getdist=to_getdist,
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
        products = {"runner": self.gpry_runner}
        products.update(
            self.mc_sampler_instance.products(
                combined=combined, skip_samples=skip_samples, to_getdist=to_getdist
            )
        )
        return products

    @classmethod
    def get_checkpoint_dir_and_surr_prefix(cls, output=None):
        """
        Folder where the checkpoint output of GPry is going to be saved, and prefix for
        the output object of the MC sample of the surrogate model, given a Cobaya
        ``OutputReadOnly`` instance.

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
        return None, None

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
        ] + [(None, os.path.join(path_checkpoint, "images"))]
        # MC sample from surrogate -- more precise if we know the sampler
        # Using OutputReadOnly and not Output here bc it can be  called just from rank 0,
        # and it's never used except as an aux object to get correct surrogate MC prefixes
        surr_mc_output = OutputReadOnly(prefix=surrogate_prefix)
        surr_mc_folder, _ = split_prefix(surrogate_prefix)
        surr_mc_sampler = (info or {}).get("mc_sampler")
        if surr_mc_sampler:
            sampler = get_component_class(surr_mc_sampler, kind="sampler")
            regexps_tuples += [
                (regexp[0], os.path.join(surr_mc_folder, regexp[1] or ""))
                for regexp in sampler.output_files_regexps(
                        output=surr_mc_output, minimal=minimal)
            ]
        else:
            regexps_tuples += \
                [(surr_mc_output.collection_regexp(name=None), surr_mc_folder)]
        return regexps_tuples

    @staticmethod
    def is_nora(info):
        """Returns True if NORA is being used."""
        acq_method = list((info or {}).get("gp_acquisition", {}) or {})
        return (
            len(acq_method) > 0 and isinstance(acq_method[0], str) and
            acq_method[0].lower() == "nora"
        )

    @classmethod
    def get_desc(cls, info=None):
        nora_string = (
            r"using the NORA parallelised acquisition approach \cite{Torrado:2023cbj}"
        )
        if info is None:
            # Unknown case (no info passed)
            nora_string = f" [(if gp_acquisition: NORA) {nora_string}]"
        else:
            nora_string = " " + nora_string if cls.is_nora(info) else ""
        return (
            "GPry: a package for Bayesian inference of expensive likelihoods "
            r"with Gaussian Processes \cite{Gammal:2022eob}" + nora_string + "."
        )

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

            # Cite only if using NORA as gp_acquisition (see desc.)
            @article{Torrado:2023cbj,
                author = {Torrado, Jes\'us and Sch\"oneberg, Nils and Gammal, Jonas El},
                title = "{Parallelized Acquisition for Active Learning using Monte Carlo Sampling}",
                eprint = "2305.19267",
                archivePrefix = "arXiv",
                primaryClass = "stat.ML",
                month = "5",
                year = "2023"
           }""")
