"""
Wrappers for external nested samplers.
"""

import os
import sys
import glob
import shutil
import tempfile
from warnings import warn
from abc import ABC, abstractmethod

import numpy as np

from gpry import mpi
from gpry.tools import (
    NumpyErrorHandling,
    generic_params_names,
    remove_0_weight_samples,
    check_and_return_bounds,
    get_Xnumber,
)


# Helper for PolyChord, that needs a scalar-returning likelihood
ensure_scalar = lambda x: x[0] if hasattr(x, "__len__") else x


class NestedSamplerNotInstalledError(Exception):
    """
    Exception to be raised at initialization of any of the interfaces if the NS failed to
    be imported.
    """


class NSInterface(ABC):
    """
    Meta-class for Nested Sampler interfaces.
    """

    @abstractmethod
    def __init__(self, bounds, verbosity=None):
        pass

    @abstractmethod
    def set_verbosity(self, verbose):
        """Sets the verbosity of the sampler at run time."""

    @abstractmethod
    def set_prior(self, bounds):
        """Sets the prior used by the nested sampler."""

    @abstractmethod
    def set_precision(self, **kwargs):
        """Sets precision parameters for the nested sampler."""

    @abstractmethod
    def run(self, logp_func, param_names=None, out_dir=None, keep_all=False, seed=None):
        """
        Runs the nested sampler.

        param_names (optional, otherwise x_[i] will be used) should be a list of sampled
        parameter names, or a list of (name, label) tuples. Labels are interpreted as
        LaTeX but should not include '$' signs.
        """

    @abstractmethod
    def delete_output(self, out_dir=None):
        """
        Deletes the last sampler output.

        If ``out_dir`` specified, deletes the one stored there instead.
        """

    @staticmethod
    def process_out_dir(out_dir, default_prefix="ns_samples", random_if_undefined=True):
        """
        Given an output root ``out_dir`` as ``folder/`` or ``folder/prefix``,
        returns separately the folder path and the file name prefix.

        If ``random_if_undefined`` is True (default), it returns a random temp folder.
        Otherwise it raises ``ValueError``.
        """
        if out_dir is None:
            if random_if_undefined:
                return tempfile.TemporaryDirectory().name, default_prefix
            raise ValueError(
                "No output root passed. Use ``random_if_undefined=True`` to generate a "
                "random one."
            )
        base_dir, file_root = os.path.split(out_dir)
        # If no slash in there, interpret as folder (since kwarg is 'out_dir')
        if not base_dir:
            base_dir, file_root = file_root, ""
        base_dir = os.path.abspath(base_dir)
        if file_root == "":
            file_root = default_prefix
        return base_dir, file_root


class InterfacePolyChord(NSInterface):
    """
    Interface for the PolyChord nested sampler, by W. Handley, M. Hobson & A. Lasenby.

    See https://github.com/PolyChord/PolyChordLite
    """

    def __init__(self, bounds, verbosity=3):
        try:
            from pypolychord.settings import PolyChordSettings  # type: ignore
            from pypolychord import run_polychord  # type: ignore

            self.globals = {"run_polychord": run_polychord}
        except ModuleNotFoundError as excpt:
            raise NestedSamplerNotInstalledError(
                "External nested sampler Polychord cannot be imported. "
                "Check out installation instructions at "
                "https://github.com/PolyChord/PolyChordLite "
                "(or select an alternative nested sampler, e.g. UltraNest)."
            ) from excpt
        bounds = check_and_return_bounds(bounds)
        self.dim = len(bounds)
        self.polychord_settings = PolyChordSettings(nDims=self.dim, nDerived=0)
        # Don't write unnecessary files: takes lots of space and wastes time
        self.settings = {
            "read_resume": False,
            "write_resume": False,
            "write_live": False,
            "write_dead": True,
            "write_prior": False,
        }
        for setting, value in self.settings.items():
            setattr(self.polychord_settings, setting, value)
        self.set_verbosity(verbosity)
        self.set_prior(bounds)
        # Storage of last sample -- will only be defined for rank-0 MPI process
        self.X_all = None
        self.y_all = None
        self.X_MC = None
        self.y_MC = None
        self.w_MC = None
        self.last_polychord_result = None

    def set_verbosity(self, verbose):
        """Sets the verbosity of the sampler at run time."""
        self.polychord_settings.feedback = verbose - 3
        # 0: print header and result; not very useful: turn it to -1 if that's the case
        if self.polychord_settings.feedback == 0:
            self.polychord_settings.feedback = -1

    def set_prior(self, bounds):
        """Sets the prior used by the nested sampler."""
        from pypolychord.priors import UniformPrior  # type: ignore

        self.prior = UniformPrior(*(bounds.T))

    def set_precision(self, **kwargs):
        """Sets precision parameters for the nested sampler."""
        known = ["nlive", "num_repeats", "precision_criterion", "nprior", "max_ncalls"]
        for p in known:
            val = kwargs.pop(p, None)
            if val is not None and p in [
                "nlive",
                "num_repeats",
                "nprior",
                "max_ncalls",
            ]:
                val = get_Xnumber(val, "d", self.dim, int, p)
            if val is not None:
                setattr(self.polychord_settings, p, val)
        if kwargs:
            warn(f"Some precision parameters not recognized; ignored: {kwargs}")

    def run(self, logp_func, param_names=None, out_dir=None, keep_all=False, seed=None):
        """
        Runs the nested sampler.

        param_names (optional, otherwise x_[i] will be used) should be a list of sampled
        parameter names, or a list of (name, label) tuples. Labels are interpreted as
        LaTeX but should not include '$' signs.
        """
        # only for NORA, not at init!!! (true like)
        #        # More efficient for const-eval-speed GP's (not very significant)
        #        self.polychord_settings.synchronous = False
        self.X_all, self.y_all = None, None
        self.X_MC, self.y_MC, self.w_MC = None, None, None
        if keep_all:
            warn("keep_all is currently experimental. It may not work as intended.")
            self.X_all = []
            self.y_all = []

            def logp_func_wrapped(*x):
                logp = logp_func(*x)
                self.X_all.append(x[0])
                self.y_all.append(logp)
                return logp, []

        else:
            logp_func_wrapped = logp_func
        # Configure folders and settings
        if mpi.is_main_process:
            base_dir, file_root = self.process_out_dir(
                out_dir, random_if_undefined=True
            )
            self.polychord_settings.base_dir = base_dir
            self.polychord_settings.file_root = file_root
        # Set seed (only that of rank 0 is used, and incremented internally by PolyChord)
        # See line 83 in PolyChordLite/src/polychord/random_utils.F90
        if seed is not None:
            self.polychord_settings.seed = seed
        mpi.share_attr(self, "polychord_settings")
        # Run PolyChord!
        # Flush stdout, since PolyChord can step over it if async (py not called with -u)
        sys.stdout.flush()
        with NumpyErrorHandling(all="ignore") as _:
            self.last_polychord_result = self.globals["run_polychord"](
                lambda X: (ensure_scalar(logp_func_wrapped(X)), []),
                nDims=self.dim,
                nDerived=0,
                settings=self.polychord_settings,
                prior=self.prior,
            )
        # Process results
        if keep_all:
            all_X_all = mpi.gather(self.X_all)
            all_y_all = mpi.gather(self.y_all)
            if mpi.is_main_process:
                self.X_all = np.concatenate(all_X_all)
                self.y_all = np.concatenate(all_y_all)
            else:
                self.X_all, self.y_all = None, None
        if mpi.is_main_process:
            if param_names is None:
                param_names = list(zip(*(2 * [generic_params_names(self.dim)])))
            elif isinstance(param_names[0], str):  # no labels specified
                param_names = [(p, p) for p in param_names]
            self.last_polychord_result.make_paramnames_files(param_names)
            samples_T = np.loadtxt(self.last_polychord_result.root + ".txt").T
            self.X_MC = samples_T[2:].T
            # PolyChord stores chi**2 in 2nd col (contrary to getdist: -logp)
            self.y_MC = -0.5 * samples_T[1]
            self.w_MC = samples_T[0]
        return self.X_MC, self.y_MC, self.w_MC

    def delete_output(self, out_dir=None):
        """
        Deletes the last PolyChord output.

        If ``out_dir`` specified, deletes the one stored there instead.
        """
        if not mpi.is_main_process:
            return
        if out_dir is None:
            base_dir = self.polychord_settings.base_dir
            file_root = self.polychord_settings.file_root
        else:
            base_dir, file_root = self.process_out_dir(
                out_dir, random_if_undefined=False
            )
        if not file_root:
            # Delete whole folder
            shutil.rmtree(base_dir)
            return
        files = glob.glob(os.path.join(base_dir, file_root + ".*"))
        files += glob.glob(os.path.join(base_dir, file_root + "_*"))
        files += glob.glob(os.path.join(base_dir, "clusters", file_root + "_*"))
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
        # Delete empty folders that may have been created by PolyChord
        try:
            os.rmdir(os.path.join(base_dir, "clusters"))
            os.rmdir(base_dir)
        except OSError:
            pass


class InterfaceNessai(NSInterface):
    """
    Interface for the ``nessai`` nested sampler, by M.J. Williams, J. Veitch and C.
    Messenger.

    See https://nessai.readthedocs.io
    """

    def __init__(self, bounds, verbosity=3):
        try:
            from nessai.model import Model as NessaiModel  # type: ignore
            from nessai.flowsampler import FlowSampler  # type: ignore

            self.globals = {"NessaiModel": NessaiModel, "FlowSampler": FlowSampler}
        except ModuleNotFoundError as excpt:
            raise NestedSamplerNotInstalledError(
                "External nested sampler 'nessai' cannot be imported. "
                "Check out installation instructions at "
                "https://nessai.readthedocs.io "
                "(or select an alternative nested sampler, e.g. UltraNest)."
            ) from excpt
        self.set_prior(bounds)
        self.dim = len(self.bounds)
        self.precision_settings = {}
        self.flow_sampler_settings = {
            "checkpointing": False,
            "resume": False,
            "plot": False,
            "log_on_iteration": None,  # will be defined later
            # seed: None,  # TODO: add seed
        }
        self.run_settings = {"plot": False, "save": False}
        # TODO: could still avoid dumping nested_sampler_resume.pkl and proposal/
        self.set_verbosity(verbosity)
        # Storage of last sample -- will only be defined for rank-0 MPI process
        self.X_all = None
        self.y_all = None
        self.X_MC = None
        self.y_MC = None
        self.w_MC = None
        self.output = None
        self.last_nessai_result = None

    def set_verbosity(self, verbose):
        """Sets the verbosity of the sampler at run time."""
        self.flow_sampler_settings["log_on_iteration"] = verbose > 3

    def set_prior(self, bounds):
        """Sets the prior used by the nested sampler."""
        self.bounds = check_and_return_bounds(bounds)

    def set_precision(self, nlive=None, precision_criterion=None, **kwargs):
        """Sets precision parameters for the nested sampler."""
        if nlive is not None:
            self.precision_settings["nlive"] = get_Xnumber(
                nlive, "d", self.dim, int, "nlive"
            )
        if precision_criterion is not None:
            self.precision_settings["stopping"] = precision_criterion
        if kwargs:
            warn(f"Some precision parameters not recognized; ignored: {kwargs}")

    def run(self, logp_func, param_names=None, out_dir=None, keep_all=False, seed=None):
        """
        Runs the nested sampler.

        param_names (optional, otherwise x_[i] will be used) should be a list of sampled
        parameter names, or a list of (name, label) tuples. Labels are interpreted as
        LaTeX but should not include '$' signs.
        """
        if keep_all:
            raise NotImplementedError("keep_all=True not yet possible for nessai.")
        NessaiModel = self.globals["NessaiModel"]
        FlowSampler = self.globals["FlowSampler"]

        class MyNessaiModel(NessaiModel):
            """
            Translates the logp function into a nessai model
            """

            def __init__(self, bounds, param_names):
                if param_names is None:
                    param_names = generic_params_names(len(bounds))
                self.log_prior_volume = np.sum(np.log(bounds[:, 1] - bounds[:, 0]))
                self.bounds = dict(
                    (name, bounds[i]) for i, name in enumerate(param_names)
                )

            @property
            def names(self):
                return list(self.bounds)

            def log_prior(self, x):
                if not self.in_bounds(x).any():
                    return -np.inf
                lp = np.single(0.0)
                return lp

            def log_likelihood(self, x):
                if x.ndim == 0:
                    point = np.array([x[p] for p in self.names])
                    points = np.atleast_2d(point)
                else:
                    points = np.array(
                        [[x[i][p] for p in self.names] for i in range(x.size)]
                    )
                return logp_func(points)

        self.output, _ = self.process_out_dir(out_dir, random_if_undefined=True)
        with NumpyErrorHandling(all="ignore") as _:
            sampler = FlowSampler(
                MyNessaiModel(self.bounds, param_names=param_names),
                output=self.output,
                seed=seed,
                **self.flow_sampler_settings,
                **self.precision_settings,
            )
            sampler.run(**self.run_settings)
        # Process results
        self.last_nessai_result = sampler.ns.get_result_dictionary()
        x = sampler.posterior_samples
        # Copy the data from the structured array to the float array
        dtype_new = np.dtype(
            {
                "names": x.dtype.names,
                "formats": tuple([np.float64] * len(x.dtype.names)),
            }
        )
        x = x.astype(dtype_new)
        posterior_samples = x.view(np.float64).reshape(x.shape[0], -1)
        self.X_MC = posterior_samples[:, :-3]
        self.y_MC = posterior_samples[:, -2]
        return self.X_MC, self.y_MC, None

    def delete_output(self, out_dir=None):
        """
        Deletes last the nessai output.

        If ``out_dir`` specified, deletes the one stored there instead.
        """
        if not mpi.is_main_process:
            return
        if out_dir is None:
            output = self.output
        else:
            output, _ = self.process_out_dir(out_dir, random_if_undefined=False)
        shutil.rmtree(output)


class InterfaceUltraNest(NSInterface):
    """
    Interface for the ``ultranest`` nested sampler, by J. Buchner.

    See https://johannesbuchner.github.io/UltraNest
    """

    def __init__(self, bounds, verbosity=3):
        try:
            from ultranest import ReactiveNestedSampler  # type: ignore

            self.globals = {"ReactiveNestedSampler": ReactiveNestedSampler}
        except ModuleNotFoundError as excpt:
            raise NestedSamplerNotInstalledError(
                "External nested sampler 'UltraNest' cannot be imported. "
                "Check out installation instructions at "
                "https://johannesbuchner.github.io/UltraNest "
                "(or select an alternative nested sampler, e.g. PolyChord or nessai)."
            ) from excpt
        self.set_prior(bounds)
        self.dim = len(self.bounds)
        self.precision_settings = {}

        self.sampler_settings = {
            "resume": "overwrite",
            "vectorized": True,
        }
        self.run_settings = {"viz_callback": False, "show_status": False}
        self.set_verbosity(verbosity)
        # Storage of last sample -- will only be defined for rank-0 MPI process
        self.X_all = None
        self.y_all = None
        self.X_MC = None
        self.y_MC = None
        self.w_MC = None
        self.output = None
        self.last_ultranest_result = None

    def set_verbosity(self, verbose):
        """Sets the verbosity of the sampler at run time."""
        # TODO
        pass

    def set_prior(self, bounds):
        """Sets the prior used by the nested sampler."""
        self.bounds = check_and_return_bounds(bounds)
        widths = self.bounds[:, 1] - self.bounds[:, 0]
        lowers = self.bounds[:, 0]
        self.uniform_prior_transform = lambda quantiles: quantiles * widths + lowers

    def set_precision(
        self, nlive=None, precision_criterion=None, max_ncalls=None, **kwargs
    ):
        """Sets precision parameters for the nested sampler."""
        if nlive is not None:
            self.precision_settings["min_num_live_points"] = get_Xnumber(
                nlive, "d", self.dim, int, "nlive"
            )
        if precision_criterion is not None:
            self.precision_settings["frac_remain"] = precision_criterion
        if max_ncalls is not None:
            self.precision_settings["max_ncalls"] = get_Xnumber(
                max_ncalls, "d", self.dim, int, "max_ncalls"
            )
        if kwargs:
            warn(f"Some precision parameters not recognized; ignored: {kwargs}")

    def run(self, logp_func, param_names=None, out_dir=None, keep_all=False, seed=None):
        """
        Runs the nested sampler.

        param_names (optional, otherwise x_[i] will be used) should be a list of sampled
        parameter names, or a list of (name, label) tuples. Labels are interpreted as
        LaTeX but should not include '$' signs.
        """
        if keep_all:
            raise NotImplementedError("keep_all=True not yet possible for ultranest.")
        if mpi.is_main_process:
            self.output, _ = self.process_out_dir(out_dir, random_if_undefined=True)
        sampler = self.globals["ReactiveNestedSampler"](
            param_names or generic_params_names(self.dim),
            logp_func,
            self.uniform_prior_transform,
            log_dir=self.output,
            **self.sampler_settings,
        )
        with NumpyErrorHandling(all="ignore") as _:
            self.last_ultranest_result = sampler.run(
                **self.precision_settings,
                **self.run_settings,
            )
        # Process results
        if mpi.is_main_process:
            w = self.last_ultranest_result["weighted_samples"]["weights"]
            X = self.last_ultranest_result["weighted_samples"]["points"]
            y = self.last_ultranest_result["weighted_samples"]["logl"]
            self.w_MC, self.X_MC, self.y_MC = remove_0_weight_samples(w, X, y)
        return self.X_MC, self.y_MC, self.w_MC

    def delete_output(self, out_dir=None):
        """
        Deletes the last ultranest output.

        If ``out_dir`` specified, deletes the one stored there instead.
        """
        if not mpi.is_main_process:
            return
        if out_dir is None:
            output = self.output
        else:
            output, _ = self.process_out_dir(out_dir, random_if_undefined=False)
        shutil.rmtree(self.output)


# Implemented interfaces as a dict, for convenience.
_ns_interfaces = {
    "polychord": InterfacePolyChord,
    "ultranest": InterfaceUltraNest,
    "nessai": InterfaceNessai,
}
