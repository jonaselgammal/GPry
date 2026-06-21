"""
Wrappers for external nested samplers.
"""

import os
import sys
import glob
import shutil
import tempfile
import importlib.abc
from warnings import warn
from abc import ABC, abstractmethod
from contextlib import contextmanager

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
_SERIAL_MPI_ENV = "GPRY_DISABLE_MPI"


class NativeNestedSamplingLogLikelihood:
    """Callable log-likelihood plus optional native backend helpers."""

    def __init__(self, numpy_loglikelihood, jax_builder=None):
        self._numpy_loglikelihood = numpy_loglikelihood
        self._jax_builder = jax_builder

    def __call__(self, X):
        return self._numpy_loglikelihood(X)

    def build_jax_loglikelihood(self, param_names_list):
        if self._jax_builder is None:
            return None
        return self._jax_builder(param_names_list)


class _BlockedModuleFinder(importlib.abc.MetaPathFinder):
    """Forces ImportError for selected top-level modules."""

    def __init__(self, blocked_names):
        self.blocked_names = tuple(blocked_names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.blocked_names or fullname.startswith(
            tuple(f"{name}." for name in self.blocked_names)
        ):
            raise ImportError(f"{fullname} import blocked for serial execution")
        return None


@contextmanager
def _serial_mpi_import_block():
    """
    Temporarily prevents mpi4py from importing so PolyChord runs in true serial mode.
    """
    if os.environ.get(_SERIAL_MPI_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        yield
        return
    removed = {}
    blocked = ("mpi4py",)
    for name in list(sys.modules):
        if name in blocked or any(name.startswith(f"{prefix}.") for prefix in blocked):
            removed[name] = sys.modules.pop(name)
    finder = _BlockedModuleFinder(blocked)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        sys.modules.update(removed)


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
            with _serial_mpi_import_block():
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
        self,
        nlive=None,
        precision_criterion=None,
        num_repeats=None,
        max_ncalls=None,
        **kwargs,
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
        if num_repeats is not None:
            # Maps to the slice-sampler's number of steps (see run()).
            self.precision_settings["nsteps"] = get_Xnumber(
                num_repeats, "d", self.dim, int, "num_repeats"
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
        # Robust configuration (ported from surrogate_gpr_split f8e2b2e):
        # PolyChord-like slice sampling + SimpleRegion. The default MLFriends
        # region + region-based stepper is fragile in higher dimensions and on
        # multimodal targets (produced broken posteriors); slice sampling with
        # mixed hit-and-run directions is the robust choice.
        import ultranest.stepsampler as unst  # type: ignore
        from ultranest.mlfriends import SimpleRegion  # type: ignore

        if mpi.is_main_process:
            self.output, _ = self.process_out_dir(out_dir, random_if_undefined=True)
        sampler = self.globals["ReactiveNestedSampler"](
            param_names or generic_params_names(self.dim),
            logp_func,
            self.uniform_prior_transform,
            log_dir=self.output,
            **self.sampler_settings,
        )
        # ``nsteps`` is set from ``num_repeats`` via set_precision. Fall back to a
        # PolyChord-like 2*dim default if a caller did not provide it, so non-NORA
        # UltraNest paths do not crash (more defensive than the original port).
        nsteps = self.precision_settings.pop("nsteps", None)
        if nsteps is None:
            nsteps = 2 * self.dim
        sampler.stepsampler = unst.SliceSampler(
            nsteps=nsteps,
            generate_direction=unst.generate_mixture_random_direction,
        )
        self.run_settings["region_class"] = SimpleRegion
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


class InterfaceBlackJAX(NSInterface):
    """
    Interface for the BlackJAX nested sampler (Handley-lab fork with NS support).

    Uses Nested Slice Sampling (NSS) with Hit-and-Run Slice Sampling as the
    inner kernel. Fully JAX-native, JIT-compilable, and GPU-compatible.

    See https://github.com/handley-lab/blackjax
    """

    def __init__(self, bounds, verbosity=3):
        try:
            from blackjax.ns.nss import as_top_level_api as _nss_api
            from blackjax.ns import utils as ns_utils

            self.globals = {"nss_api": _nss_api, "ns_utils": ns_utils}
        except (ModuleNotFoundError, ImportError, AttributeError) as excpt:
            raise NestedSamplerNotInstalledError(
                "BlackJAX nested sampler (handley-lab fork) cannot be imported. "
                "Install it with: pip install git+https://github.com/handley-lab/blackjax.git "
                "(or select an alternative nested sampler, e.g. UltraNest)."
            ) from excpt
        bounds = check_and_return_bounds(bounds)
        self.dim = len(bounds)
        self.bounds = bounds
        self.precision_settings = {
            # Match GPry's standard NORA cap used with PolyChord by default.
            "nlive": 25 * self.dim,
            "max_steps": 5000,
            "num_inner_steps": 5 * self.dim,
            "precision_criterion": 0.01,
            # ``num_delete`` controls how many particles are replaced per
            # NSS outer step. The BlackJAX NSS kernel ``vmap``s the inner
            # MCMC over this dimension; with the upstream default of 1, the
            # vmap is over a singleton so each step does no parallel work.
            # David Yallup's recommended target is roughly ``nlive//5``:
            # the outer-step count drops ~5x, each step does ~5x more
            # vectorized work, and per-call wall drops measurably (~40%
            # in the 5D GaussianMix5D profile). Floor at 1 to retain
            # correctness in pathological cases.
            "num_delete": max(1, (25 * self.dim) // 5),
        }
        self.set_verbosity(verbosity)
        # Storage
        self.X_MC = None
        self.y_MC = None
        self.w_MC = None
        self._compiled_runtime_cache = {}

    def set_verbosity(self, verbose):
        """Sets the verbosity of the sampler at run time."""
        self.verbose = verbose

    def set_prior(self, bounds):
        """Sets the prior used by the nested sampler."""
        self.bounds = check_and_return_bounds(bounds)

    def set_precision(
        self,
        nlive=None,
        max_steps=None,
        num_inner_steps=None,
        precision_criterion=None,
        num_repeats=None,
        max_ncalls=None,
        nprior=None,
        num_delete=None,
        **kwargs,
    ):
        """Sets precision parameters for the nested sampler."""
        if nlive is not None:
            new_nlive = get_Xnumber(nlive, "d", self.dim, int, "nlive")
            self.precision_settings["nlive"] = new_nlive
            # Keep num_delete proportional to nlive unless the caller is also
            # setting it explicitly (handled below).
            if num_delete is None:
                self.precision_settings["num_delete"] = max(1, new_nlive // 5)
        if num_delete is not None:
            self.precision_settings["num_delete"] = max(
                1, get_Xnumber(num_delete, "d", self.dim, int, "num_delete")
            )
        if num_inner_steps is None and num_repeats is not None:
            num_inner_steps = num_repeats
        if num_inner_steps is not None:
            self.precision_settings["num_inner_steps"] = get_Xnumber(
                num_inner_steps, "d", self.dim, int, "num_inner_steps"
            )
        if max_steps is None and max_ncalls is not None:
            # One NSS outer step entails at least one deleted particle and a constrained
            # MCMC update with multiple inner steps. This lower-bound mapping is still
            # imperfect, but is materially closer to a likelihood-call budget than
            # equating max_ncalls with the number of outer iterations.
            inner_steps = max(1, self.precision_settings["num_inner_steps"])
            max_steps = max(1, int(max_ncalls) // inner_steps)
        if max_steps is not None:
            self.precision_settings["max_steps"] = int(max_steps)
        if precision_criterion is not None:
            self.precision_settings["precision_criterion"] = float(
                precision_criterion
            )
        if nprior is not None:
            # Kept for compatibility with the PolyChord precision contract.
            self.precision_settings["nprior"] = int(nprior)
        if kwargs:
            warn(f"Some precision parameters not recognized; ignored: {kwargs}")

    @staticmethod
    def _logdiffexp(log_a, log_b):
        """Returns log(exp(log_a) - exp(log_b)) for log_a > log_b."""
        return log_a + np.log1p(-np.exp(log_b - log_a))

    def _stop_by_remaining_evidence(
        self,
        dead_logls,
        dead_count,
        live_logl,
        nlive,
        precision_criterion,
    ):
        if precision_criterion is None or precision_criterion <= 0 or dead_count <= 0:
            return False, None
        logx_prev = 0.0
        logz_dead = -np.inf
        for idx, dead_logl in enumerate(dead_logls, start=1):
            logx_curr = -idx / float(nlive)
            logdx = self._logdiffexp(logx_prev, logx_curr)
            logz_dead = np.logaddexp(logz_dead, float(dead_logl) + logdx)
            logx_prev = logx_curr
        logz_live_upper = float(np.max(live_logl)) + logx_prev
        logz_total_upper = np.logaddexp(logz_dead, logz_live_upper)
        frac_remain = float(np.exp(logz_live_upper - logz_total_upper))
        return frac_remain < precision_criterion, frac_remain

    def run(self, logp_func, param_names=None, out_dir=None, keep_all=False, seed=None):
        """
        Runs the BlackJAX nested sampler.

        Parameters
        ----------
        logp_func : callable
            Log-likelihood function. Takes array X of shape (d,) or (n, d)
            and returns scalar or array.
        param_names : list, optional
            Parameter names.
        out_dir : str, optional
            Not used (BlackJAX is in-memory), but kept for interface compatibility.
        keep_all : bool
            Not supported for BlackJAX.
        seed : int, optional
            Random seed.

        Returns
        -------
        (X_MC, y_MC, w_MC) : arrays of samples, log-likelihoods, and normalized weights.
        """
        import jax
        import jax.numpy as jnp
        from blackjax.ns import utils as ns_utils

        if keep_all:
            raise NotImplementedError("keep_all=True not supported for BlackJAX.")

        nlive = self.precision_settings["nlive"]
        max_steps = self.precision_settings["max_steps"]
        num_inner_steps = self.precision_settings["num_inner_steps"]
        precision_criterion = self.precision_settings.get("precision_criterion")
        num_delete = max(1, int(self.precision_settings.get("num_delete", 1)))

        if seed is None:
            seed = np.random.randint(0, 2**31)
        rng_key = jax.random.PRNGKey(seed)

        # Build parameter bounds dict for BlackJAX
        if param_names is None:
            param_names_list = generic_params_names(self.dim)
        elif isinstance(param_names[0], (list, tuple)):
            param_names_list = [p[0] for p in param_names]
        else:
            param_names_list = list(param_names)

        bounds_dict = {
            name: (float(self.bounds[i, 0]), float(self.bounds[i, 1]))
            for i, name in enumerate(param_names_list)
        }

        # Generate initial particles from uniform prior
        rng_key, init_key = jax.random.split(rng_key)
        particles, logprior_fn = ns_utils.uniform_prior(init_key, nlive, bounds_dict)

        # Build log-likelihood that takes a dict of params -> scalar.
        # Prefer an end-to-end JAX path when GPry provides one; otherwise fall back
        # to pure_callback around the numpy likelihood.
        jax_loglikelihood_builder = None
        # Stable cache identity for the runtime-cache key. Bound methods
        # (``logp_func.build_jax_loglikelihood``) have a fresh ``id`` on every
        # attribute access, so keying the cache on the bound method id makes
        # every call miss. Use the underlying ``_jax_builder`` function id
        # instead — that's what the surrogate caches in
        # ``SurrogateModel.make_ns_loglikelihood_adapter``, and it is stable
        # across calls.
        jax_loglikelihood_builder_id = None
        if isinstance(logp_func, NativeNestedSamplingLogLikelihood):
            jax_loglikelihood_builder = logp_func.build_jax_loglikelihood
            jax_loglikelihood_builder_id = id(
                getattr(logp_func, "_jax_builder", None)
                or logp_func.build_jax_loglikelihood
            )
        if jax_loglikelihood_builder is not None:
            loglikelihood_fn = jax_loglikelihood_builder(param_names_list)
        else:
            loglikelihood_fn = None
        if loglikelihood_fn is not None:
            jax_path = True
        else:
            pure_callback_kwargs = (
                {"vmap_method": "sequential"}
                if "vmap_method" in jax.pure_callback.__code__.co_varnames
                else {"vectorized": False}
            )

            def loglikelihood_fn(params):
                x = jnp.array([params[name] for name in param_names_list])

                def _numpy_logp(x_np):
                    # ``logp_func`` may return a 0-d or 1-element array; force
                    # to a plain Python float before promoting to np.float64
                    # to avoid the numpy-1.25 ndarray->scalar deprecation
                    # firing inside every NORA NS step (this callback is the
                    # tight inner loop of BlackJAX when no JAX adapter is
                    # available, e.g. in the e2e test runs).
                    return np.float64(
                        float(np.asarray(logp_func(np.asarray(x_np))).reshape(()))
                    )

                result = jax.pure_callback(
                    _numpy_logp, jax.ShapeDtypeStruct((), jnp.float64), x,
                    **pure_callback_kwargs,
                )
                return result
            jax_path = False

        # Build the NSS algorithm
        nss_api = self.globals["nss_api"]
        runtime_key = None
        if jax_path:
            runtime_key = (
                tuple(param_names_list),
                tuple(map(tuple, np.asarray(self.bounds))),
                int(nlive),
                int(num_inner_steps),
                int(num_delete),
                "builder",
                jax_loglikelihood_builder_id,
            )
        cached_runtime = self._compiled_runtime_cache.get(runtime_key)
        if cached_runtime is None:
            algorithm = nss_api(
                logprior_fn=logprior_fn,
                loglikelihood_fn=loglikelihood_fn,
                num_inner_steps=num_inner_steps,
                num_delete=num_delete,
            )
            if jax_path:
                init_fn = jax.jit(algorithm.init)
                step_fn = jax.jit(algorithm.step)
                self._compiled_runtime_cache[runtime_key] = (init_fn, step_fn)
            else:
                init_fn = algorithm.init
                step_fn = algorithm.step
        else:
            init_fn, step_fn = cached_runtime

        # Initialize
        rng_key, init_key = jax.random.split(rng_key)
        state = init_fn(particles, rng_key=init_key)

        # Run the NS loop, collecting dead particles
        dead_list = []
        dead_logls = []

        for i in range(max_steps):
            rng_key, step_key = jax.random.split(rng_key)
            state, info = step_fn(step_key, state)
            dead_list.append(info)
            dead_logls.extend(np.sort(np.asarray(info.particles.loglikelihood).ravel()))

            if precision_criterion is not None and precision_criterion > 0 and i > 0:
                # Check every ``nlive`` *dead particles*, not every ``nlive``
                # steps. With ``num_delete > 1``, each outer step generates
                # ``num_delete`` deaths -- the original ``(i + 1) % nlive``
                # check fires once per ``nlive * num_delete`` particles, so
                # convergence detection lags by ``num_delete x``. For
                # ``num_delete=25`` and the precision_criterion=0.01
                # default, that lag inflates the dead-particle pool ~10x
                # and adds significant wall-time in NORA's downstream
                # ranking + finalise.
                check_every = max(1, nlive // max(1, num_delete))
                should_check = ((i + 1) % check_every == 0)
                if should_check:
                    should_stop, frac_remain = self._stop_by_remaining_evidence(
                        dead_logls=dead_logls,
                        dead_count=len(dead_logls),
                        live_logl=np.asarray(state.particles.loglikelihood),
                        nlive=nlive,
                        precision_criterion=precision_criterion,
                    )
                    if should_stop:
                        if self.verbose > 3:
                            print(
                                f"BlackJAX NS converged at step {i + 1}, "
                                f"remaining evidence fraction = {frac_remain:.4g}"
                            )
                        break

        # Finalise: combine dead particles with final live points
        dead_info = ns_utils.finalise(state, dead_list, update_info=False)

        # Return dead particles with importance weights (not resampled posterior).
        # Dead particles span the full prior volume, giving NORA diverse
        # proposals for ranking. Resampled posterior would concentrate all
        # samples at the mode, where training points already exist.
        rng_key, weight_key = jax.random.split(rng_key)
        # ``ns_utils.log_weights`` is responsible for ~50% of per-call
        # BlackJAX time (~0.9s on 5D NORA, per /tmp/gpry_finalise_timing.py).
        # A naive ``jax.jit`` wrapper fails with NonConcreteBooleanIndex
        # because ``logX`` inside uses data-dependent boolean indexing -- so
        # the win, if there is one, is in the upstream blackjax library.
        # Logged as a follow-up optimization candidate.
        logw = ns_utils.log_weights(weight_key, dead_info).mean(axis=-1)
        logw = np.asarray(logw)

        particles = dead_info.particles
        n_particles = len(particles.loglikelihood)

        # Vectorized construction of X_mc. The previous double-loop with
        # ``float(particles.position[name][j])`` forced one host-side scalar
        # read per (particle, dim) cell — at ~3M cells per call on a 5D
        # GaussianMix5D run with nlive=125, max_steps=5000, that adds ~0.4s
        # *per BlackJAX call* (15% of the 5D NORA wall). ``np.asarray`` on a
        # CPU JAX array shares the buffer, and ``column_stack`` builds the
        # (n_particles, d) layout in a single pass.
        X_mc = np.column_stack([
            np.asarray(particles.position[name]) for name in param_names_list
        ])
        y_mc = np.asarray(particles.loglikelihood)

        # Convert log-weights to normalized importance weights.
        w_mc = np.exp(logw - np.max(logw))
        w_mc = w_mc / np.sum(w_mc)

        self.X_MC = X_mc
        self.y_MC = y_mc
        self.w_MC = w_mc
        return self.X_MC, self.y_MC, self.w_MC

    def delete_output(self, out_dir=None):
        """BlackJAX is in-memory, nothing to delete."""
        pass


# Implemented interfaces as a dict, for convenience.
_ns_interfaces = {
    "polychord": InterfacePolyChord,
    "ultranest": InterfaceUltraNest,
    "nessai": InterfaceNessai,
    "blackjax": InterfaceBlackJAX,
}
