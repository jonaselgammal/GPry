"""
Wrappers for external nested samplers.
"""

import shutil
import tempfile
from warnings import warn

import numpy as np

from gpry import mpi
from gpry.tools import NumpyErrorHandling


class NestedSamplerNotInstalledError(Exception):
    """
    Exception to be raised at initialization of any of the interfaces if the NS failed to
    be imported.
    """


class InterfacePolyChord:
    """
    Interface for the PolyChord nested sampler, by W. Handley, M. Hobson & A. Lasenby.

    See https://github.com/PolyChord/PolyChordLite
    """

    def __init__(self, bounds, verbosity=3):
        try:
            # pylint: disable=import-outside-toplevel
            from pypolychord.settings import PolyChordSettings
        except ModuleNotFoundError as excpt:
            raise NestedSamplerNotInstalledError(
                "External nested sampler Polychord cannot be imported. "
                "Check out installation instructions at "
                "https://github.com/PolyChord/PolyChordLite "
                "(or select an alternative nested sampler, e.g. UltraNest)."
            ) from excpt
        self.dim = len(np.atleast_2d(bounds))
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
        self.last_polychord_output = None

    def set_verbosity(self, verbose):
        """Sets the verbosity of the sampler at run time."""
        self.polychord_settings.feedback = verbose - 3
        # 0: print header and result; not very useful: turn it to -1 if that's the case
        if self.polychord_settings.feedback == 0:
            self.polychord_settings.feedback = -1

    def set_prior(self, bounds):
        """Sets the prior used by the nested sampler."""
        # pylint: disable=import-outside-toplevel
        from pypolychord.priors import UniformPrior

        self.prior = UniformPrior(*np.atleast_2d(bounds.T))

    def set_precision(self, **kwargs):
        """Sets precision parameters for the nested sampler."""
        known = ["nlive", "num_repeats", "precision_criterion", "nprior", "max_ncalls"]
        for p in known:
            val = kwargs.pop(p, None)
            if val is not None:
                setattr(self.polychord_settings, p, val)
        if kwargs:
            warn(f"Some precision parameters not recognized; ignored: {kwargs}")

    def run(self, logp_func, out_dir=None, keep_all=False):
        """Runs the nested sampler."""
        # only for NORA, not at init!!! (true like)
        #        # More efficient for const-eval-speed GP's (not very significant)
        #        self.polychord_settings.synchronous = False

        self.X_all, self.y_all = None, None
        self.X_MC, self.y_MC, self.w_MC = None, None, None
        if keep_all:
            self.X_all = []
            self.y_all = []

            def logp_func_wrapped(*x):
                logp = logp_func(*x)
                self.X_all.append(x)
                self.y_all.append(logp)
                return logp

        else:
            logp_func_wrapped = logp_func
        # Configure folders and settings
        if mpi.is_main_process:
            if out_dir is None:
                # pylint: disable=consider-using-with
                out_dir = tempfile.TemporaryDirectory().name
            setattr(self.polychord_settings, "base_dir", out_dir)
            setattr(self.polychord_settings, "file_root", "test")
        mpi.share_attr(self, "polychord_settings")
        # Run PolyChord!
        from pypolychord import run_polychord  # pylint: disable=import-outside-toplevel

        with NumpyErrorHandling(all="ignore") as _:
            self.last_polychord_output = run_polychord(
                logp_func_wrapped,
                nDims=self.dim,
                nDerived=0,
                settings=self.polychord_settings,
                prior=self.prior,
            )
        # Process results
        if keep_all:
            all_X_all = mpi.comm.gather(self.X_all)
            all_y_all = mpi.comm.gather(self.y_all)
            if mpi.is_main_process:
                self.X_all = np.concatenate(all_X_all)
                self.y_all = np.concatenate(all_y_all)
            else:
                self.X_all, self.y_all = None, None
        if mpi.is_main_process:
            dummy_paramnames = [tuple(2 * [f"x_{i + 1}"]) for i in range(self.dim)]
            self.last_polychord_output.make_paramnames_files(dummy_paramnames)
            samples_T = np.loadtxt(self.last_polychord_output.root + ".txt").T
            self.X_MC = samples_T[2:].T
            self.y_MC = -0.5 * samples_T[1]  # this one stores chi2
            self.w_MC = samples_T[0]
            # Delete products from tmp folder
            shutil.rmtree(self.polychord_settings.base_dir)
        return self.X_MC, self.y_MC, self.w_MC
