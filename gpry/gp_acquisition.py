"""
GPAcquisition classes, which take care of proposing new locations where to evaluate the
true function.
"""

import sys
import shutil
import warnings
import inspect
import tempfile
from time import time
from copy import deepcopy
from typing import Mapping
from functools import partial
import numpy as np
import scipy.optimize
from sklearn.base import is_regressor

import gpry.acquisition_functions as gpryacqfuncs
from gpry.proposal import PartialProposer, CentroidsProposer, Proposer, UniformProposer
from gpry import mpi
from gpry.tools import NumpyErrorHandling, get_Xnumber, generic_params_names


# TODO: inconsistent use of random_state: passed at init, and also at acquisition time


def builtin_names():
    """
    Lists all names of all built-in acquisition functions criteria.
    """
    list_names = [name for name, obj in inspect.getmembers(sys.modules[__name__])
                  if (issubclass(obj.__class__, GenericGPAcquisition.__class__) and
                      obj is not GenericGPAcquisition)]
    return list_names


class GenericGPAcquisition():
    """Generic class for acquisition objects."""

    def __init__(self,
                 bounds,
                 preprocessing_X=None,
                 random_state=None,
                 verbose=1,
                 acq_func="LogExp",
                 # DEPRECATED ON 13-09-2023:
                 zeta=None,
                 zeta_scaling=None,
                 ):
        self.bounds = np.array(bounds)
        self.n_d = bounds.shape[0]
        self.preprocessing_X = preprocessing_X
        self.verbose = verbose
        self.random_state = random_state
        if gpryacqfuncs.is_acquisition_function(acq_func):
            self.acq_func = acq_func
        elif isinstance(acq_func, (Mapping, str)):
            if isinstance(acq_func, str):
                acq_func = {acq_func: {}}
            # DEPRECATED ON 13-09-2023:
            if zeta is not None or zeta_scaling is not None:
                print(
                    "*Warning*: 'zeta' and 'zeta_scaling' have been deprecated as kwargs "
                    "and should be passed inside the 'acq_func' arg dict, e.g. "
                    "'acq_func={\"LogExp\": {\"zeta_scaling\": 0.85}}'. The given values "
                    "are being used, but this will fail in the future."
                )
                acq_func[list(acq_func)[0]].update(
                    {"zeta": zeta, "zeta_scaling": zeta_scaling})
            # END OF DEPRECATION BLOCK
            acq_func_name = list(acq_func)[0]
            acq_func_args = acq_func[acq_func_name] or {}
            acq_func_args["dimension"] = self.n_d
            try:
                acq_func_class = getattr(gpryacqfuncs, acq_func_name)
            except AttributeError as excpt:
                raise ValueError(
                    f"Unknown AcquisitionFunction class {acq_func_name}. "
                    f"Available convergence criteria: {gpryacqfuncs.builtin_names()}"
                ) from excpt
            try:
                self.acq_func = acq_func_class(**acq_func_args)
            except Exception as excpt:
                raise ValueError(
                    "Error when initialising the AcquisitionFunction object "
                    f"{acq_func_name} with arguments {acq_func_args}: "
                    f"{str(excpt)}"
                ) from excpt
        else:
            raise TypeError(
                "acq_func should be an AcquisitionFunction or a str or dict "
                f"specification. Got {acq_func}"
            )

    def __call__(self, X, gpr, eval_gradient=False):
        """Returns the value of the acquision function at ``X`` given a ``gpr``."""
        return self.acq_func(X, gpr, eval_gradient=eval_gradient)

    def multi_add(self, gpr, n_points=1, random_state=None):
        """
        Method to query multiple points where the objective function
        shall be evaluated.
        """

class BatchOptimizer(GenericGPAcquisition):
    """
    Run Gaussian Process acquisition.

    Works similarly to a GPRegressor but instead of optimizing the kernel's
    hyperparameters it optimizes the Acquisition function in order to find one
    or multiple points at which the likelihood/posterior should be evaluated
    next.

    Furthermore contains a framework for different lying strategies in order to
    improve the performance if multiple processors are available

    Use this class directly if you want to control the iterations of your
    bayesian quadrature loop.

    Parameters
    ----------
    bounds : array
        Bounds in which to optimize the acquisition function,
        assumed to be of shape (d,2) for d dimensional prior

    proposer : Proposer object, optional (default: "ParialProposer", producing a mixture
        of points drawn from an "UniformProposer" and from a "CentroidsProposer")
        Proposes points from which the acquisition function should be optimized.

    acq_func : GPry Acquisition Function, optional (default: "LogExp")
        Acquisition function to maximize/minimize. If none is given the
        `LogExp` acquisition function will be used

    acq_optimizer : string or callable, optional (default: "auto")
        Can either be one of the internally supported optimizers for optimizing
        the acquisition function, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

            def optimizer(obj_func, initial_guess, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_guess': the initial value for X, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of X
                ....
                # Returned are the best found X and
                # the corresponding value of the target function.
                return X_opt, func_min

        if set to 'auto' either the 'fmin_l_bfgs_b' or 'sampling' algorithm
        from scipy.optimize is used depending on whether gradient information
        is available or not.

        .. note::
            The default optimizers are designed to **maximize** the acquisition
            function.

    preprocessing_X : X-preprocessor, Pipeline_X, optional (default: None)
        Single preprocessor or pipeline of preprocessors for X. Preprocessing
        makes sense if the scales along the different dimensions are vastly
        different which means that the optimizer struggles to find the maximum
        of the acquisition function. If None is passed the data is not
        preprocessed.

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the maximum of the
        acquisition function. The first run of the optimizer is performed from
        the last X fit to the model if available, otherwise it is drawn at
        random.

        The remaining ones (if any) from X's sampled uniform randomly
        from the space of allowed X-values. Note that n_restarts_optimizer == 0
        implies that one run is performed.

    random_state : int or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    zeta_scaling : float, optional (default: 1.1)
        The scaling of the acquisition function's zeta parameter with dimensionality
        (Only if "LogExp" is passed as acquisition_function)

    zeta: float, optional (default: None, uses zeta_scaling)
        Specifies the value of the zeta parameter directly.

    verbose : 1, 2, 3, optional (default: 1)
        Level of verbosity. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise.

    Attributes
    ----------
    gpr_ : GaussianProcessRegressor
            The GP Regressor which is currently used for optimization.

    """

    def __init__(self,
                 bounds,
                 preprocessing_X=None,
                 random_state=None,
                 verbose=1,
                 acq_func="LogExp",
                 # DEPRECATED ON 13-09-2023:
                 zeta=None,
                 zeta_scaling=None,
                 # Class-specific:
                 proposer=None,
                 acq_optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer="5d",
                 n_repeats_propose=10,
                 ):
        super().__init__(
            bounds=bounds, preprocessing_X=preprocessing_X, random_state=random_state,
            verbose=verbose, acq_func=acq_func, zeta=zeta, zeta_scaling=zeta_scaling)
        self.proposer = proposer
        self.obj_func = None

        # If nothing is provided for the proposal, we use a centroids proposer with
        # a fraction of uniform samples.
        if self.proposer is None:
            self.proposer = PartialProposer(self.bounds, CentroidsProposer(self.bounds))
        else:
            if not isinstance(proposer, Proposer):
                raise TypeError(
                    "'proposer' must be a Proposer instance. "
                    f"Got {proposer} of type {type(proposer)}."
                )
            self.proposer = proposer

        # Configure optimizer
        # decide optimizer based on gradient information
        if acq_optimizer == "auto":
            if self.acq_func.hasgradient:
                self.acq_optimizer = "fmin_l_bfgs_b"
            else:
                self.acq_optimizer = "sampling"

        elif isinstance(acq_optimizer, str):
            if acq_optimizer == "fmin_l_bfgs_b":
                if not self.acq_func.hasgradient:
                    raise ValueError(
                        "In order to use the 'fmin_l_bfgs_b' "
                        "optimizer the acquisition function needs to be able "
                        "to return gradients. Got %s" % self.acq_func)
                self.acq_optimizer = "fmin_l_bfgs_b"
            elif acq_optimizer == "sampling":
                self.acq_optimizer = "sampling"
            else:
                raise ValueError("Supported internal optimizers are 'auto', "
                                 "'lbfgs' or 'sampling', "
                                 "got {0}".format(acq_optimizer))
        else:
            self.acq_optimizer = acq_optimizer
        self.n_restarts_optimizer = get_Xnumber(
            n_restarts_optimizer, "d", self.n_d, int, "n_restarts_optimizer"
        )
        self.n_repeats_propose = n_repeats_propose
        self.mean_ = None
        self.cov = None

    def optimize_acquisition_function(self, gpr, i, random_state=None):
        """Exposes the optimization method for the acquisition function. When
        called it proposes a single point where for where to evaluate the true
        model next. It is internally called in the :meth:`multi_add` method.

        Parameters
        ----------
        gpr : GaussianProcessRegressor
            The GP Regressor which is used as surrogate model.

        i : int
            Internal counter which is used to enable MPI support. If you want
            to optimize from a single location and rerun the optimizer from
            multiple starting locations loop over this parameter.

        random_state : int or numpy.RandomState, optional
            The generator used to initialize the centers. If an integer is
            given, it fixes the seed. Defaults to the global numpy random
            number generator.

        Returns
        -------
        X_opt : numpy.ndarray, shape = (X_dim,)
            The X value of the found optimum
        func : float
            The value of the acquisition function at X_opt
        """
        # Update proposer with new gpr
        self.proposer.update(gpr)

        # If we do a first-time run, use this
        if not self.obj_func:

            # Check whether gpr is a GP regressor
            if not is_regressor(gpr):
                raise ValueError("surrogate model has to be a GP Regressor. "
                                 "Got %s instead." % gpr)

            # Check whether the GP has been fit to data before
            if not hasattr(gpr, "X_train_"):
                raise AttributeError(
                    "The model which is given has not been fed "
                    "any points. Please make sure, that the model already "
                    "contains data when trying to optimize an acquisition "
                    "function on it as optimizing priors is not supported yet.")

            def obj_func(X, eval_gradient=False):

                # TODO: optionally suppress this checks if called by optimiser
                # Check inputs
                X = np.asarray(X)
                X = np.expand_dims(X, axis=0)
                if X.ndim != 2:
                    raise ValueError("X is {}-dimensional, however, "
                                     "it must be 2-dimensional.".format(X.ndim))
                if self.preprocessing_X is not None:
                    X = self.preprocessing_X.inverse_transform(X)

                if eval_gradient:
                    acq, grad = self.acq_func(X, gpr, eval_gradient=True)
                    return -1 * acq, -1 * grad
                else:
                    return -1 * self.acq_func(X, gpr, eval_gradient=False)

            self.obj_func = obj_func

        # Preprocessing
        if self.preprocessing_X is not None:
            transformed_bounds = self.preprocessing_X.transform_bounds(
                self.bounds)
        else:
            transformed_bounds = self.bounds

        if i == 0:
            # Perform first run from last training point
            x0 = gpr.X_train[-1]
            if self.preprocessing_X is not None:
                x0 = self.preprocessing_X.transform(x0)
            return self._constrained_optimization(self.obj_func, x0,
                                                  transformed_bounds)
        else:
            n_tries = 10 * self.bounds.shape[0] * self.n_restarts_optimizer
            x0s = np.empty((self.n_repeats_propose + 1, self.bounds.shape[0]))
            values = np.empty(self.n_repeats_propose + 1)
            ifull = 0
            for n_try in range(n_tries):
                x0 = self.proposer.get(random_state=random_state)
                value = self.acq_func(x0, gpr)
                if not np.isfinite(value):
                    continue
                x0s[ifull] = x0
                values[ifull] = value
                ifull += 1
                if ifull > self.n_repeats_propose:
                    x0 = x0s[np.argmax(values)]
                    if self.preprocessing_X is not None:
                        x0 = self.preprocessing_X.transform(x0)
                    return self._constrained_optimization(self.obj_func, x0,
                                                          transformed_bounds)
            # if there's at least one finite value try optimizing from
            # there, otherwise take the last x0 and add that to the GP
            if ifull > 0:
                x0 = x0s[np.argmax(values[:ifull])]
                if self.preprocessing_X is not None:
                    x0 = self.preprocessing_X.transform(x0)
                return self._constrained_optimization(self.obj_func, x0,
                                                      transformed_bounds)
            else:
                if self.verbose > 1:
                    print(f"of {n_tries} initial samples for the "
                          "acquisition optimizer none returned a "
                          "finite value")
                if self.preprocessing_X is not None:
                    x0 = self.preprocessing_X.transform(x0)
                return x0, -1 * value

    def multi_add(self, gpr, n_points=1, random_state=None, force_resample=False):
        r"""Method to query multiple points where the objective function
        shall be evaluated. The strategy which is used to query multiple
        points is by using the :math:`f(x)\sim \mu(x)` strategy and and not
        changing the hyperparameters of the model.

        This is done to increase speed since then the blockwise matrix
        inversion lemma can be used to invert the K matrix. The optimization
        for a single point is done using the :meth:`optimize_acq_func` method.

        When run in parallel (MPI), returns the same values for all processes.

        Parameters
        ----------
        gpr : GaussianProcessRegressor
            The GP Regressor which is used as surrogate model.

        n_points : int, optional (default=1)
            Number of points returned by the optimize method
            If the value is 1, a single point to evaluate is returned.

            Otherwise a list of points to evaluate is returned of size
            n_points. This is useful if you can evaluate your objective
            in parallel, and thus obtain more objective function evaluations
            per unit of time.

        random_state : int or numpy.RandomState, optional
            The generator used to initialize the centers. If an integer is
            given, it fixes the seed. Defaults to the global numpy random
            number generator.

        Returns
        -------
        X : numpy.ndarray, shape = (X_dim, n_points)
            The X values of the found optima
        y_lies : numpy.ndarray, shape = (n_points,)
            The predicted values of the GP at the proposed sampling locations
        fval : numpy.ndarray, shape = (n_points,)
            The values of the acquisition function at X_opt
        """
        # Check if n_points is positive and an integer
        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(
                "n_points should be int > 0, got " + str(n_points)
            )
        if mpi.is_main_process:
            # Initialize arrays for storing the optimized points
            X_opts = np.empty((n_points,
                               gpr.d))
            y_lies = np.empty(n_points)
            acq_vals = np.empty(n_points)
            # Copy the GP instance as it is modified during
            # the optimization. The GP will be reset after the
            # Acquisition is done.
            gpr_ = deepcopy(gpr)
        gpr_ = mpi.comm.bcast(gpr_ if mpi.is_main_process else None)
        n_acq_per_process = \
            mpi.split_number_for_parallel_processes(self.n_restarts_optimizer)
        n_acq_this_process = n_acq_per_process[mpi.RANK]
        i_acq_this_process = sum(n_acq_per_process[:mpi.RANK])
        proposal_X = np.empty((n_acq_this_process, gpr_.d))
        acq_X = np.empty((n_acq_this_process,))
        for ipoint in range(n_points):
            # Optimize the acquisition function to get a few possible next proposal points
            # (done in parallel)
            for i in range(n_acq_this_process):
                proposal_X[i], acq_X[i] = self.optimize_acquisition_function(
                    gpr_, i + i_acq_this_process, random_state=random_state)
            proposal_X_main, acq_X_main = mpi.multi_gather_array(
                [proposal_X, acq_X])
            # Reset the objective function, such that afterwards the correct one is used
            self.obj_func = None
            # Now take the best and add it to the gpr (done in sequence)
            if mpi.is_main_process:
                # Find out which one of these is the best
                max_pos = np.argmin(acq_X_main) if np.any(
                    np.isfinite(acq_X_main)) else len(acq_X_main) - 1
                X_opt = proposal_X_main[max_pos]
                # Transform X and clip to bounds
                if self.preprocessing_X is not None:
                    X_opt = self.preprocessing_X.inverse_transform(X_opt, copy=True)
                # Get the value of the acquisition function at the optimum value
                acq_val = -1 * acq_X_main[max_pos]
                X_opt = np.array([X_opt])
                # Get the "lie" (prediction of the GP at X)
                y_lie = gpr_.predict(X_opt)
                # Try to append the lie to change uncertainties (and thus acq func)
                # (no need to append if it's the last iteration)
                if ipoint < n_points - 1:
                    # Take the mean of errors as supposed measurement error
                    if np.iterable(gpr_.noise_level):
                        lie_noise_level = np.array(
                            [np.mean(gpr_.noise_level)])
                        # Add lie to GP
                        gpr_.append_to_data(
                            X_opt, y_lie, noise_level=lie_noise_level,
                            fit=False)
                    else:
                        # Add lie to GP
                        gpr_.append_to_data(X_opt, y_lie, fit=False)
                # Append the points found to the array
                X_opts[ipoint] = X_opt[0]
                y_lies[ipoint] = y_lie[0]
                acq_vals[ipoint] = acq_val
            # Send this new gpr_ instance to all mpi
            gpr_ = mpi.comm.bcast(gpr_ if mpi.is_main_process else None)
        gpr.n_eval = gpr_.n_eval  # gather #evals of the GP, for cost monitoring
        return mpi.comm.bcast(
            (X_opts, y_lies, acq_vals) if mpi.is_main_process else (None, None, None))

    def _constrained_optimization(self, obj_func, initial_X, bounds):

        if self.acq_optimizer == "fmin_l_bfgs_b":
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            opt_res = scipy.optimize.fmin_l_bfgs_b(
                obj_func, initial_X, args={"eval_gradient": True},
                bounds=bounds, approx_grad=False)
            theta_opt, func_min = opt_res[0], opt_res[1]
        elif self.acq_optimizer == "sampling":
            opt_res = scipy.optimize.minimize(
                obj_func, initial_X, args=(False), method="Powell",
                bounds=bounds)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.acq_optimizer):
            theta_opt, func_min = \
                self.acq_optimizer(obj_func, initial_X, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.acq_optimizer)

        return theta_opt, func_min


# DEPRECATED ON 13-09-2023:
class GPAcquisition(BatchOptimizer):

    def __init__(self, *args, **kwargs):
        print(
            "*Warning*: This class has been renamed to BatchOptimizer. "
            "This will fail in the future."
        )
        super().__init__(*args, **kwargs)


class NORA(GenericGPAcquisition):
    """
    Run Gaussian Process acquisition with NORA (Nested sampling Optimization for Ranked
    Acquistion).

    Uses kriging believer while it samples the acquisition function using nested
    sampling (with PolyChord or UltraNest).

    Parameters
    ----------
    bounds : array
        Bounds in which to optimize the acquisition function,
        assumed to be of shape (d,2) for d dimensional prior

    acq_func : GPry Acquisition Function, optional (default: "LogExp")
        Acquisition function to maximize/minimize. If none is given the
        `LogExp` acquisition function will be used

    mc_every : int
        If >1, only calls the MC sampler every `mc_steps`, and reuses previous X
        otherwise, recomputing y and sigma with the new GPR.

    random_state : int or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    zeta_scaling : float, optional (default: 1.1)
        The scaling of the acquisition function's zeta parameter with dimensionality
        (Only if "LogExp" is passed as acquisition_function)

    zeta: float, optional (default: None, uses zeta_scaling)
        Specifies the value of the zeta parameter directly.

    use_prior_sample: bool
        Whether to use the initial prior sample from Nested Sampling for the ranking.
        Can be a large number of them in high dimension. Default: False.

    nlive_per_training: int
        live points per sample in the current training set.
        Not recommended to decrease it.

    nlive_per_dim_max: int
        live points max cap (times dimension).

    num_repeats_per_dim: int
        length of slice-chains times dimension.

    precision_criterion_target: float
        Cap on precision criterion of Nested Sampling

    nprior_per_nlive: int
        Number of initial samples times dimension.

    preprocessing_X : X-preprocessor, Pipeline_X, optional (default: None)
        Single preprocessor or pipeline of preprocessors for X. Preprocessing
        makes sense if the scales along the different dimensions are vastly
        different which means that the optimizer struggles to find the maximum
        of the acquisition function. If None is passed the data is not
        preprocessed.

    verbose : 1, 2, 3, optional (default: 1)
        Level of verbosity. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise.

    Attributes
    ----------
    gpr_ : GaussianProcessRegressor
            The GP Regressor which is currently used for optimization.

    """

    def __init__(self,
                 bounds,
                 preprocessing_X=None,
                 random_state=None,
                 verbose=1,
                 acq_func="LogExp",
                 # DEPRECATED ON 13-09-2023:
                 zeta=None,
                 zeta_scaling=None,
                 # Class-specific:
                 sampler="ultranest",
                 mc_every="1d",
                 use_prior_sample=False,
                 nlive_per_training=3,
                 nlive_per_dim_max=25,
                 num_repeats_per_dim=5,
                 precision_criterion_target=0.01,
                 nprior_per_nlive=10,
                 max_ncalls=None,
                 tmpdir=None,
                 ):
        super().__init__(
            bounds=bounds, preprocessing_X=preprocessing_X, random_state=random_state,
            verbose=verbose, acq_func=acq_func, zeta=zeta, zeta_scaling=zeta_scaling)
        self.mc_every = get_Xnumber(mc_every, "d", self.n_d, int, "mc_every")
        self.mc_every_i = 0
        self.use_prior_sample = use_prior_sample
        self.tmpdir = tmpdir
        self.i = 0
        self.acq_func_y_sigma = None
        # Configure nested sampler
        self.sampler = sampler
        if self.sampler.lower() == "polychord":
            try:
                # pylint: disable=import-outside-toplevel
                from pypolychord.settings import PolyChordSettings
                from pypolychord.priors import UniformPrior
            except ImportError as excpt:
                raise ImportError(
                    "Please install PolyChord or select an alternative nested sampler "
                    "(e.g. UltraNest)."
                ) from excpt
            self.polychord_settings = PolyChordSettings(nDims=self.n_d, nDerived=0)
            # More efficient for const-eval-speed GP's (not very significant)
            self.polychord_settings.synchronous = False
            # Don't write unnecessary files: take lots of space and waste time
            self.polychord_settings.read_resume = False
            self.polychord_settings.write_resume = False
            self.polychord_settings.write_live = False
            self.polychord_settings.write_dead = True
            self.polychord_settings.write_prior = self.use_prior_sample
            self.polychord_settings.feedback = verbose - 3
            # 0: print header and result; not very useful: turn it to -1 if that's the case
            if self.polychord_settings.feedback == 0:
                self.polychord_settings.feedback = -1
            self.prior = UniformPrior(*self.bounds.T)
            self.last_polychord_output = None
        elif self.sampler.lower() == "ultranest":
            try:
                # pylint: disable=import-outside-toplevel
                import ultranest
            except ImportError as excpt:
                raise ImportError(
                    "Please install UltraNest or select an alternative nested sampler "
                    "(e.g. PolyChord)."
                ) from excpt
        elif self.sampler.lower() == "nessai":
            try:
                # pylint: disable=import-outside-toplevel
                import nessai
                import nessai.model
                from nessai.flowsampler import FlowSampler
            except ImportError as excpt:
                raise ImportError(
                    "Please install nessai or select an alternative nested sampler "
                    "(e.g. PolyChord)."
                ) from excpt
        # TODO: fix this!
        # # Using rng state as seed for PolyChord
        # if self.random_state is not None:
        #     self.polychord_settings.seed = \
        #         random_state.bit_generator.state["state"]["state"] + mpi.RANK
        # Prepare precision parameters
        self.nlive_per_training = nlive_per_training
        self.nlive_per_dim_max = nlive_per_dim_max
        self.num_repeats_per_dim = num_repeats_per_dim
        self.precision_criterion_target = precision_criterion_target
        self.nprior_per_nlive = nprior_per_nlive
        self.max_ncalls = max_ncalls
        # Pool for storing intermediate results during parallelised acquisition
        self.pool = None
        self.X_mc, self.y_mc, self.sigma_y_mc, self.acq_mc = None, None, None, None
        self.log_header = f"[ACQUISITION : {self.__class__.__name__}] "

    @property
    def pool_size(self):
        """Size of the pool of points."""
        if self.pool is None:
            return None
        return len(self.pool)

    def update_NS_precision(self, gpr):
        """
        Updates NS precision parameters:
        - num_repeats: constant for now
        - nlive: `nlive_per_training` times the size of the training set, capped at
            `nlive_per_dim_cap` (typically 25) times the dimension.
        - precision_criterion: takes a line that passes through some
            (log_max_preccrit, max_logKL) and some (log_min_preccrit, min_logKL)
            and interpolates for the exponential running mean of the logKL's
        """
        nlive = min(self.nlive_per_training * gpr.n, self.nlive_per_dim_max * self.n_d)
        return {"nlive": nlive,
                "num_repeats": self.num_repeats_per_dim * self.n_d,
                "precision_criterion": self.precision_criterion_target,
                "nprior": int(self.nprior_per_nlive * nlive),
                "max_ncalls": self.max_ncalls
        }

    def log(self, msg, level=None):
        """
        Print a message if its verbosity level is equal or lower than the given one (or
        always if ``level=None``.
        """
        if level is None or level <= self.verbose:
            print(self.log_header + msg)

    def get_MC_sample(self, gpr, random_state=None, sampler=None):
        """

        Returns
        -------
        X, y, sigma_y, weights
            May return None for any of y, sigma_y, weights
        """
        if sampler is None:
            sampler = self.sampler
        if sampler.lower() == "uniform":
            return self._get_MC_sample_uniform(gpr, random_state)
        if sampler.lower() == "polychord":
            return self._get_MC_sample_polychord(gpr, random_state)
        if sampler.lower() == "ultranest":
            return self._get_MC_sample_ultranest(gpr, random_state)
        if sampler.lower() == "nessai":
            return self._get_MC_sample_nessai(gpr, random_state)
        raise ValueError(f"Sampler '{sampler}' not known.")

    def _get_MC_sample_uniform(self, gpr, random_state):
        if not mpi.is_main_process:
            return None, None, None
        proposer = UniformProposer(self.bounds)
        n_total = 8 * gpr.d
        X = np.empty(shape=(n_total, gpr.d))
        for i in range(n_total):
            X[i] = proposer.get(random_state=random_state)
        return X, None, None, None
    
    def _get_MC_sample_nessai(self, gpr, random_state):
        # write a code to sample from nessai
        import nessai
        import nessai.model
        from nessai.flowsampler import FlowSampler
        class Nessai_model(nessai.model.Model):
            """
            Translates the GP to a nessai model
            """
            def __init__(self, gpr, bounds):
                self.gpr = gpr
                # Routine to generate tighter bounds which only capture the finite part of the GP.
                X_train_finite = self.gpr.X_train
                n_d = self.gpr.d
                self.gpry_bounds = bounds
                modified_bounds = np.empty((n_d, 2))
                for d in range(n_d):
                    modified_bounds[d, 0] = np.min(X_train_finite[:, d])
                    modified_bounds[d, 1] = np.max(X_train_finite[:, d])
                self.names = generic_params_names(n_d)
                self.modified_bounds = modified_bounds # self.gpry_model.prior.bounds(confidence_for_unbounded=0.99995)
                
                self.log_prior_volume = np.sum(np.log(self.gpry_bounds[:,1] - self.gpry_bounds[:,0]))
                
                self.bounds = {}
                for i in range(len(self.names)):
                    self.bounds[self.names[i]] = self.modified_bounds[i]
            
            def gpry_log_likelihood(self, point):
                return self.gpr.predict(point) + self.log_prior_volume

            def log_likelihood(self, livepoint):
                if livepoint.ndim == 0:
                    point = np.array([livepoint[p] for p in self.names])
                    point = np.atleast_2d(point)
                    ll = self.gpry_log_likelihood(point)
                else:
                    points = np.array([[livepoint[i][p] for p in self.names] for i in range(livepoint.size)])
                    ll = self.gpry_log_likelihood(points)
                        
                return ll

            def log_prior(self, livepoint):
                if not self.in_bounds(livepoint).any():
                    return -np.inf
                lp = np.single(0.)
                return lp
        nessai_model = Nessai_model(gpr, self.bounds)
        updated_settings = self.update_NS_precision(gpr)

        if self.tmpdir is None:
            # TODO: add to checkpoint folder?
            tmpdir = tempfile.TemporaryDirectory().name
        else:
            tmpdir = f"{self.tmpdir}/{self.i}"
            self.i += 1
        # TODO: add more options for nessai
        sampler = FlowSampler(nessai_model, output=tmpdir,
                              nlive=updated_settings['nlive'],
                              plot=False, resume=False)
        sampler.run(plot=False, save=False)
        result = sampler.ns.get_result_dictionary()
        X = result['nested_samples']
        x = sampler.posterior_samples

        # Copy the data from the structured array to the float array
        dtype_new = np.dtype({'names': x.dtype.names, 'formats': tuple([np.float64]*len(x.dtype.names))})
        x = x.astype(dtype_new)
        posterior_samples = x.view(np.float64).reshape(x.shape[0], -1)

        X = posterior_samples[:, :-3]
        y = posterior_samples[:, -2]

        return X, y, None, None
        
        



    def _get_MC_sample_polychord(self, gpr, random_state):

        # Initialise "likelihood" -- returns GPR value and deals with pooling/ranking
        def logp(X):
            """
            Returns the predicted value at a given point (-inf if prior=0).
            """
            return gpr.predict(np.array([X]), return_std=False, validate=False)[0], []

        # Update PolyChord precision settings
        new_prec = self.update_NS_precision(gpr)
        for k, v in new_prec.items():
            setattr(self.polychord_settings, k, v)
        from pypolychord import run_polychord  # pylint: disable=import-outside-toplevel
        if mpi.is_main_process:
            if self.tmpdir is None:
                # TODO: add to checkpoint folder?
                tmpdir = tempfile.TemporaryDirectory().name
            else:
                tmpdir = f"{self.tmpdir}/{self.i}"
                self.i += 1
            # ALT: persistent folder:
            # tmpdir = tempfile.mkdtemp()
            self.polychord_settings.base_dir = tmpdir
            self.polychord_settings.file_root = "test"
        mpi.share_attr(self, "polychord_settings")
        with NumpyErrorHandling(all="ignore") as _:
            self.last_polychord_output = run_polychord(
                logp,
                nDims=self.n_d, nDerived=0,
                settings=self.polychord_settings,
                prior=self.prior)
        if mpi.is_main_process:
            dummy_paramnames = [tuple(2 * [f"x_{i + 1}"]) for i in range(gpr.d)]
            self.last_polychord_output.make_paramnames_files(dummy_paramnames)
            samples_T = np.loadtxt(self.last_polychord_output.root + ".txt").T
            X = samples_T[2:].T
            y = -0.5 * samples_T[1]  # this one stores chi2
            w = samples_T[0]
            if self.use_prior_sample:
                raise ValueError("For now use_prior_sample is disabled.")
                # Old code, for reference.
                # They do not have weights, so should be stored in a different list
                # TODO: check if this improves multimodality tests!
                # prior_T = np.loadtxt(self.last_polychord_output.root + "_prior.txt").T
                # X_prior = prior_T[2:].T
                # y_prior = - prior_T[1] / 2  # this one is stored as chi2
                # X = np.concatenate([X_prior, X])
                # y = np.concatenate([y_prior, y])
            # Delete products from tmp folder
            shutil.rmtree(self.polychord_settings.base_dir)
            return X, y, None, w
        return None, None, None, None

    def _get_MC_sample_ultranest(self, gpr, random_state):
        # Ultranest cannot deal with -np.inf
        old_minus_inf_value = gpr.minus_inf_value
        gpr.minus_inf_value = -1e300
        widths = self.bounds[:, 1] - self.bounds[:, 0]
        lowers = self.bounds[:, 0]

        # Initialise "likelihood" -- returns GPR value and deals with pooling/ranking
        def logp(X):
            """
            Returns the predicted value at a given point (-inf if prior=0).
            """
            return gpr.predict(np.atleast_2d(X), return_std=False, validate=False)

        def uniform_prior_transform(quantiles):
            return quantiles * widths + lowers

        # Update precision settings
        updated_settings = self.update_NS_precision(gpr)
        tmpdir = None
        if mpi.is_main_process:
            if self.tmpdir is None:
                # TODO: add to checkpoint folder?
                tmpdir = tempfile.TemporaryDirectory().name
            else:
                tmpdir = f"{self.tmpdir}/{self.i}"
                self.i += 1
            # ALT: persistent folder:
            # tmpdir = tempfile.mkdtemp()
        tmpdir = mpi.comm.bcast(tmpdir)
        updated_settings["log_dir"] = tmpdir
        from ultranest import ReactiveNestedSampler
        sampler = ReactiveNestedSampler(
            [f"x_{i + 1}" for i in range(gpr.d)],
            logp,
            uniform_prior_transform, log_dir=updated_settings["log_dir"],
            resume="overwrite",
            vectorized=True,
        )
        with NumpyErrorHandling(all="ignore") as _:
            result = sampler.run(
                min_num_live_points=updated_settings["nlive"],
                max_ncalls=updated_settings["max_ncalls"],
                frac_remain=updated_settings["precision_criterion"],
                viz_callback=False, show_status=False,
            )
        gpr.minus_inf_value = old_minus_inf_value
        if mpi.is_main_process:
            w = result['weighted_samples']['weights']
            X = result["weighted_samples"]["points"]
            y = result["weighted_samples"]["logl"]
            if self.use_prior_sample:
                raise NotImplementedError("use_prior not implemented yet for UltraNest.")
            # Delete products from tmp folder
            shutil.rmtree(updated_settings["log_dir"])
            return X, y, None, w
        return None, None, None, None

    def multi_add(self, gpr, n_points=1, random_state=None, force_resample=False):
        r"""Method to query multiple points where the objective function
        shall be evaluated.

        The strategy which is used to query multiple points is by using
        the :math:`f(x)\sim \mu(x)` strategy and and not changing the
        hyperparameters of the model.

        It runs NS on the mean of the GP model, tracking the value
        of the acquisition function at every evaluation, and keeping a
        pool of candidates which is re-sorted whenever a new good candidate
        is found.

        When run in parallel (MPI), returns the same values for all processes.

        Parameters
        ----------
        gpr : GaussianProcessRegressor
            The GP Regressor which is used as surrogate model.

        n_points : int, optional (default=1)
            Number of points returned by the optimize method
            If the value is 1, a single point to evaluate is returned.

            Otherwise a list of points to evaluate is returned of size
            n_points. This is useful if you can evaluate your objective
            in parallel, and thus obtain more objective function evaluations
            per unit of time.

        random_state : int or numpy.RandomState, optional
            The generator used to initialize the centers. If an integer is
            given, it fixes the seed. Defaults to the global numpy random
            number generator.

        Returns
        -------
        X : numpy.ndarray, shape = (X_dim, n_points)
            The X values of the found optima
        y_lies : numpy.ndarray, shape = (n_points,)
            The predicted values of the GP at the proposed sampling locations
        fval : numpy.ndarray, shape = (n_points,)
            The values of the acquisition function at X_opt
        """
        # Check if n_points is positive and an integer
        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(f"n_points should be int > 0, got {n_points}")
        # Gather an MC sample
        if mpi.is_main_process:
            start_sample = time()
        mc_sample_this_time = not bool(self.mc_every_i % self.mc_every) or force_resample
        if mc_sample_this_time:
            self.X_mc, self.y_mc, self.sigma_y_mc, self.w_mc = self.get_MC_sample(
                gpr, random_state
            )
            # Save the necessary values for reweighting
            self.y_mc_last_sampled = np.copy(self.y_mc)
            if self.w_mc is not None:
                self.w_mc_last_sampled = np.copy(self.w_mc)
            else:
                self.w_mc_last_sampled = None
        else:  # reuse
            self.X_mc, self.y_mc, self.sigma_y_mc = self.X_mc, None, None
        self.mc_every_i += 1
        # Compute acq functions and missing quantities.
        self.acq_func_y_sigma = partial(
            self.acq_func.f, baseline=gpr.y_max,
            noise_level=gpr.noise_level, zeta=self.acq_func.zeta)
        this_X, this_y, this_sigma_y, this_acq = \
            self._compute_acq_and_missing_y_sigma(gpr=gpr)
        # Reweight the last acquired samples to the new gp
        if mpi.is_main_process:
            if not mc_sample_this_time:
                reweight_factor = np.exp(self.y_mc - self.y_mc_last_sampled)
                self.w_mc = (
                    self.w_mc_last_sampled if self.w_mc_last_sampled is not None else 1
                ) * reweight_factor
                self.w_mc /= max(self.w_mc)
        mpi.sync_processes()
        if mpi.is_main_process:
            what_we_did = ("Obtained new MC sample" if mc_sample_this_time
                           else "Re-evaluated previous MC sample")
            self.log(
                f"({(time()-start_sample):.2g} sec) {what_we_did}")
        # Rank to get best points:
        mpi.sync_processes()
        if mpi.is_main_process:
            start_rank = time()
        # TODO: still testing in realistic scenarios whether it is faster to rank in
        #       parallel (doesn't matter a lot, since it's way faster than NS anyway)
        # merged_pool = self._rank(n_points, gpr)
        merged_pool = self._parallel_rank_and_merge(
            this_X, this_y, this_sigma_y, this_acq, n_points, gpr, method="auto")
        # In case the pool is not full (not enough "good" points added), drop empty slots
        merged_pool = merged_pool.copy(drop_empty=True)
        with np.errstate(divide='ignore'):
            merged_pool_acq = self.acq_func_y_sigma(
                merged_pool.y[:n_points], merged_pool.sigma[:n_points])
        mpi.sync_processes()
        self.pool.reset_cache()  # reduces size of pickled object
        if mpi.is_main_process:
            self.log(
                f"({(time()-start_rank):.2g} sec) Ranked pool of candidates.")
        return (
            merged_pool.X[:n_points], merged_pool.y[:n_points], merged_pool_acq[:n_points]
        )

    def _compute_acq_and_missing_y_sigma(self, gpr):
        """
        Ensures a full set of `X, y, sigma_y, acq_value`, attributes starting from current
        attribute values, which are assumed equal-valued for all ranks.

        Returns scattered arrays for these values for all processes.

        Parallelises the computation if possible (returns split arrays per process).
        """
        X = mpi.comm.bcast(self.X_mc)
        # We don't use mpi.split_number_for_parallel_processes because for the ranking
        # it is good to have similar y scaling in all MPI processes. But if we use that
        # function, some processes get the top of the dist, and others the bottom, and
        # the bottom one's scaling does not perform well with the add_one method, even
        # after sorting, and the slowest MPI process sets the global speed
        this_X = X[mpi.RANK::mpi.SIZE]
        y = mpi.comm.bcast(self.y_mc)
        sigma_y = mpi.comm.bcast(self.sigma_y_mc)
        if y is None:  # assume sigma_y is also None
            if len(this_X) > 0:
                this_y, this_sigma_y = gpr.predict(
                    this_X, return_std=True, validate=False)
            else:
                this_y = np.array([], dtype=float)
                this_sigma_y = np.array([], dtype=float)
            self._undo_step_splitting(this_y, "y_mc")
            self._undo_step_splitting(this_sigma_y, "sigma_y_mc")
        elif sigma_y is None:
            this_y = y[mpi.RANK::mpi.SIZE]
            if len(this_y) > 0:
                this_sigma_y = gpr.predict_std(this_X, validate=False)
            else:
                this_sigma_y = np.array([], dtype=float)
            self._undo_step_splitting(this_sigma_y, "sigma_y_mc")
        else:  # both y and sigma_y are known
            this_y = y[mpi.RANK::mpi.SIZE]
            this_sigma_y = sigma_y[mpi.RANK::mpi.SIZE]
        with np.errstate(divide='ignore'):
            this_acq = self.acq_func_y_sigma(this_y, this_sigma_y)
        self._undo_step_splitting(this_acq, "acq_mc")
        return this_X, this_y, this_sigma_y, this_acq

    def _undo_step_splitting(self, values, attr_name):
        """
        Gather and undoes the ::mpi.SIZE process splitting and sets the corresponding
        attribute at the rank=0 process.
        """
        values_step = mpi.comm.gather(values)
        if mpi.is_main_process:
            attr_value = np.zeros(len(self.X_mc))
            for i, v in enumerate(values_step):
                attr_value[i::mpi.SIZE] = v
            setattr(self, attr_name, attr_value)

    # Non-parallel version of the ranking of the MC points
    def _rank(self, n_points, gpr):
        if mpi.is_main_process:
            self.pool = RankedPool(
                n_points, gpr=gpr, acq_func=self.acq_func_y_sigma,
                verbose=self.verbose - 3)
            with np.errstate(divide='ignore'):
                self.pool.add(
                    self.X_mc, self.y_mc, self.sigma_y_mc, self.acq_mc,
                    method="single sort acq")
        mpi.share_attr(self, "pool")
        return self.pool

    # Parallel version of the ranking of the MC points
    def _parallel_rank_and_merge(
            self, this_X, this_y, this_sigma_y, this_acq, n_points, gpr, method=None):
        # For dimensionalities 4 and smaller, bulk adding is expected to be faster.
        merge_method = None
        if method.lower() == "auto":
            method = "bulk" if gpr.d <= 4 else "single sort acq"
            merge_method = "bulk"
        # The size of the pool should be at least the amount of points to be acquired.
        # If running several processes in parallel, it can be reduced down to the number
        #   of points to be evaluated per process, but with less guarantee to find an
        #   optimal set.
        self.pool = RankedPool(
            n_points, gpr=gpr, acq_func=self.acq_func_y_sigma, verbose=self.verbose - 3)
        with np.errstate(divide='ignore'):
            self.pool.add(this_X, this_y, this_sigma_y, this_acq, method=method)
            merged_pool = self._merge_pools(n_points, gpr, method=merge_method)
        return merged_pool

    def _gather_pools(self):
        """
        Merges the points in all pools, discarding the empty ones.

        rank-0 process returns [X, y, sigma, acq], where the last two are the
        unconditioned input ones.
        """
        pool_X = mpi.comm.gather(self.pool.X[:len(self.pool)])
        pool_y = mpi.comm.gather(self.pool.y[:len(self.pool)])
        pool_sigma = mpi.comm.gather(self.pool.sigma[:len(self.pool)])
        pool_acq = mpi.comm.gather(self.pool.acq[:len(self.pool)])
        # Using the conditional acq value just to discard empty slots (acq=-inf)
        # Later discarded (not returned), since they need to be recomputed anyway.
        pool_acq_cond = mpi.comm.gather(self.pool.acq_cond[:len(self.pool)])
        if mpi.is_main_process:
            # Discard unfilled positions
            pool_acq_cond = np.concatenate(pool_acq_cond)
            i_notnan = np.isfinite(pool_acq_cond)
            pool_X = np.concatenate(pool_X)[i_notnan]
            pool_y = np.concatenate(pool_y)[i_notnan]
            pool_sigma = np.concatenate(pool_sigma)[i_notnan]
            pool_acq = np.concatenate(pool_acq)[i_notnan]
            return pool_X, pool_y, pool_sigma, pool_acq
        return None, None, None, None

    def _merge_pools(self, n_points, gpr, method=None):
        """
        Merges the pools of parallel processes to find ``n_points`` optimal locations.

        Returns all these locations for all processes.

        Acquisition values returned are *unconditional*.
        """
        if not mpi.multiple_processes:  # no need to merge and re-sort
            return self.pool
        pool_X, pool_y, pool_sigma, pool_acq = self._gather_pools()
        merged_pool = None
        if mpi.is_main_process:
            merged_pool = RankedPool(
                n_points, gpr=gpr, acq_func=self.acq_func_y_sigma,
                verbose=self.pool.verbose)
            merged_pool.add(pool_X, pool_y, pool_sigma, pool_acq, method=method)
        merged_pool = mpi.comm.bcast(merged_pool)
        return merged_pool


class RankedPool():
    """
    Keeps a ranked pool of sample proposal for Krigging-believer, given a GP regressor
    and an acquisition function.

    Parameters
    ----------
    size : int
        Number of points sampled proposals targeted.

    gpr : GaussianProcessRegressor
        The GP Regressor which is used as surrogate model.

    acq_func : callable
        Acquisition function used to rank the pool. Must be a function of ``(y, sigma)``
        only, partially evaluated its hyperparameters, if necessary.

    verbose : 1, 2, 3, optional (default: 1)
        Level of verbosity. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise.
    """

    def __init__(self, size, gpr, acq_func, verbose=1):
        self._gpr = gpr
        self._acq_func = acq_func
        self.verbose = verbose
        # The pool should have one more element than the number of desired points.
        self.X = np.zeros((size + 1, gpr.d))
        self.y = np.zeros((size + 1))
        # Condioned acquisition, used for ranking; -np.inf means "empty slot"
        self.acq_cond = np.full((size + 1), -np.inf)
        # Input quantities, only used at point, stored just for logging
        self.sigma = np.zeros((size + 1))
        self.acq = np.zeros((size + 1))
        # Cached conditioned GPR's
        self.reset_cache()
        # Counter how many models have been cached, for efficieny checks
        self.cache_counter = 0

    def __len__(self):
        return len(self.y) - 1

    @property
    def min_acq(self):
        """
        Minimum acquisition function value in order for a point to be considered, i.e.
        the conditioned acquisition function value of the last element in the pool.

        NB: while the pool is not yet fool, empty points are assigned minus infinity as
        acquisition function value, so one can still use the condition
        ``acq_value > RankedPool.min_acq`` in order to decide whether to add a point.
        """
        return self.acq_cond[len(self) - 1]

    # TODO: abstract these to a class
    def log(self, level=None, msg=""):
        """
        Print a message if its verbosity level is equal or lower than the given one (or
        always if ``level=None``.
        """
        if level is None or level <= self.verbose:
            print(msg)

    def str_point(self, X, y, sigma, acq, sigma_cond=None, acq_cond=None):
        """Retuns a standardised string to log a point."""
        sigma_cond_str = f" (cond: {sigma_cond})" if sigma_cond is not None else ""
        acq_cond_str = f" (cond: {acq_cond})" if acq_cond is not None else ""
        return f"{X}, y = {y} +/- {sigma}{sigma_cond_str}; acq = {acq}{acq_cond_str}"

    def str_pool(
            self, include_last=False, last_sorted=None, prefix=None, suffix_last=None):
        """Returns a string representation of the current pool."""
        pool_str = ""
        for i in range(len(self.X) + (-1 if not include_last else 0)):
            pool_str += (
                (prefix or "") + f"{i + 1} : " + self.str_point(
                    self.X[i], self.y[i], self.sigma[i], self.acq[i],
                    acq_cond=self.acq_cond[i]
                ) + (" [last sorted]" if i == last_sorted else "") + "\n"
            )
        return pool_str.rstrip("\n") + (
            f" {suffix_last}" if include_last and suffix_last else "")

    def log_pool(
            self, level=4, include_last=False, last_sorted=None, prefix=None,
            suffix_last=None):
        """Prints the current pool."""
        if self.verbose >= level:
            self.log(level=level, msg=self.str_pool(
                include_last=include_last, last_sorted=last_sorted, prefix=prefix,
                suffix_last=suffix_last))

    def __str__(self):
        return self.str_pool(include_last=False)

    def add(self, X, y=None, sigma=None, acq=None, method="single sort acq"):
        """
        Adds points to the pool.

        Parameters
        ----------
        X: np.ndarray (1- or 2-dimensional)
            Position of the proposed sample.

        y: np.ndarray (1 dimension fewer than X) or float, optional
            Predicted value under the GPR.

        sigma: np.ndarray (1 dimension fewer than X) or float, optional
            Predicted standard deviation under the GPR. Will be computed if not passed.

        acq: np.ndarray (1 dimension fewer than X) or float, optional
            Acquisition function values (unconditioned). Will be computed if not passed.

        method: {"single", "single sort acq", "single sort y", "bulk"}
            Uses the one-by-one algorithm ("single", with pre-sorting accorting to X if
            "single sort X"), or the bulk algorithm.
        """
        if len(X.shape) < 2:
            self.add(X, y, sigma, method=method)
            return
        if X.shape[1] == 1:
            self.add(
                X[0], y[0] if y else None, sigma[0] if sigma else None, method=method)
            return
        if y is None:
            y, sigma = self._gpr.predict(X, return_std=True, validate=False)
        elif sigma is None:
            sigma = self._gpr.predict_std(X, validate=False)
        if acq is None:
            acq = self._acq_func(y, sigma)
        if method.lower() == "bulk":
            self.add_bulk(X, y, sigma, acq)
        elif method.lower().startswith("single"):
            i_sort = None
            if "sort" in method.lower():
                i_sort = np.argsort(
                    {"acq": acq, "y": y}[method.lower().split()[-1]])[::-1]
            # Descending order of unconditioned acq or unconditional mean prediction:
            # minimizes the number of swaps: model caches + calculation of acq_cond
            for i in (i_sort if i_sort is not None else range(len(X))):
                self.add_one(X[i], y[i], sigma[i], acq[i])
        else:
            ValueError(f"Algorithm '{method}' not known.")

    def add_bulk(self, X, y, sigma, acq, i_start=0):
        """
        Tries to fill the pull using a batch of points at once:

        1. Compute their acquisition value conditioned to the position above (if any).
        2. Pick the best and delete infinities (acq cannot grow with more conditioning).
        3. Place it in the current position, and do a recursive call for the next one.

        The advantage of this method with respect to ``add_one`` is that it can use
        vectorization to compute the std's, but on the other hand it needs to compute
        many more of them, so it will be better only up to some dimension and some amount
        of training, and then ``add_one`` will take over.
        """
        # Compute acq using the model just above (and cache it if needed)
        if i_start == 0:  # no need to condition
            gpr = self._gpr
            acq_cond = acq if isinstance(acq, np.ndarray) else np.array(acq)
        else:
            gpr = self.cache_model(i_start - 1)
            sigma_cond = gpr.predict_std(X, validate=False)
            acq_cond = self._acq_func(y, sigma_cond)
        if acq_cond.size == 0:
            self.log(
                level=4,
                msg=f"No finite acq points to fill the pool from [{i_start}] down."
            )
            return
        # Find best
        i_max = np.argmax(acq_cond)
        acq_cond_max = acq_cond[i_max]
        if acq_cond_max == np.inf:
            self.log(
                level=4,
                msg=f"No finite acq points to fill the pool from [{i_start}] down."
            )
            return
        self.X[i_start] = X[i_max]
        self.y[i_start] = y[i_max]
        self.sigma[i_start] = sigma[i_max]
        self.acq[i_start] = acq[i_max]
        self.acq_cond[i_start] = acq_cond_max
        if i_start == len(self) - 1:
            # Last position just filled. We are done.
            return
        # Remove infinities (cond acq will cannot grow when conditioning even more points)
        i_finite_acq_cond = np.logical_not(acq_cond == -np.inf)
        i_finite_acq_cond[i_max] = False  # also remove point just added
        self.add_bulk(
            X[i_finite_acq_cond],
            y[i_finite_acq_cond],
            sigma[i_finite_acq_cond],
            acq[i_finite_acq_cond],
            i_start=i_start + 1,
        )

    def add_one(self, X, y=None, sigma=None, acq=None, acq_nan_is_null=False):
        """
        Tries to add one point to the pool:

        1. Computes its acquisition function value.
           (If the pools is not full, just adds it and re-sorts.)

        2. Finds the provisional position of the point in the list, without KB.
           Notice that once KB is taken into account, the KB-ranked position can only be
           lower. (If the acq. fun. value is lower than the last point, it is discarded.)

        3. Updates its aquisition function value KB-informed by the points above, finds
           the new provisional position, and repeats until the position stabilises.

        4. Sorts the list of points below the new one, recursively applying KB.

        Parameters
        ----------
        X: np.ndarray with 1 dimension
            Position of the proposed sample.

        y: float, optional
            Predicted value under the GPR.

        sigma: float, optional
            Predicted standard deviation under the GPR.

        acq: float, optional
            Value of the acquisition function.

        acq_nan_is_null: bool, optional (default: False)
            Whether NaN's in the acquisition function should be interpreted as null value.

        Raises
        ------
        ValueError: if invalid acquisition function value, unless ``acq_nan_is_null=True``.
        """
        # Discard the point as early as possible!
        # NB: The equals sign below takes care of the case in which we are trying to add a
        # point with -inf acq. func. to a pool which is not full (min acq. func. = -inf)
        if acq <= self.min_acq:
            self.log(level=4, msg="[pool.add] Acq. func. value too small. Ignoring.")
            return
        X = np.atleast_2d(X)
        if y is None:  # assume sigma is also None
            y, sigma = self._gpr.predict(X, return_std=True, validate=False)
            y, sigma = y[0], sigma[0]
        elif not hasattr(y, "__len__"):
            y = np.array([y])
        if sigma is None:
            sigma = self._gpr.predict_std(X, validate=False)
        if acq is None:
            acq = self._acq_func(y, sigma)
        if self.verbose >= 4:
            self.log(
                level=4, msg=("[pool.add] Checking point " +
                              self.str_point(X, y, sigma, acq)))
        # Repeat the min acq test above to leave asap
        if acq <= self.min_acq:
            self.log(level=4, msg="[pool.add] Acq. func. value too small. Ignoring.")
            return
        if np.isnan(acq):
            if not acq_nan_is_null:
                raise ValueError(f"Acquisition function value not a number: {acq}")
            acq = -np.inf
        self.log(level=4, msg="[pool.add] Initial pool:")
        self.log_pool(level=4, prefix="[pool.add] ")
        # Find its position in the list, conditioned to those on top
        # Shortcut: start from the bottom, ignore last element (just a placeholder)
        i_new_last = len(self)
        acq_cond = deepcopy(acq)
        sigma_cond = sigma
        while True:
            try:
                # Notice that we start always from the bottom of the list, in case a point
                # with high acq. func. value is close to one of the top points, and its
                # conditioned acq. func. value drops to a small value after conditioning.
                # The equals sign below prevents -inf's from climbing up the list.
                i_new = (len(self) -
                         next(i for i in range(len(self))
                              if self.acq_cond[-(i + 2)] >= acq_cond))
            except StopIteration:  # top of the list reached
                i_new = 0
            self.log(level=4, msg=f"[pool.add] Provisional position: [{i_new + 1}]")
            # top, same as last or last: final ranking reached
            if i_new in [0, i_new_last, len(self)]:
                break
            # Otherwise, compute conditioned acquisition value, using point above,
            # and continue to the next iteration to re-rank
            sigma_cond = self.gpr_cond[i_new - 1].predict_std(X, validate=False)[0]
            # New acquisition should not be higher than the old one, since the new one
            # corresponds to a model with more training points (though fake ones).
            # This may happen anyway bc numerical errors, e.g. when the correlation
            # length is really huge. Also, when alpha or the noise level for two cached
            # models are different. We can just ignore it.
            # (Sometimes, it's just ~1e6 relative differences, which is not worrying)
            acq_cond = min(acq_cond, self._acq_func(y, sigma_cond))
            i_new_last = i_new
            if self.verbose >= 4:  # avoid creating the f-strings
                self.log(level=4,
                         msg=f"[pool.add] Updated conditional std: {sigma_cond}")
                self.log(level=4,
                         msg=f"[pool.add] Updated conditional acquisition: {acq_cond}")
        # The last position is just a place-holder: don't save it if it falls there.
        if i_new >= len(self):
            self.log(level=4, msg="[pool.add] Discarded!")
            return
        self.log(level=4, msg=f"[pool.add] Final position: [{i_new + 1}] of {len(self)}")
        # Insert the new one in its place, and push the rest down one place.
        # We track the conditioned acq. value (but not the sigma), to retain the
        # information about whether each slot was empty (acq = -inf)
        # (but not for the acq values, which will be updated below)
        for pool, value in [(self.X, X), (self.y, y),
                            (self.sigma, sigma), (self.acq, acq),
                            (self.acq_cond, acq_cond)]:
            pool[i_new + 1:] = pool[i_new:-1]
            pool[i_new] = value
        # If not in the last position, we can safely assume that it has finite value,
        # since -inf's from conditional acq cannot climb.
        assert self.acq_cond[i_new] > -np.inf
        self.log(level=4, msg="[pool.add] Current unsorted pool:")
        self.log_pool(level=4, include_last=True, last_sorted=i_new, prefix="[pool.add] ")
        # Sort the sublist below the new element
        self.sort(i_new + 1)
        self.log(level=4, msg="[pool.add] The new pool, sorted:")
        self.log_pool(level=4, include_last=True, prefix="[pool.add] ",
                      suffix_last="[unused]")
        # Make sure that the last slot (buffer) is marked as empty
        self.acq_cond[-1] == -np.inf

    def cache_model(self, i):
        """
        Cache the GP model that contains the training set plus the pool points up to
        position ``i`` (0-based), with predicted dummy y, keeping the GPR hyperparameters
        unchanged.

        Stores and returns the conditioned gpr (or the original one if ``i=-1``).
        """
        # This function accounts for ~50% of the ranking time in add_one() (the rest is
        # mostly predict_std()).
        # Taking dim=8 as reference, deepcopy is ~1/3 and append_to_data ~2/3 of the cost.
        # Possible optimization strategies:
        # - Disable SVM in cached models (no need to copy, fit, or evaluate), assuming all
        #   passed points are finite [tested to improve <10% overall in add_one{}]
        # - Copy model above instead of original one, and fit to single new point only
        #   [tested: no appreciable gain]
        # - Keep a single cached model, adding points to it when going down the list.
        #   If there are very few inversions, we save the cost of copying [potentially
        #   ~30% this function, 15% overall in add_one()]
        # - Disable the copying of stuff not needed to compute std.
        # - At append_to_data, compute only what is strictly needed for std (I think the
        #   kernel gradient is the only such thing.
        # - Create model anew with fixed given kernel and no SVM, and fit all points.
        # In any case, the cost is now very low compared to nested sampling and hyperparam
        # fitting.
        if i < 0:
            return self._gpr
        self.log(level=4, msg=f"[pool.cache] Caching model [{i + 1}]")
        self.gpr_cond[i] = deepcopy(self._gpr)
        self.gpr_cond[i].append_to_data(self.X[:i + 1], self.y[:i + 1], fit=False)
        self.cache_counter += 1
        return self.gpr_cond[i]

    def reset_cache(self):
        """
        Deletes the cached GPR models when there are not needed any more.
        """
        # No need to store the last element in the list (or the last buffer slot)
        self.gpr_cond = [None] * len(self.X - 1)

    def __getstate__(self):
        return deepcopy(self).__dict__

    # Remove references to external objects (gpr and acq_func) and cached gpr's
    def __deepcopy__(self, memo=None):
        attrs_ignore_at_copy = ["_gpr", "_acq_func", "gpr_cond"]
        new = self.__class__.__new__(self.__class__)
        new.__dict__ = {k: deepcopy(v) for k, v in self.__dict__.items()
                        if k not in attrs_ignore_at_copy}
        return new

    def copy(self, drop_empty=False):
        """
        Returns a copy of the pool, missing references to external objects.

        If ``drop_empty=True`` (default: ``False``), the returned copy has its size
        reduced to contain just the final set of finite conditioned acquisition points.
        """
        copy_ = deepcopy(self)
        if drop_empty:
            try:
                i_first_empty = next(
                    i for i, acq in enumerate(copy_.acq_cond[:-1]) if acq == -np.inf
                )
            except StopIteration:
                return copy_
            copy_.X = copy_.X[:i_first_empty]
            copy_.y = copy_.y[:i_first_empty]
            copy_.acq_cond = copy_.acq_cond[:i_first_empty]
            copy_.sigma = copy_.sigma[:i_first_empty]
            copy_.acq = copy_.acq[:i_first_empty]
            return copy_
        return copy_

    def sort(self, i_start=0):
        """
        Sorts in descending order of acquisition function value, where the acq of the
        ``i``-th element (0-based) is conditioned on the GPR model that includes the
        points with j<i with their predicted (mean) y.

        If ``i_start!=0`` is given, assumes the upper elements in the list are already
        sorted following this criterion.

        This function assumes that the augmented model just abobe ``i_start`` has not
        already been cached, and starts by caching it and computing acquisition function
        values ``i_start`` down.
        """
        # If beyond last position, nothing to do (the = case is the last buffer slot):
        if i_start >= len(self):
            self.log(
                level=4,
                msg=f"[pool.sort] Nothing to do (sorting beyond last position).")
            return
        self.log(
            level=4,
            msg=f"[pool.sort] Sorting the pool starting at [{i_start + 1}]",
        )
        upper_gpr_cond = self.cache_model(i_start - 1)
        # If list not full (first sublist element's acq_cond=-inf), do nothing
        if self.acq_cond[i_start] == -np.inf:
            self.log(level=4, msg=f"[pool.sort] Nothing to do (list is not full yet).")
            return
        # Compute acq_cond for non-empty points (acq != -inf) only. Assume always sorted.
        try:
            i_1st_inf = next(i for i, ac in enumerate(self.acq_cond) if ac == -np.inf)
        except StopIteration:
            i_1st_inf = len(self) + 1
        sigma_cond = upper_gpr_cond.predict_std(
            self.X[i_start:i_1st_inf], validate=False)
        # Cond acq cannot be higher than less cond one. This clipping takes care of
        # numerical noise that may make it higher. In particular, keeps -inf if it was so.
        acq_cond = np.clip(
            self._acq_func(self.y[i_start:i_1st_inf], sigma_cond),
            None, np.inf if i_start == 0 else self.acq_cond[i_start - 1]
        )
        if self.verbose >= 4:  # avoid creating the f-strings
            self.log(level=4, msg=f"[pool.sort] New conditioned std: {sigma_cond}")
            self.log(level=4, msg=f"[pool.sort] New conditioned acq: {acq_cond}")
        j_sort = np.argsort(-acq_cond)  # descending order! -- This is a *sub*list index!
        acq_cond_max = acq_cond[j_sort[0]]
        # If the max found was -inf, no need to re-sort points: disable all and return
        if acq_cond_max == -np.inf:
            self.log(
                level=4,
                msg=f"[pool.sort] Nothing to do (all sublist elements have -inf acq)."
            )
            self.acq_cond[i_start:i_1st_inf] = -np.inf
            return
        self.log(level=4, msg=(
            f"[pool.sort] New max acq_cond = {acq_cond_max} "
            f"at position [{i_start + j_sort[0] + 1}]")
        )
        # Reorder (not enough to swap the max here, because new -inf may have been
        # generated, and we need to push them too the bottom.
        i_sort_partial = i_start + j_sort
        self.X[i_start:i_1st_inf] = self.X[i_sort_partial]
        self.y[i_start:i_1st_inf] = self.y[i_sort_partial]
        self.sigma[i_start:i_1st_inf] = self.sigma[i_sort_partial]
        self.acq[i_start:i_1st_inf] = self.acq[i_sort_partial]
        # Strictly, we only need to assign the highest acq_cond, since the rest will be
        # discarded, but logging is cleaner if all assigned sorted
        self.acq_cond[i_start:i_1st_inf] = acq_cond[j_sort]
        self.log(level=4, msg=f"[pool.sort] Partial sort up to {i_start + 1}:")
        self.log_pool(
            level=4, include_last=True, last_sorted=i_start, prefix="[pool.sort] ")
        # Sort the sublist below (if empty or single element, taken care of inside)
        self.sort(i_start + 1)
