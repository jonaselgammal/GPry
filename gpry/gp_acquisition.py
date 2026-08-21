"""
This module implements tools for active sampling with Gaussian Process surrogate models.

It implements two acquisition *engines*: a *classic* one based on optimizint the
acquisition function (:class:`BatchOptimizer`), and a more exploratory one based on
a nested sampling exploration of the mean of the GP Regressor (:class:`NORA`).

All classes are MPI-enabled, and can produce multiple optima where to evaluate the true
log-posterior in parallel.
"""

import os
import sys
import warnings
import inspect
import tempfile
from time import time
from copy import deepcopy
from typing import Mapping
from functools import partial
from warnings import warn

import numpy as np
import scipy.optimize  # type: ignore

import gpry.acquisition_functions as gpryacqfuncs
from gpry.proposal import PartialProposer, CentroidsProposer, Proposer, UniformProposer
from gpry import mpi
from gpry.tools import (
    NumpyErrorHandling,
    get_Xnumber,
    remove_0_weight_samples,
    is_in_bounds,
)
import gpry.ns_interfaces as nsint
from gpry.mc import samples_dict_to_getdist, _name_logp, _name_loglike, _name_logprior


def builtin_names():
    """
    Lists all names of all built-in acquisition functions criteria.
    """
    list_names = [
        name
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if (
            issubclass(obj.__class__, GenericGPAcquisition.__class__)
            and obj is not GenericGPAcquisition
        )
    ]
    return list_names


class GPAcquisitionError(Exception):
    """
    Exception to be raised whenever there is an error finding new acquisition optima.
    """


class GenericGPAcquisition:
    """Generic class for acquisition objects."""

    def __init__(
        self,
        bounds,
        preprocessing_X=None,
        verbose=1,
        acq_func="LogExp",
    ):
        self.bounds_ = np.array(bounds).copy()
        self.n_d = bounds.shape[0]
        self.preprocessing_X = preprocessing_X
        self.verbose = verbose if verbose is not None else 3
        if gpryacqfuncs.is_acquisition_function(acq_func):
            self.acq_func = acq_func
        elif isinstance(acq_func, (Mapping, str)):
            if isinstance(acq_func, str):
                acq_func = {acq_func: {}}
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

    def __call__(self, X, surrogate, eval_gradient=False, validate=True):
        """Returns the value of the acquision function at ``X`` given a ``surrogate``."""
        return self.acq_func(
            X, surrogate, eval_gradient=eval_gradient, validate=validate
        )

    def multi_add(self, surrogate, n_points=1, bounds=None, rng=None):
        r"""Method to query multiple points where the objective function
        shall be evaluated.

        The strategy differs depending on the acquisition class.

        When run in parallel (MPI), it must return the same values for all processes.

        Parameters
        ----------
        surrogate : SurrogateModel
            The GP Regressor which is used as surrogate model.

        n_points : int, optional (default=1)
            Number of points to be returned. A value large than 1 is useful if you can
            evaluate your objective in parallel, and thus obtain more objective function
            evaluations per unit of time.

        bounds : np.array, optional
            Bounds inside which to look for the next proposals, e.g. the surrogate model's
            trust region. If not defined, the prior bounds are used.

        rng : int or numpy.random.Generator, optional
            The generator used to perform the acquisition process. If an integer is given,
            it is used as a seed for the default global numpy random number generator.

        Returns
        -------
        X : numpy.ndarray, shape = (X_dim, n_points)
            The X values of the found optima
        y_lies : numpy.ndarray, shape = (n_points,)
            The predicted values of the GP at the proposed sampling locations
        fval : numpy.ndarray, shape = (n_points,)
            The values of the acquisition function at X_opt
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

    acq_func : GPry Acquisition Function, dict, optional (default: "LogExp")
        Acquisition function to maximize/minimize. If none is given the
        `LogExp` acquisition function will be used. Can also be a dictionary with the name
        of the acquisition function as the single key, and as value a dict of its
        arguments.

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

    verbose : 1, 2, 3, optional (default: 1)
        Level of verbosity. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise.

    Attributes
    ----------
    surrogate_ : GaussianProcessRegressor
            The GP Regressor which is currently used for optimization.

    """

    def __init__(
        self,
        bounds,
        preprocessing_X=None,
        verbose=1,
        acq_func="LogExp",
        # Class-specific:
        proposer=None,
        acq_optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer="5d",
        n_repeats_propose=10,
    ):
        super().__init__(
            bounds=bounds,
            preprocessing_X=preprocessing_X,
            verbose=verbose,
            acq_func=acq_func,
        )
        self.proposer = proposer
        self.obj_func = None

        # If nothing is provided for the proposal, we use a centroids proposer with
        # a fraction of uniform samples.
        if self.proposer is None:
            self.proposer = PartialProposer(
                self.bounds_, CentroidsProposer(self.bounds_)
            )
        else:
            if not isinstance(proposer, Proposer):
                raise TypeError(
                    "'proposer' must be a Proposer instance. "
                    f"Got {proposer} of type {type(proposer)}."
                )
            self.proposer = proposer
            self.proposer.update_bounds(self.bounds_)

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
                        "to return gradients. Got %s" % self.acq_func
                    )
                self.acq_optimizer = "fmin_l_bfgs_b"
            elif acq_optimizer == "sampling":
                self.acq_optimizer = "sampling"
            else:
                raise ValueError(
                    "Supported internal optimizers are 'auto', "
                    "'lbfgs' or 'sampling', "
                    "got {0}".format(acq_optimizer)
                )
        else:
            self.acq_optimizer = acq_optimizer
        self.n_restarts_optimizer = get_Xnumber(
            n_restarts_optimizer, "d", self.n_d, int, "n_restarts_optimizer"
        )
        self.n_repeats_propose = n_repeats_propose
        self.mean_ = None
        self.cov = None

    def optimize_acquisition_function(self, surrogate, i, bounds=None, rng=None):
        """Exposes the optimization method for the acquisition function. When
        called it proposes a single point where for where to evaluate the true
        model next. It is internally called in the :meth:`multi_add` method.

        Parameters
        ----------
        surrogate : SurrogateModel
            The GP Regressor which is used as surrogate model.

        i : int
            Internal counter which is used to enable MPI support. If you want
            to optimize from a single location and rerun the optimizer from
            multiple starting locations loop over this parameter.

        rng : numpy.random.Generator, optional
            The generator used for the optimization process.

        Returns
        -------
        X_opt : numpy.ndarray, shape = (X_dim,)
            The X value of the found optimum
        func : float
            The value of the acquisition function at X_opt
        """
        # Update proposer with new surrogate model and new bounds
        self.proposer.update(surrogate)
        use_bounds = self.bounds_ if bounds is None else bounds
        self.proposer.update_bounds(use_bounds)

        # If we do a first-time run, use this
        if not self.obj_func:
            # Check whether the GP has been fit to data before
            if not surrogate.fitted:
                raise AttributeError(
                    "The model which is given has not been fed "
                    "any points. Please make sure, that the model already "
                    "contains data when trying to optimize an acquisition "
                    "function on it as optimizing priors is not supported yet."
                )

            def obj_func(X, eval_gradient=False):
                # TODO: optionally suppress this checks if called by optimiser
                # Check inputs
                X = np.asarray(X)
                X = np.expand_dims(X, axis=0)
                if X.ndim != 2:
                    raise ValueError(
                        "X is {}-dimensional, however, "
                        "it must be 2-dimensional.".format(X.ndim)
                    )
                if self.preprocessing_X is not None:
                    X = self.preprocessing_X.inverse_transform(X)

                if eval_gradient:
                    acq, grad = self.acq_func(
                        X, surrogate, eval_gradient=True, validate=False
                    )
                    return -1 * acq, -1 * grad
                else:
                    return -1 * self.acq_func(
                        X, surrogate, eval_gradient=False, validate=False
                    )

            self.obj_func = obj_func

        # Preprocessing
        if self.preprocessing_X is not None:
            transformed_bounds = self.preprocessing_X.transform_bounds(use_bounds)
        else:
            transformed_bounds = use_bounds

        if i == 0:
            # Perform first run from last (in-bounds) training point.
            # Cannot raise StopIteration if trust_region contains at least one point!
            x0 = next(
                X
                for X in surrogate.X_regress[::-1]
                if np.all(is_in_bounds([X], bounds, validate=False))
            )
            if self.preprocessing_X is not None:
                x0 = self.preprocessing_X.transform(x0)
            return self._constrained_optimization(self.obj_func, x0, transformed_bounds)
        else:
            d = self.bounds_.shape[0]
            n_tries = 10 * d * self.n_restarts_optimizer
            x0s = np.empty((self.n_repeats_propose + 1, d))
            values = np.empty(self.n_repeats_propose + 1)
            ifull = 0
            for n_try in range(n_tries):
                x0 = self.proposer.get(rng=rng)
                value = self.acq_func(x0, surrogate, validate=False)
                if not np.isfinite(value):
                    continue
                x0s[ifull] = x0
                values[ifull] = value
                ifull += 1
                if ifull > self.n_repeats_propose:
                    x0 = x0s[np.argmax(values)]
                    if self.preprocessing_X is not None:
                        x0 = self.preprocessing_X.transform(x0)
                    return self._constrained_optimization(
                        self.obj_func, x0, transformed_bounds
                    )
            # if there's at least one finite value try optimizing from
            # there, otherwise take the last x0 and add that to the GP
            if ifull > 0:
                x0 = x0s[np.argmax(values[:ifull])]
                if self.preprocessing_X is not None:
                    x0 = self.preprocessing_X.transform(x0)
                return self._constrained_optimization(
                    self.obj_func, x0, transformed_bounds
                )
            else:
                if self.verbose > 1:
                    print(
                        f"of {n_tries} initial samples for the "
                        "acquisition optimizer none returned a "
                        "finite value"
                    )
                if self.preprocessing_X is not None:
                    x0 = self.preprocessing_X.transform(x0)
                return x0, -1 * value

    def multi_add(
        self, surrogate, n_points=1, bounds=None, rng=None, force_resample=False
    ):
        r"""Method to query multiple points where the objective function
        shall be evaluated. The strategy which is used to query multiple
        points is by using the :math:`f(x)\sim \mu(x)` strategy and and not
        changing the hyperparameters of the model.

        This is done to increase speed since then the blockwise matrix
        inversion lemma can be used to invert the K matrix. The optimization
        for a single point is done using the :meth:`optimize_acquisition_func` method.

        When run in parallel (MPI), returns the same values for all processes.

        Parameters
        ----------
        surrogate : SurrogateModel
            The GP Regressor which is used as surrogate model.

        n_points : int, optional (default=1)
            Number of points to be returned. A value large than 1 is useful if you can
            evaluate your objective in parallel, and thus obtain more objective function
            evaluations per unit of time.

        bounds : np.array, optional
            Bounds inside which to look for the next proposals, e.g. the surrogate models'
            trust region. If not defined, the prior bounds are used.

        rng : int or numpy.random.Generator, optional
            The generator used to perform the acquisition process. If an integer is given,
            it is used as a seed for the default global numpy random number generator.

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
        # Create (parallel) generator(s) if int passed as rng
        rng = mpi.get_random_generator(rng)
        use_bounds = self.bounds_ if bounds is None else bounds
        if mpi.is_main_process:
            # Initialize arrays for storing the optimized points
            X_opts = np.empty((n_points, surrogate.d))
            y_lies = np.empty(n_points)
            acq_vals = np.empty(n_points)
            # Copy the GP instance as it is modified during
            # the optimization. The GP will be reset after the
            # Acquisition is done.
            surrogate_ = deepcopy(surrogate)
        surrogate_ = mpi.bcast(surrogate_ if mpi.is_main_process else None)
        n_acq_per_process = mpi.split_number_for_parallel_processes(
            self.n_restarts_optimizer
        )
        n_acq_this_process = n_acq_per_process[mpi.RANK]
        i_acq_this_process = sum(n_acq_per_process[: mpi.RANK])
        proposal_X = np.empty((n_acq_this_process, surrogate_.d))
        acq_X = np.empty((n_acq_this_process,))
        for ipoint in range(n_points):
            # Optimize the acquisition function to get a few possible next proposal points
            # (done in parallel)
            for i in range(n_acq_this_process):
                proposal_X[i], acq_X[i] = self.optimize_acquisition_function(
                    surrogate_, i + i_acq_this_process, bounds=use_bounds, rng=rng
                )
            proposal_X_main, acq_X_main = mpi.multi_gather_array([proposal_X, acq_X])
            # Reset the objective function, such that afterwards the correct one is used
            self.obj_func = None
            # Now take the best and add it to the surrogate model (done in sequence)
            if mpi.is_main_process:
                # Find out which one of these is the best
                max_pos = (
                    np.argmin(acq_X_main)
                    if np.any(np.isfinite(acq_X_main))
                    else len(acq_X_main) - 1
                )
                X_opt = proposal_X_main[max_pos]
                # Transform X and clip to bounds
                if self.preprocessing_X is not None:
                    X_opt = self.preprocessing_X.inverse_transform(X_opt)
                # Get the value of the acquisition function at the optimum value
                acq_val = -1 * acq_X_main[max_pos]
                X_opt = np.array([X_opt])
                # Get the "lie" (prediction of the GP at X)
                y_lie = surrogate_.predict(X_opt, validate=False)
                # Try to append the lie to change uncertainties (and thus acq func)
                # (no need to append if it's the last iteration)
                if ipoint < n_points - 1:
                    # Take the mean of errors as supposed measurement error
                    lie_noise_level = (
                        np.array([np.mean(surrogate_.noise_level)])
                        if np.iterable(surrogate_.noise_level)
                        else None
                    )
                    # Add lie to GP
                    surrogate_.append(
                        X_opt,
                        y_lie,
                        noise_level=lie_noise_level,
                        fit_gpr=False,
                        fit_classifier=False,
                    )
                # Append the points found to the array
                X_opts[ipoint] = X_opt[0]
                y_lies[ipoint] = y_lie[0]
                acq_vals[ipoint] = acq_val
            # Send this new surrogate_ instance to all mpi
            surrogate_ = mpi.bcast(surrogate_ if mpi.is_main_process else None)
        # gather #evals of the GP, for cost monitoring
        surrogate.n_eval = surrogate_.n_eval
        return mpi.bcast(
            (X_opts, y_lies, acq_vals) if mpi.is_main_process else (None, None, None)
        )

    def _constrained_optimization(self, obj_func, initial_X, bounds):
        if self.acq_optimizer == "fmin_l_bfgs_b":
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            opt_res = scipy.optimize.fmin_l_bfgs_b(
                obj_func,
                initial_X,
                args={"eval_gradient": True},
                bounds=bounds,
                approx_grad=False,
            )
            theta_opt, func_min = opt_res[0], opt_res[1]
        elif self.acq_optimizer == "sampling":
            opt_res = scipy.optimize.minimize(
                obj_func, initial_X, args=(False), method="Powell", bounds=bounds
            )
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.acq_optimizer):
            theta_opt, func_min = self.acq_optimizer(obj_func, initial_X, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.acq_optimizer)

        return theta_opt, func_min


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

    acq_func : GPry Acquisition Function, dict, optional (default: "LogExp")
        Acquisition function to maximize/minimize. If none is given the
        `LogExp` acquisition function will be used. Can also be a dictionary with the name
        of the acquisition function as the single key, and as value a dict of its
        arguments.

    sampler : str, optional (default: None)
        Which nested sampler to use for exploration: ``polychord``, ``ultranest``,
        ``nessai``. If left unspecified, PolyChord will be used if available, otherwise
        UltraNest. For uniform sampling within prior bounds, for tests, use ``uniform``.

    mc_every : int
        If >1, only calls the MC sampler every `mc_steps`, and reuses previous X
        otherwise, recomputing ``y`` and ``sigma`` with the new surrogate model.
        ``d``-dimensional notation can be used (e.g. ``2d`` for once every
        twice-the-dimensionality iterations).

    nlive_per_training : int
        live points per sample in the current training set.
        Not recommended to decrease it. ``d``-dimensional notation can be used.

    nlive_max : int
        live points max cap. ``d``-dimensional notation can be used.

    num_repeats : int
        length of slice-chains for the PolyChord sampler. ``d``-dimensional notation can
        be used.

    precision_criterion_target : float
        Cap on precision criterion of Nested Sampling.

    nprior_per_nlive : int
        Number of initial samples times dimension. ``d``-dimensional notation can be used.

    max_ncalls : int, optional (default: None)
        Total number of calls of the surrogate model allowed during the sampler run.
        ``d``-dimensional notation can be used.

    rank_method : str (default: "single sort acq")
        Ranking method used on the MC samples to extract candidates. For possible values
        check the documentation of :class:`gq_acquisition.RankedPool`.

    rank_merge_method : str (default: "bulk")
        When running more than one MPI processes, ranking method used to merge the
        individual ranked pools. For possible values check the documentation of
        :class:`gq_acquisition.RankedPool`.

    tmpdir : str, optional (default: None)
        Folder where to store the results of the sampler chains, if they want to be kept
        for further tests. If unspecified, a temporary folder, managed by the OS, is used.

    Attributes
    ----------
    surrogate_ : SurrogateModel
            The GP Regressor which is currently used for optimization.

    """

    def __init__(
        self,
        bounds,
        preprocessing_X=None,
        verbose=1,
        acq_func="LogExp",
        # Class-specific:
        sampler=None,
        mc_every="2",
        nlive_per_training=3,
        nlive_max="25d",
        nlive_per_dim_max=None,  # deprecated
        num_repeats="5d",
        num_repeats_per_dim=None,  # deprecated
        precision_criterion_target=0.01,
        nprior_per_nlive=10,
        max_ncalls=None,
        rank_method="single sort acq",
        rank_merge_method="bulk",
        tmpdir=None,
    ):
        super().__init__(
            bounds=bounds,
            preprocessing_X=preprocessing_X,
            verbose=verbose,
            acq_func=acq_func,
        )
        self.log_header = f"[ACQUISITION : {self.__class__.__name__}] "
        self.mc_every = get_Xnumber(mc_every, "d", self.n_d, int, "mc_every")
        self.mc_every_i = 0
        self.tmpdir = tmpdir
        self.i = 0
        self.acq_func_y_sigma = None
        # Configure nested sampler
        self.sampler = sampler
        self._init_nested_sampler()
        self.nlive_per_training = nlive_per_training
        if nlive_per_dim_max is not None:
            self.log(
                "*Warning*: 'nlive_per_dim_max' is deprecated. Use e.g. 'nlive_max: 25d'."
                " This will fail in the future."
            )
            self.nlive_max = nlive_per_dim_max * self.n_d
        else:
            self.nlive_max = get_Xnumber(nlive_max, "d", self.n_d, int, "nlive_max")
        if num_repeats_per_dim is not None:
            self.log(
                "*Warning: 'num_repeats_per_dim' is deprecated. Use e.g. "
                "'num_repeats: 5d'. This will fail in the future."
            )
            self.num_repeats = num_repeats_per_dim * self.n_d
        else:
            self.num_repeats = get_Xnumber(
                num_repeats, "d", self.n_d, int, "num_repeats"
            )
        self.precision_criterion_target = precision_criterion_target
        self.nprior_per_nlive = nprior_per_nlive
        self.max_ncalls = max_ncalls
        # Pool for storing intermediate results during parallelised acquisition
        self._X_mc, self._y_mc, self._sigma_y_mc, self._w_mc = None, None, None, None
        self._X_mc_reweight, self._y_mc_reweight = None, None
        self._sigma_y_mc_reweight, self._w_mc_reweight = None, None
        self.is_last_mc_reweighted = None
        self.rank_method = rank_method
        self.rank_merge_method = rank_merge_method
        self.pool = None
        self._acq_mc = None

    @property
    def pool_size(self):
        """Size of the pool of points."""
        if self.pool is None:
            return None
        return len(self.pool)

    def _init_nested_sampler(self, sampler=None):
        """
        Initializes the nested sampler, managing defaults with a recursive call.

        Raises
        ------
        ``gpry.ns_interfaces.NestedSamplerNotInstalledError`` if the requested or fallback
        sampler cannot be loaded, or ``ValueError`` if the sampler is not recognised.
        """
        this_sampler = sampler or self.sampler
        # Manage defaults
        if this_sampler is None:
            try:
                self._init_nested_sampler("polychord")
                return
            except nsint.NestedSamplerNotInstalledError as excpt:
                self.log(
                    f"Importing the default NS PolyChord failed (Err msg: {excpt}). "
                    "Defaulting to UltraNest."
                )
                self._init_nested_sampler("ultranest")
                return
        # Load the requested sampler
        try:
            self.sampler_interface = nsint._ns_interfaces[this_sampler.lower()](
                self.bounds_, self.verbose
            )
        except (AttributeError, KeyError) as excpt:
            if this_sampler.lower() != "uniform":
                raise ValueError(
                    "No interface found for the requested nested sampler "
                    f"'{this_sampler}'. Use one of {list(nsint._ns_interfaces)}"
                ) from excpt
        self.sampler = this_sampler

    def update_NS_precision(self, surrogate):
        """
        Updates NS precision parameters:

        - num_repeats: constant for now

        - nlive: `nlive_per_training` times the size of the training set, capped at
            `nlive_max` (typically 25 * dimension).

        - precision_criterion: constant for now.
        """
        nlive = min(self.nlive_per_training * surrogate.n_regress, self.nlive_max)
        return {
            "nlive": nlive,
            "num_repeats": self.num_repeats,
            "precision_criterion": self.precision_criterion_target,
            "nprior": int(self.nprior_per_nlive * nlive),
            "max_ncalls": self.max_ncalls,
        }

    def log(self, msg, level=None):
        """
        Print a message if its verbosity level is equal or lower than the given one (or
        always if ``level=None``.
        """
        if level is None or level <= self.verbose:
            print(self.log_header + msg)

    def _get_output_folder(self):
        """
        Prepare a new output folder. Returns always with a ``/`` at the end.

        If one was specified at init, it saves to a subfolder with an increasing index.
        Otherwise, it creates a random one every time.
        """
        if not mpi.is_main_process:
            return None
        if self.tmpdir is None:
            tmpdir = tempfile.TemporaryDirectory().name
        else:
            tmpdir = os.path.abspath(os.path.join(self.tmpdir, str(self.i)))
            self.i += 1
        if not tmpdir.endswith("/"):
            tmpdir += "/"
        return tmpdir

    def do_mc_sample(self, surrogate, bounds, rng=None, sampler=None):
        """

        Returns
        -------
        X, y, sigma_y, weights, logZ, logZstd
            May return None for any of y, sigma_y, weights
        """
        if sampler is None:
            sampler = self.sampler
        if sampler.lower() == "uniform":
            return self._do_mc_sample_uniform(surrogate, bounds=bounds, rng=rng)
        if sampler.lower() == "polychord":
            return self._do_mc_sample_polychord(surrogate, bounds=bounds, rng=rng)
        if sampler.lower() == "ultranest":
            return self._do_mc_sample_ultranest(surrogate, bounds=bounds, rng=rng)
        if sampler.lower() == "nessai":
            return self._do_mc_sample_nessai(surrogate, bounds=bounds, rng=rng)
        if sampler.lower() == "blackjax":
            return self._do_mc_sample_blackjax(surrogate, bounds=bounds, rng=rng)
        raise ValueError(f"Sampler '{sampler}' not known.")

    # For tests only.
    # TODO: merge samples for >1 MPI processes.
    def _do_mc_sample_uniform(self, surrogate, bounds=None, rng=None):
        if not mpi.is_main_process:
            return None, None, None, None, None, None
        proposer = UniformProposer(self.bounds_ if bounds is None else bounds)
        n_total = 1000 * surrogate.d**2
        X = np.empty(shape=(n_total, surrogate.d))
        for i in range(n_total):
            X[i] = proposer.get(rng=rng)
        return X, None, None, None, None, None

    def _do_mc_sample_polychord(self, surrogate, bounds=None, rng=None):
        # NB: unlike UltraNest (below), PolyChord takes -inf directly (it compares
        # against its own `logzero`), so `surrogate.minus_inf_value` is left at -inf
        # and there is no finite stand-in here that could be mis-ordered.
        # Update prior bounds
        self.sampler_interface.set_prior(self.bounds_ if bounds is None else bounds)
        # Update PolyChord precision settings
        self.sampler_interface.set_precision(**self.update_NS_precision(surrogate))
        # Prepare seed for reproducibility (positive integer < 2^31); only rank 0 used.
        seed = rng.integers(2**31 - 1) if rng is not None else None
        # Output (PolyChord needs a "/" at the end).
        # Run and get products
        # NB: the atleast_2d slows down the code a bit, but allows us to drop validation
        X_mc, y_mc, w_mc, logZ, logZstd = self.sampler_interface.run(
            lambda X: surrogate.predict(
                np.atleast_2d(X), return_std=False, validate=False
            )[0],
            out_dir=self._get_output_folder(),
            seed=seed,
        )
        self.sampler_interface.delete_output()
        # We will recompute y values, because quantities in PolyChord have to go through
        # text i/o, and some precision may be lost -- so we do not return them.
        y_mc = None
        return X_mc, y_mc, None, w_mc, logZ, logZstd

    def _do_mc_sample_ultranest(self, surrogate, bounds=None, rng=None):
        # UltraNest cannot handle -inf likelihoods, so points the infinities
        # classifier masks out need a FINITE stand-in. That stand-in must be
        # strictly worse than anything the sampler can find in the finite region.
        #
        # It used to be -1e-300, which is zero to ~300 digits, i.e. the LARGEST
        # log-posterior representable rather than the smallest. Nested sampling
        # then converges onto the masked region instead of the posterior: it
        # terminates after ~2 e-folds reporting `Explored until L=-1e-300`, having
        # never sampled the target. The failure scales with the masked fraction of
        # the prior, so it is invisible in low dimension and total in high.
        #
        # `SurrogateModel.minus_inf_value_substitute` (-1e30) is far below any
        # realistic log-posterior, and an upper clip cannot raise it, so
        # entirely-masked batches and mixed batches agree (`predict` clips on both
        # of its return paths). See `gpry.surrogate.MINUS_INF_SUBSTITUTE` for why
        # -1e30 and not something more extreme.
        _sentinel = surrogate.minus_inf_value_substitute

        def logp(X):
            """
            Returns the predicted value at a given point (a finite, very low
            stand-in where the prior/classifier says -inf).
            """
            prev_miv = surrogate.minus_inf_value
            surrogate.minus_inf_value = _sentinel
            logp = surrogate.predict(np.atleast_2d(X), return_std=False, validate=False)
            surrogate.minus_inf_value = prev_miv
            return logp

        # Update prior bounds
        self.sampler_interface.set_prior(self.bounds_ if bounds is None else bounds)
        # Update precision settings
        prec_settings = {
            k: v
            for k, v in self.update_NS_precision(surrogate).items()
            if k in ["nlive", "precision_criterion", "num_repeats", "max_ncalls"]
        }
        self.sampler_interface.set_precision(**prec_settings)
        # Seeding -- for now UltraNest does not accept an rng or a seed, only setting the
        # np.random seed globally, which is dangerous!
        # TODO: Create a PR for taking a custom RNG in UltraNest.
        if rng is not None:
            if mpi.is_main_process:
                warnings.warn("Seeded runs are not supported for UltraNest.")
        # Run and get products
        X_mc, y_mc, w_mc, logZ, logZstd = self.sampler_interface.run(
            logp, out_dir=self._get_output_folder()
        )
        self.sampler_interface.delete_output()
        # We will recompute y values, because quantities in PolyChord have to go through
        # text i/o, and some precision may be lost -- so we do not return them.
        y_mc = None
        return X_mc, y_mc, None, w_mc, logZ, logZstd

    def _do_mc_sample_nessai(self, surrogate, bounds=None, rng=None):
        if not mpi.is_main_process:
            return None, None, None, None
        if mpi.multiple_processes:
            warnings.warn(
                "Support for Nessai is experimental at the moment, and not MPI-compatible"
                " (running in rank-0 process only)."
            )

        # Initialise "likelihood" -- returns surrogate value and deals with pooling/ranking
        # NB: as for PolyChord and unlike UltraNest, nessai takes -inf directly, so
        # `surrogate.minus_inf_value` is left at -inf: no finite stand-in to mis-order.
        def logp(X):
            """
            Returns the predicted value at a given point (-inf if prior=0).
            """
            return surrogate.predict(X, return_std=False, validate=False)

        # Update prior bounds
        self.sampler_interface.set_prior(self.bounds_ if bounds is None else bounds)
        # Update precision settings
        prec_settings = {
            k: v
            for k, v in self.update_NS_precision(surrogate).items()
            if k in ["nlive", "precision_criterion"]
        }
        self.sampler_interface.set_precision(**prec_settings)
        # Prepare seed for reproducibility (positive integer < 2^31); only rank 0 used.
        seed = rng.integers(2**31 - 1) if rng is not None else None
        # Run and get products
        X_mc, y_mc, w_mc, logZ, logZstd = self.sampler_interface.run(
            logp,
            out_dir=self._get_output_folder(),
            seed=seed,
        )
        self.sampler_interface.delete_output()
        # We will recompute y values, because quantities in PolyChord have to go through
        # text i/o, and some precision may be lost -- so we do not return them.
        y_mc = None
        return X_mc, y_mc, None, w_mc, logZ, logZstd

    def _do_mc_sample_blackjax(self, surrogate, bounds=None, rng=None):
        warn("Support for BlackJax is experimental (and slow).")
        # Update prior bounds
        self.sampler_interface.set_prior(self.bounds_ if bounds is None else bounds)
        # Update precision settings
        self.sampler_interface.set_precision(**self.update_NS_precision(surrogate))
        # Prepare seed for reproducibility (positive integer < 2^31); only rank 0 used.
        seed = rng.integers(2**31 - 1) if rng is not None else None
        # Run and get products
        X_mc, y_mc, w_mc, logZ, logZstd = self.sampler_interface.run(
            lambda X: surrogate.predict(
                np.atleast_2d(X), return_std=False, validate=False
            )[0],
            out_dir=self._get_output_folder(),
            seed=seed,
        )
        self.sampler_interface.delete_output()
        # For safety, and since it is very cheap, we will recompute y values later.
        y_mc = None
        return X_mc, y_mc, None, w_mc, logZ, logZstd

    def _set_mc_sample(
        self, X, y, sigma_y, w, logZ, logZstd, ensure_y_sigma_y=False, surrogate=None
    ):
        """
        Stores the MC sample as attributes.

        If either `y` and/or `sigma_y` are passed as `None`, you can ensure their
        calculation (in parallel) with `ensure_y_sigma=True`. In that case, a
        ``surrogate`` is needed.

        Use ``last_mc_sample[_getdist]`` to retrieve the sample arrays. The last
        log-evidence calculation and its uncertainty as available as attributes ``logZ``
        and ``logZstd``, respectively.
        """
        self.is_last_mc_reweighted = False
        self._X_mc, self._y_mc, self._sigma_y_mc, self._w_mc = X, y, sigma_y, w
        if ensure_y_sigma_y:
            self._y_mc, self._sigma_y_mc = mpi.compute_y_parallel(
                surrogate, self._X_mc, self._y_mc, self._sigma_y_mc, ensure_sigma_y=True
            )
        self._logZ, self._logZstd = logZ, logZstd

    def _reweight_last_mc_sample(self, surrogate, bounds=None, ensure_sigma_y=False):
        """Stores the MC sample as attributes. Use ``last_mc_sample`` to retrieve it."""
        self.is_last_mc_reweighted = True
        X_excpt, y_excpt = None, None
        if mpi.is_main_process and self._X_mc is None:
            X_excpt = ValueError("No samples yet!")
        X_excpt = mpi.bcast(X_excpt)
        if X_excpt is not None:
            raise X_excpt
        if mpi.is_main_process and self._y_mc is None:
            y_excpt = ValueError("Original logp was not stored. Cannot reweight!")
        y_excpt = mpi.bcast(y_excpt)
        if y_excpt is not None:
            raise y_excpt
        # Ensure y and sigma_y (optional) are computed
        self._X_mc_reweight = None
        if mpi.is_main_process:
            self._X_mc_reweight = np.copy(self._X_mc)
            if bounds is not None:
                # Keep points within new bounds (maybe none!)
                i_within = is_in_bounds(self._X_mc_reweight, bounds, validate=False)
                self._X_mc_reweight = self._X_mc_reweight[i_within]
                # TODO: not handled: there could be 0 points within new bounds
        self._y_mc_reweight, self._sigma_y_mc_reweight = mpi.compute_y_parallel(
            surrogate, self._X_mc_reweight, None, None, ensure_sigma_y=ensure_sigma_y
        )
        if mpi.is_main_process:
            # Reweight, and drop 0 weights
            with NumpyErrorHandling(all="ignore") as _:
                y_mc = self._y_mc
                w_mc = self._w_mc
                if bounds is not None:
                    y_mc = y_mc[i_within]
                    w_mc = w_mc[i_within] if w_mc is not None else None
                reweight_factor = np.exp(self._y_mc_reweight - y_mc)
                w_mc_reweight = (
                    w_mc
                    if w_mc is not None
                    else np.ones(shape=self._X_mc_reweight.shape[0])
                ) * reweight_factor
                w_mc_reweight /= max(w_mc_reweight)
            (
                self._w_mc_reweight,
                self._X_mc_reweight,
                self._y_mc_reweight,
                self._sigma_y_mc_reweight,
            ) = remove_0_weight_samples(
                w_mc_reweight,
                self._X_mc_reweight,
                self._y_mc_reweight,
                self._sigma_y_mc_reweight,
            )

    def last_mc_sample(self, copy=False, warn_reweight=True):
        """
        Returns the last MC sample as ``(X, y, sigma_y, weights)``. ``y, sigma_y``
        may be None if not computed while sampling. They can be generated with the
        surrogate model. If ``weights`` is None, all samples should be assumed to have
        equal weights.

        Prints a warning if it is a reweighted sample.
        """
        if self.is_last_mc_reweighted:
            if warn_reweight:
                warnings.warn(
                    "This is a reweighted sample! (disable with `warn_reweight=False`)"
                )
            return_values = (
                self._X_mc_reweight,
                self._y_mc_reweight,
                self._sigma_y_mc_reweight,
                self._w_mc_reweight,
            )
        else:
            return_values = (self._X_mc, self._y_mc, self._sigma_y_mc, self._w_mc)
        if copy:
            return_values = tuple(
                (np.copy(val) if val is not None else None) for val in return_values
            )
        return return_values

    @property
    def mean(self):
        Xs, _, _, ws = self.last_mc_sample(copy=False, warn_reweight=False)
        return np.average(Xs.T, weights=ws, axis=-1)

    @property
    def cov(self):
        Xs, _, _, ws = self.last_mc_sample(copy=False, warn_reweight=False)
        return np.cov(Xs.T, aweights=ws, ddof=0)

    def last_mc_sample_getdist(self, params, warn_reweight=True, logprior_func=None):
        """
        Returns the last MC sample as a ``getdist.MCSamples`` instance.

        If ``logprior_func`` is passed, includes log-prior and log-likelihood in the
        sample as derived paramters.

        Prints a warning if it is a reweighted sample.
        """
        X, y, _, w = self.last_mc_sample(warn_reweight=warn_reweight)
        samples_dict = {"w": w, "X": X, _name_logp: y}
        if logprior_func:
            samples_dict[_name_logprior] = np.array([logprior_func(x) for x in X])
            samples_dict[_name_loglike] = y - samples_dict[_name_logprior]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return samples_dict_to_getdist(
                samples_dict,
                params=params,
                bounds=self.bounds_,
                sampler_type="nested",
            )

    def multi_add(
        self, surrogate, n_points=1, bounds=None, rng=None, force_resample=False
    ):
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
        surrogate : SurrogateModel
            The GP Regressor which is used as surrogate model.

        n_points : int, optional (default=1)
            Number of points to be returned. A value large than 1 is useful if you can
            evaluate your objective in parallel, and thus obtain more objective function
            evaluations per unit of time.

        bounds : np.array, optional
            Bounds inside which to look for the next proposals, e.g. the surrogate model's
            trust region. If not defined, the prior bounds are used.

        rng : int or numpy.random.Generator, optional
            The generator used to perform the acquisition process. If an integer is given,
            it is used as a seed for the default global numpy random number generator.

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
        # Create (parallel) generator(s) if int passed as rng
        rng = mpi.get_random_generator(rng)
        # Gather an MC sample, only not-None for rank 0; bcasted by _split_and_compute_acq
        if mpi.is_main_process:
            start_sample = time()
        mc_sample_this_time = (
            not bool(self.mc_every_i % self.mc_every) or force_resample
        )
        if mc_sample_this_time:
            try:
                mc_output = self.do_mc_sample(surrogate, bounds=bounds, rng=rng)
            except Exception as excpt:
                self.log(
                    level=0,
                    msg=f"Error when finding new acquisition optima with {self.sampler}: "
                    f"{excpt}. Attempting uniform prior sampling.",
                )
                mc_output = self.do_mc_sample(
                    surrogate, bounds=bounds, rng=rng, sampler="uniform"
                )
            self._set_mc_sample(*mc_output, ensure_y_sigma_y=True, surrogate=surrogate)
            self._X_already_proposed = np.empty(shape=(0, surrogate.d))
        else:
            self._reweight_last_mc_sample(surrogate, bounds=bounds, ensure_sigma_y=True)
        self.mc_every_i += 1
        X_mc, y_mc, sigma_y_mc, _ = self.last_mc_sample(warn_reweight=False)
        # Find indices of already used elements to exclude them.
        # Needs to be here because _reweight_last_mc_sample changes the indices.
        # Both the X's of the MC sample and the pool are assumed unique.
        if mpi.is_main_process and self._X_already_proposed.size > 0:
            i_already_proposed = []
            for X_i in self._X_already_proposed:
                i_this_one = np.flatnonzero(
                    np.all(np.isin(X_mc, X_i, assume_unique=True), axis=1)
                )
                if i_this_one.size > 0:
                    i_already_proposed.append(i_this_one[0])
            X_mc = np.delete(X_mc, i_already_proposed, axis=0)
            y_mc = np.delete(y_mc, i_already_proposed, axis=0)
            sigma_y_mc = np.delete(sigma_y_mc, i_already_proposed, axis=0)
        # Compute acq functions and missing quantities.
        self.acq_func_y_sigma = partial(
            self.acq_func.f,
            baseline=surrogate.y_max,
            noise_level=surrogate.noise_level,
            zeta=self.acq_func.zeta,
        )
        # *Split* among MPI processes and compute acq func value (in parallel)
        this_X, this_y, this_sigma_y, this_acq = self._split_and_compute_acq(
            X_mc, y_mc, sigma_y_mc
        )
        if mpi.is_main_process:
            what_we_did = (
                f"Obtained new MC sample with {self.sampler}"
                if mc_sample_this_time
                else "Re-evaluated previous MC sample"
            )
            self.log(f"({(time() - start_sample):.2g} sec) {what_we_did}")
        # Rank to get best points:
        mpi.sync_processes()
        if mpi.is_main_process:
            start_rank = time()
        args = (this_X, this_y, this_sigma_y, this_acq, n_points, surrogate)
        merged_pool = self._parallel_rank_and_merge(
            *args, method=self.rank_method, merge_method=self.rank_merge_method
        )
        # In case the pool is not full (not enough "good" points added), drop empty slots
        merged_pool = merged_pool.copy(drop_empty=True)
        X_pool, y_pool = merged_pool.X[:n_points], merged_pool.y[:n_points]
        with np.errstate(divide="ignore"):
            acq_pool = self.acq_func_y_sigma(y_pool, merged_pool.sigma[:n_points])
        # Track the used ones, to ignore them until new MC sample drawn.
        self._X_already_proposed = np.concatenate([self._X_already_proposed, X_pool])
        mpi.sync_processes()
        self.pool.reset_cache()  # reduces size of pickled object
        if mpi.is_main_process:
            self.log(f"({(time() - start_rank):.2g} sec) Ranked pool of candidates.")
        return X_pool, y_pool, acq_pool

    def _split_and_compute_acq(self, X, y, sigma_y):
        """
        Scatters `(X, y, sigma_y)` between processes, and returns them together with the
        acquisition function values, computed in parallel.
        """
        # We don't use mpi.split_number_for_parallel_processes because for the ranking
        # it is good to have similar y scaling in all MPI processes. But if we use that
        # function, some processes get the top of the dist, and others the bottom, and
        # the bottom one's scaling does not perform well with the add_one method, even
        # after sorting, and the slowest MPI process sets the global speed
        this_X = mpi.step_split(X)
        this_y = mpi.step_split(y)
        this_sigma_y = mpi.step_split(sigma_y)
        with np.errstate(divide="ignore"):
            this_acq = self.acq_func_y_sigma(this_y, this_sigma_y)
        return this_X, this_y, this_sigma_y, this_acq

    # Parallel version of the ranking of the MC points
    def _parallel_rank_and_merge(
        self,
        this_X,
        this_y,
        this_sigma_y,
        this_acq,
        n_points,
        surrogate,
        method=None,
        merge_method=None,
    ):
        if method is None:
            method = "auto"
        # For dimensionalities 4 and smaller, bulk adding is expected to be faster.
        if method.lower() == "auto":
            method = "bulk" if surrogate.d <= 4 else "single sort acq"
            merge_method = "bulk"
        # The size of the pool should be at least the amount of points to be acquired.
        # If running several processes in parallel, it can be reduced down to the number
        #   of points to be evaluated per process, but with less guarantee to find an
        #   optimal set.
        self.pool = RankedPool(
            n_points,
            surrogate=surrogate,
            acq_func=self.acq_func_y_sigma,
            verbose=self.verbose - 3,
        )
        with np.errstate(divide="ignore"):
            self.pool.add(this_X, this_y, this_sigma_y, this_acq, method=method)
            merged_pool = self._merge_pools(n_points, surrogate, method=merge_method)
        return merged_pool

    def _gather_pools(self):
        """
        Merges the points in all pools, discarding the empty ones.

        rank-0 process returns [X, y, sigma, acq], where the last two are the
        unconditioned input ones.
        """
        pool_X = mpi.gather(self.pool.X[: len(self.pool)])
        pool_y = mpi.gather(self.pool.y[: len(self.pool)])
        pool_sigma = mpi.gather(self.pool.sigma[: len(self.pool)])
        pool_acq = mpi.gather(self.pool.acq[: len(self.pool)])
        # Using the conditional acq value just to discard empty slots (acq=-inf)
        # Later discarded (not returned), since they need to be recomputed anyway.
        pool_acq_cond = mpi.gather(self.pool.acq_cond[: len(self.pool)])
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

    def _merge_pools(self, n_points, surrogate, method=None):
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
                n_points,
                surrogate=surrogate,
                acq_func=self.acq_func_y_sigma,
                verbose=self.pool.verbose,
            )
            merged_pool.add(pool_X, pool_y, pool_sigma, pool_acq, method=method)
        merged_pool = mpi.bcast(merged_pool)
        return merged_pool


class RankedPool:
    """
    Keeps a ranked pool of sample proposal for Krigging-believer, given a GP regressor
    and an acquisition function.

    Parameters
    ----------
    size : int
        Number of points sampled proposals targeted.

    surrogate : SurrogateModel
        The GP Regressor which is used as surrogate model.

    acq_func : callable
        Acquisition function used to rank the pool. Must be a function of ``(y, sigma)``
        only, partially evaluated its hyperparameters, if necessary.

    verbose : 1, 2, 3, optional (default: 1)
        Level of verbosity. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise.
    """

    def __init__(self, size, surrogate, acq_func, verbose=1):
        self._surrogate = surrogate
        self._acq_func = acq_func
        self.verbose = verbose if verbose is not None else 3
        # The pool should have one more element than the number of desired points.
        self.X = np.zeros((size + 1, surrogate.d))
        self.y = np.zeros((size + 1))
        # Condioned acquisition, used for ranking; -np.inf means "empty slot"
        self.acq_cond = np.full((size + 1), -np.inf)
        # Input quantities, only used at point, stored just for logging
        self.sigma = np.zeros((size + 1))
        self.acq = np.zeros((size + 1))
        # Cached conditioned surrogates
        self.reset_cache()
        # Counter how many models have been cached, for efficiency checks
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
        self, include_last=False, last_sorted=None, prefix=None, suffix_last=None
    ):
        """Returns a string representation of the current pool."""
        pool_str = ""
        for i in range(len(self.X) + (-1 if not include_last else 0)):
            pool_str += (
                (prefix or "")
                + f"{i + 1} : "
                + self.str_point(
                    self.X[i],
                    self.y[i],
                    self.sigma[i],
                    self.acq[i],
                    acq_cond=self.acq_cond[i],
                )
                + (" [last sorted]" if i == last_sorted else "")
                + "\n"
            )
        return pool_str.rstrip("\n") + (
            f" {suffix_last}" if include_last and suffix_last else ""
        )

    def log_pool(
        self,
        level=4,
        include_last=False,
        last_sorted=None,
        prefix=None,
        suffix_last=None,
    ):
        """Prints the current pool."""
        if self.verbose >= level:
            self.log(
                level=level,
                msg=self.str_pool(
                    include_last=include_last,
                    last_sorted=last_sorted,
                    prefix=prefix,
                    suffix_last=suffix_last,
                ),
            )

    def __str__(self):
        return self.str_pool(include_last=False)

    def normalize_input(self, X, y, sigma, acq):
        """Ensures vector types and computes missing quantities."""
        X = np.atleast_2d(X)
        if y is None:  # assume sigma is also None
            y, sigma = self._surrogate.predict(X, return_std=True, validate=False)
        elif not hasattr(y, "__len__"):
            y = np.array([y])
        if sigma is None:
            sigma = self._surrogate.predict_std(X, validate=False)
        if acq is None:
            acq = self._acq_func(y, sigma)
        return X, np.atleast_1d(y), np.atleast_1d(sigma), np.atleast_1d(acq)

    def add(self, X, y=None, sigma=None, acq=None, method="single sort acq"):
        """
        Adds points to the pool.

        Parameters
        ----------
        X: np.ndarray (1- or 2-dimensional)
            Position of the proposed sample.

        y: np.ndarray (1 dimension fewer than X) or float, optional
            Predicted value under the surrogate model.

        sigma: np.ndarray (1 dimension fewer than X) or float, optional
            Predicted standard deviation under the surrogate model. Will be computed if
            not passed.

        acq: np.ndarray (1 dimension fewer than X) or float, optional
            Acquisition function values (unconditioned). Will be computed if not passed.

        method: {"single", "single sort acq", "single sort y", "bulk"}
            Uses the one-by-one algorithm ("single", with pre-sorting according to X if
            "single sort X"), or the bulk algorithm.
        """
        X, y, sigma, acq = self.normalize_input(X, y, sigma, acq)
        if method.lower() == "bulk":
            self.add_bulk(X, y, sigma, acq)
        elif method.lower().startswith("single"):
            i_sort = None
            if "sort" in method.lower():
                i_sort = np.argsort({"acq": acq, "y": y}[method.lower().split()[-1]])[
                    ::-1
                ]
            # Descending order of unconditioned acq or unconditional mean prediction:
            # minimizes the number of swaps: model caches + calculation of acq_cond
            for i in i_sort if i_sort is not None else range(len(X)):
                self.add_one(X[i], y[i], sigma[i], acq[i])
        else:
            raise ValueError(f"Algorithm '{method}' not known.")

    def add_bulk(self, X, y, sigma, acq, i_start=0):
        """
        Tries to fill the pool using a batch of points at once:

        1. Compute their acquisition value conditioned to the position above (if any).
        2. Pick the best and delete infinities (acq cannot grow with more conditioning).
        3. Place it in the current position, and do a recursive call for the next one.

        The advantage of this method with respect to ``add_one`` is that it can use
        vectorization to compute the std's, but on the other hand it needs to compute
        many more of them, so it will be better only up to some dimension and some amount
        of training, and then ``add_one`` will take over.
        """
        logpref = "[pool.add_bulk] "
        if self.verbose >= 4:
            self.log(level=4, msg=logpref + f"Bulk-adding for position {i_start + 1}")
        # Compute acq using the model just above (and cache it if needed)
        if i_start == 0:  # no need to condition
            surrogate = self._surrogate
            acq_cond = acq if isinstance(acq, np.ndarray) else np.array(acq)
        else:
            surrogate = self.cache_model(i_start - 1)
            sigma_cond = surrogate.predict_std(X, validate=False)
            acq_cond = self._acq_func(y, sigma_cond)
        if self.verbose >= 4:
            self.log(level=4, msg=logpref + "Current batch:")
            i_max = np.argmax(acq_cond)
            for i in range(len(X)):
                str_point = self.str_point(
                    X[i], y[i], sigma[i], acq[i], acq_cond=acq_cond[i]
                )
                prefix = "--->" if i == i_max else "    "
                self.log(level=4, msg=logpref + prefix + str_point)
        if acq_cond.size == 0:
            self.log(
                level=4,
                msg=(
                    logpref
                    + f"No finite acq points to fill the pool from [{i_start}] down."
                ),
            )
            return
        # Find best
        i_max = np.argmax(acq_cond)
        acq_cond_max = acq_cond[i_max]
        if acq_cond_max == np.inf:
            self.log(
                level=4,
                msg=(
                    logpref
                    + f"No finite acq points to fill the pool from [{i_start}] down."
                ),
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
            Predicted value under the surrogate model.

        sigma: float, optional
            Predicted standard deviation under the surrogate model.

        acq: float, optional
            Value of the acquisition function.

        acq_nan_is_null: bool, optional (default: False)
            Whether NaN's in the acquisition function should be interpreted as null value.

        Raises
        ------
        ValueError: if invalid acq. function value, unless ``acq_nan_is_null=True``.
        """
        logpref = "[pool.add_one] "
        if self.verbose >= 4:
            self.log(
                level=4,
                msg=(logpref + "Checking point " + self.str_point(X, y, sigma, acq)),
            )
        # Discard the point as early as possible!
        # NB: The equals sign below takes care of the case in which we are trying to add a
        # point with -inf acq. func. to a pool which is not full (min acq. func. = -inf)
        if acq <= self.min_acq:
            self.log(level=4, msg=logpref + "Acq. func. value too small. Ignoring.")
            return
        X, y, sigma, acq = self.normalize_input(X, y, sigma, acq)
        # In this case, for pool asignment, enforce some scalars
        y, sigma, acq = np.squeeze(y)[()], np.squeeze(sigma)[()], np.squeeze(acq)[()]
        # Repeat the min acq test above to leave asap
        if acq <= self.min_acq:
            self.log(level=4, msg=logpref + "Acq. func. value too small. Ignoring.")
            return
        if np.isnan(acq):
            if not acq_nan_is_null:
                raise ValueError(f"Acquisition function value not a number: {acq}")
            acq = -np.inf
        self.log(level=4, msg=logpref + "Initial pool:")
        self.log_pool(level=4, prefix=logpref)
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
                i_new = len(self) - next(
                    i for i in range(len(self)) if self.acq_cond[-(i + 2)] >= acq_cond
                )
            except StopIteration:  # top of the list reached
                i_new = 0
            self.log(level=4, msg=logpref + f"Provisional position: [{i_new + 1}]")
            # top, same as last or last: final ranking reached
            if i_new in [0, i_new_last, len(self)]:
                break
            # Otherwise, compute conditioned acquisition value, using point above,
            # and continue to the next iteration to re-rank
            sigma_cond = self.surrogate_cond[i_new - 1].predict_std(X, validate=False)[
                0
            ]
            # New acquisition should not be higher than the old one, since the new one
            # corresponds to a model with more training points (though fake ones).
            # This may happen anyway bc numerical errors, e.g. when the correlation
            # length is really huge. Also, when alpha or the noise level for two cached
            # models are different. We can just ignore it.
            # (Sometimes, it's just ~1e6 relative differences, which is not worrying)
            acq_cond = min(acq_cond, self._acq_func(y, sigma_cond))
            i_new_last = i_new
            if self.verbose >= 4:  # avoid creating the f-strings
                self.log(
                    level=4, msg=logpref + f"Updated conditional std: {sigma_cond}"
                )
                self.log(
                    level=4,
                    msg=logpref + f"Updated conditional acquisition: {acq_cond}",
                )
        # The last position is just a place-holder: don't save it if it falls there.
        if i_new >= len(self):
            self.log(level=4, msg=logpref + "Discarded!")
            return
        self.log(level=4, msg=logpref + f"Final position: [{i_new + 1}] of {len(self)}")
        # Insert the new one in its place, and push the rest down one place.
        # We track the conditioned acq. value (but not the sigma), to retain the
        # information about whether each slot was empty (acq = -inf)
        # (but not for the acq values, which will be updated below)
        for pool, value in [
            (self.X, X),
            (self.y, y),
            (self.sigma, sigma),
            (self.acq, acq),
            (self.acq_cond, acq_cond),
        ]:
            pool[i_new + 1 :] = pool[i_new:-1]
            pool[i_new] = value
        # If not in the last position, we can safely assume that it has finite value,
        # since -inf's from conditional acq cannot climb.
        assert self.acq_cond[i_new] > -np.inf
        self.log(level=4, msg=logpref + "Current unsorted pool:")
        self.log_pool(level=4, include_last=True, last_sorted=i_new, prefix=logpref)
        # Sort the sublist below the new element
        self.sort(i_new + 1)
        self.log(level=4, msg=logpref + "The new pool, sorted:")
        self.log_pool(
            level=4, include_last=True, prefix=logpref, suffix_last="[unused]"
        )
        # Make sure that the last slot (buffer) is marked as empty
        self.acq_cond[-1] = -np.inf

    def cache_model(self, i):
        """
        Cache the GP model that contains the training set plus the pool points up to
        position ``i`` (0-based), with predicted dummy y, keeping the surrogate model
        hyperparameters unchanged.

        Stores and returns the conditioned surrofate (or the original one if ``i=-1``).
        """
        # This function accounts for ~50% of the ranking time in add_one() (the rest is
        # mostly predict_std()).
        # Taking dim=8 as reference, deepcopy is ~1/3 and append ~2/3 of the cost.
        # Possible optimization strategies:
        # - Disable SVM in cached models (no need to copy, fit, or evaluate), assuming all
        #   passed points are finite [tested to improve <10% overall in add_one{}]
        # - Copy model above instead of original one, and fit to single new point only
        #   [tested: no appreciable gain]
        # - Keep a single cached model, adding points to it when going down the list.
        #   If there are very few inversions, we save the cost of copying [potentially
        #   ~30% this function, 15% overall in add_one()]
        # - Disable the copying of stuff not needed to compute std.
        # - At append, compute only what is strictly needed for std (I think the
        #   kernel gradient is the only such thing.
        # - Create model anew with fixed given kernel and no SVM, and fit all points.
        # In any case, the cost is now very low compared to nested sampling and hyperparam
        # fitting.
        if i < 0:
            return self._surrogate
        self.log(level=4, msg=f"[pool.cache] Caching model [{i + 1}]")
        self.surrogate_cond[i] = deepcopy(self._surrogate)
        self.surrogate_cond[i].append(
            self.X[: i + 1], self.y[: i + 1], fit_gpr=False, fit_classifier=False
        )
        self.cache_counter += 1
        return self.surrogate_cond[i]

    def reset_cache(self):
        """
        Deletes the cached surrogate models when there are not needed any more.
        """
        # No need to store the last element in the list (or the last buffer slot)
        self.surrogate_cond = [None] * len(self.X - 1)

    def __getstate__(self):
        return deepcopy(self).__dict__

    # Remove references to external objects (surrogate and acq_func) and cached surrogates
    def __deepcopy__(self, memo=None):
        attrs_ignore_at_copy = ["_surrogate", "_acq_func", "surrogate_cond"]
        new = self.__class__.__new__(self.__class__)
        new.__dict__ = {
            k: deepcopy(v)
            for k, v in self.__dict__.items()
            if k not in attrs_ignore_at_copy
        }
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
        ``i``-th element (0-based) is conditioned on the surrogate model that includes the
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
                level=4, msg="[pool.sort] Nothing to do (sorting beyond last position)."
            )
            return
        self.log(
            level=4,
            msg=f"[pool.sort] Sorting the pool starting at [{i_start + 1}]",
        )
        upper_surrogate_cond = self.cache_model(i_start - 1)
        # If list not full (first sublist element's acq_cond=-inf), do nothing
        if self.acq_cond[i_start] == -np.inf:
            self.log(level=4, msg="[pool.sort] Nothing to do (list is not full yet).")
            return
        # Compute acq_cond for non-empty points (acq != -inf) only. Assume always sorted.
        try:
            i_1st_inf = next(i for i, ac in enumerate(self.acq_cond) if ac == -np.inf)
        except StopIteration:
            i_1st_inf = len(self) + 1
        sigma_cond = upper_surrogate_cond.predict_std(
            self.X[i_start:i_1st_inf], validate=False
        )
        # Cond acq cannot be higher than less cond one. This clipping takes care of
        # numerical noise that may make it higher. In particular, keeps -inf if it was so.
        acq_cond = np.clip(
            self._acq_func(self.y[i_start:i_1st_inf], sigma_cond),
            None,
            np.inf if i_start == 0 else self.acq_cond[i_start - 1],
        )
        if self.verbose >= 4:  # avoid creating the f-strings
            self.log(level=4, msg=f"[pool.sort] New conditioned std: {sigma_cond}")
            self.log(level=4, msg=f"[pool.sort] New conditioned acq: {acq_cond}")
        # descending order! -- This is a *sub*list index!
        j_sort = np.argsort(-acq_cond)
        acq_cond_max = acq_cond[j_sort[0]]
        # If the max found was -inf, no need to re-sort points: disable all and return
        if acq_cond_max == -np.inf:
            self.log(
                level=4,
                msg="[pool.sort] Nothing to do (all sublist elements have -inf acq).",
            )
            self.acq_cond[i_start:i_1st_inf] = -np.inf
            return
        self.log(
            level=4,
            msg=(
                f"[pool.sort] New max acq_cond = {acq_cond_max} "
                f"at position [{i_start + j_sort[0] + 1}]"
            ),
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
            level=4, include_last=True, last_sorted=i_start, prefix="[pool.sort] "
        )
        # Sort the sublist below (if empty or single element, taken care of inside)
        self.sort(i_start + 1)
