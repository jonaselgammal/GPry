import warnings
import numpy as np
import scipy.optimize
from copy import deepcopy
from sklearn.base import is_regressor
from sklearn.utils import check_random_state

from gpry.acquisition_functions import Log_exp, Nonlinear_log_exp
from gpry.acquisition_functions import is_acquisition_function
from gpry.proposal import UniformProposer
from gpry.mpi import mpi_comm, mpi_rank, is_main_process, \
    split_number_for_parallel_processes, multi_gather_array


class GP_Acquisition(object):
    """Run Gaussian Process acquisition.

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

    proposer : Proposer object, optional (default: "UniformProposer")
        Proposer to propose points from which the acquisition function should be optimized.

    acq_func : GPry Acquisition Function, optional (default: "Log_exp")
        Acquisition function to maximize/minimize. If none is given the
        `Log_exp` acquisition function will be used

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
        (Only if "Log_exp" is passed as acquisition_function)

    verbose : 1, 2, 3, optional (default: 1)
        Level of verbosity. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise.

    Attributes
    ----------

    gpr_ : GaussianProcessRegressor
            The GP Regressor which is currently used for optimization.

    """

    def __init__(self, bounds,
                 proposer=None,
                 acq_func="Log_exp",
                 acq_optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0,
                 n_repeats_propose=0,
                 preprocessing_X=None,
                 random_state=None,
                 verbose=1,
                 random_proposal_fraction=None,
                 zeta_scaling = None):

        self.bounds = bounds
        self.n_d = np.shape(bounds)[0]

        self.proposer = proposer
        self.obj_func = None

        self.rng = check_random_state(random_state)

        # If nothing is provided for the proposal, we use a uniform sampling
        if self.proposer is None:
            self.proposer = UniformProposer(self.bounds)
        else:
            # TODO: Catch error if it's not the right instance
            self.proposer = proposer

        if is_acquisition_function(acq_func):
            self.acq_func = acq_func
        elif acq_func == "Log_exp":
            # If the Log_exp acquisition function is chosen it's zeta is set
            # automatically using the dimensionality of the prior.
            self.acq_func = Log_exp(
                dimension=self.n_d, zeta_scaling=zeta_scaling)
        elif acq_func == "Nonlinear_log_exp":
            # If the Log_exp acquisition function is chosen it's zeta is set
            # automatically using the dimensionality of the prior.
            self.acq_func = Nonlinear_log_exp(
                dimension=self.n_d, zeta_scaling=zeta_scaling)
        else:
            raise TypeError("acq_func needs to be an Acquisition_Function "
                            "or 'Log_exp' or 'Nonlinear_log_exp', instead got %s" % acq_func)

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

        self.n_restarts_optimizer = n_restarts_optimizer
        self.n_repeats_propose = n_repeats_propose

        self.preprocessing_X = preprocessing_X

        self.verbose = verbose

        self.mean_ = None
        self.cov = None

    def propose(self, gpr, i, random_state=None):

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

            # Make the surrogate instance so it can be used in the objective
            # function
            self.gpr_ = gpr

            def obj_func(X, eval_gradient=False):

                # Check inputs
                X = np.asarray(X)
                X = np.expand_dims(X, axis=0)
                if X.ndim != 2:
                    raise ValueError("X is {}-dimensional, however, "
                                     "it must be 2-dimensional.".format(X.ndim))
                if self.preprocessing_X is not None:
                    X = self.preprocessing_X.inverse_transform(X)

                if eval_gradient:
                    acq, grad = self.acq_func(X, self.gpr_,
                                              eval_gradient=True)
                    return -1 * acq, -1 * grad
                else:
                    return -1 * self.acq_func(X, self.gpr_,
                                              eval_gradient=False)

            self.obj_func = obj_func

        # Preprocessing
        if self.preprocessing_X is not None:
            transformed_bounds = self.preprocessing_X.transform_bounds(
                self.bounds)
        else:
            transformed_bounds = self.bounds

        if i == 0:
            # Perform first run from last training point
            x0 = self.gpr_.X_train[-1]
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
                value = self.acq_func(x0, self.gpr_)
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
                #quit()
                if self.verbose > 1:
                    print(f"of {n_tries} initial samples for the "
                          "acquisition optimizer none returned a "
                          "finite value")
                if self.preprocessing_X is not None:
                    x0 = self.preprocessing_X.transform(x0)
                return x0, -1 * value

    def multi_add(self, gpr, n_points=1, random_state=None):
        r"""Method to query multiple points where the objective function
        shall be evaluated. The strategy which is used to query multiple
        points is by using the :math:`f(x)\sim \mu(x)` strategy and and not
        changing the hyperparameters of the model.

        This is done to increase speed since then the blockwise matrix
        inversion lemma can be used to invert the K matrix. The optimization
        for a single point is done using the :meth:`optimize_acq_func` method.

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
        if is_main_process:
            # Initialize arrays for storing the optimized points
            X_opts = np.empty((n_points,
                               gpr.d))
            y_lies = np.empty(n_points)
            acq_vals = np.empty(n_points)
            # Copy the GP instance as it is modified during
            # the optimization. The GP will be reset after the
            # Acquisition is done.
            gpr_ = deepcopy(gpr)
        gpr_ = mpi_comm.bcast(gpr_ if is_main_process else None)
        n_acq_per_process = \
            split_number_for_parallel_processes(self.n_restarts_optimizer)
        n_acq_this_process = n_acq_per_process[mpi_rank]
        i_acq_this_process = sum(n_acq_per_process[:mpi_rank])
        proposal_X = np.empty((n_acq_this_process, gpr_.d))
        acq_X = np.empty((n_acq_this_process,))
        for ipoint in range(n_points):
            # Optimize the acquisition function to get a few possible next proposal points
            # (done in parallel)
            for i in range(n_acq_this_process):
                proposal_X[i], acq_X[i] = self.propose(
                    gpr_, i + i_acq_this_process, random_state=random_state)
            proposal_X_main, acq_X_main = multi_gather_array(
                [proposal_X, acq_X])
            # Reset the objective function, such that afterwards the correct one is used
            self.obj_func = None
            # Now take the best and add it to the gpr (done in sequence)
            if is_main_process:
                # Find out which one of these is the best
                max_pos = np.argmin(acq_X_main) if np.any(
                    np.isfinite(acq_X_main)) else len(acq_X_main) - 1
                X_opt = proposal_X_main[max_pos]
                # Transform X and clip to bounds
                if self.preprocessing_X is not None:
                    X_opt = self.preprocessing_X.inverse_transform(X_opt,
                                                                   copy=True)

                # Get the value of the acquisition function at the optimum value
                acq_val = -1 * acq_X_main[max_pos]
                X_opt = np.array([X_opt])
                # Get the "lie" (prediction of the GP at X)
                y_lie = gpr_.predict(X_opt)
                ##print("\n\n\n\n --- GP ACQ SUMMARY ---- \n")
                ##print(np.hstack([gpr_.X_train,gpr_.y_train[:,None]]))
                ##print(proposal_X_main)
                print("gp_ac.py[{}]={} (pos={},val={}) -- lie = {}".format(ipoint,acq_X_main,max_pos,acq_X_main[max_pos],y_lie))
                ##print(gpr_.predict(X_opt,return_std=True,doprint=False))
                ##print(X_opt,X_opt-gpr_.X_train[-1])
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
            gpr_ = mpi_comm.bcast(gpr_ if is_main_process else None)

        gpr.n_eval = gpr_.n_eval  # gather #evals of the GP, for cost monitoring
        return ((X_opts, y_lies, acq_vals) if is_main_process else (None, None, None))

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

    def _has_already_been_sampled(self, gpr, new_X):
        """
        Method for determining whether points which have been found by the
        acquisition algorithm are already in the GP. This is called from the
        optimize_acq_func method to determine whether points may have been
        sampled multiple times which could break the GP.
        This is meant to be used for a **single** point new_X.
        """
        X_train = np.copy(gpr.X_train)

        for i, xi in enumerate(X_train):
            if np.allclose(new_X, xi):
                if self.verbose > 1:
                    warnings.warn("A point has been sampled multiple times. "
                                  "Excluding this.")
                return True
        return False
