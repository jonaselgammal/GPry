import sys
import warnings
from math import log
from numbers import Number

import numpy as np
from numpy.linalg import det
from numpy import trace as tr

import scipy.optimize
from scipy.special import logsumexp

from sklearn.base import clone
from sklearn.base import is_regressor
from joblib import Parallel, delayed
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import check_random_state
from sklearn.utils.optimize import _check_optimize_result
from gpry.acquisition_functions import Expected_improvement as EI
from gpry.acquisition_functions import is_acquisition_function
from gpry.gpr import GaussianProcessRegressor

from copy import deepcopy

import pdb
class GP_Acquisition(object):
    """Run Gaussian Process acquisition.

    Works similarly to a GPRegressor but instead of optimizing the kernel's
    hyperparameters it optimizes the Acquisition function in order to find one
    or multiple points at which the actual function should be evaluated next.

    Furthermore contains a framework for different lying strategies in order to improve
    performance on multiple processors and for different evaluation speeds of the function 
    to approximate.

    Use this class directly if you want to control the iterations of your
    bayesian quadrature loop.

    Parameters
    ----------
    bounds : array-like, shape=(n_dims,2)
        Array of bounds of the prior [lower, upper] along each dimension.

    surrogate_model : SKLearn Gaussian Process Regressor, optional (default: None)
        The GP Regressor which is used as surrogate model. 
        If None is given a GPRegressor with the standard settings (kernel=1.0*RBF(1.0) 
        along each dimension, restart_optimizer=0) is passed.
        If the GP Regressor already contains training points those points will not
        be deleted and just re-used.
 
    acq_func : GPry Acquisition Function, optional (default: "EI")
        Acquisition function to maximize/minimize. If none is given the `Expected 
        Improvement` acquisition function will be used

    acq_optimizer : string or callable, optional (default: "auto")
        Can either be one of the internally supported optimizers for optimizing
        the acquisition functionbounds, specified by a string, or an externally
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

        if set to "auto" either the 'fmin_l_bfgs_b' or 'sampling' algorithm 
        from scipy.optimize is used depending on whether gradient information
        is available or not.
    
    optimize_direction : "maximize" or "minimize", optional (default="maximize")
        Whether the acquisition function is supposed to be maximized or minimized.
        Set this parameter depending on the choice of acquisition function.

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the maximum of the
        acquisition function. The first run of the optimizer is performed from 
        the last X fit to the model if available, otherwise they are drawn at
        random.

        The remaining ones (if any) from X's sampled uniform randomly
        from the space of allowed X-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.

    random_state : int or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    model_queue_size : int or None, default=None
        Keeps list of models only as long as the argument given. In the
        case of None, the list has no capped length.

    Attributes
    ----------

    models : list
        Regression models used to fit observations and compute acquisition
        function. The X_train_ and y_train_ attributes of the regressors correspond to the
        points which were fit to the GP model.

    **Methods:**

    .. autosummary::
        :toctree: stubs

        multi_optimization
        optimize_acq_func
    """

    def __init__(self, dimensions, 
                 surrogate_model=None,
                 acq_func="EI",
                 acq_optimizer="fmin_l_bfgs_b",
                 optimize_direction="maximize",
                 random_state=None,
                 model_queue_size=None,
                 n_restarts_optimizer=0,
                 whiten_for_acquisition=True,
                 bto_for_acquisition=True):

        self._dimensions = dimensions # Keep copy of original dimensions
        self.dimensions = dimensions
        self.bto_for_acquisition = bto_for_acquisition
        if bto_for_acquisition:
            self.transformed_dimensions = surrogate_model.bto.transformed_bounds
        
        self.whiten_for_acquisition = whiten_for_acquisition # Just a dummy for now

        # surrogate model
        if surrogate_model is None:
            self.surrogate_model = GaussianProcessRegressor()
        elif is_regressor(surrogate_model):
            self.surrogate_model = surrogate_model
        else:
            raise ValueError(
                "surrogate model has to be a SKLearn regressor or 'RBF'."
                "got %s instead." % surrogate_model)

        self.rng = check_random_state(random_state)

        if is_acquisition_function(acq_func):
            self.acq_func = acq_func
        elif self.acq_func == "EI":
            self.acq_func = EI(1e-5)
        else:
            raise TypeError("acq_func needs to be an Acquisition_Function "
                            "or 'EI', instead got %s"%acq_func)

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
                    raise ValueError("In order to use the 'fmin_l_bfgs_b' optimizer the "
                                     "acquisition function needs to be able to return "
                                     "gradients. got %s"%self.acq_func)
                self.acq_optimizer = "fmin_l_bfgs_b"
            elif acq_optimizer == "sampling":
                self.acq_optimizer = "sampling"
            else:
                raise ValueError("Supported internal optimizers are 'auto', 'lbfgs' or "
                                "'sampling', got {0}".format(acq_optimizer))
       
        else:
            self.acq_optimizer = acq_optimizer

        if optimize_direction not in ["maximize", "minimize"]:
            raise ValueError("allowed values for optimize_direction are"
                             "'maximize' and 'minimize', got %s"%optimize_direction)
        self.optimize_direction = optimize_direction
        self.n_restarts_optimizer = n_restarts_optimizer

        # Initialize storage for optimization
        if not isinstance(model_queue_size, (int, type(None))):
            raise TypeError("model_queue_size should be an int or None, "
                            "got {}".format(type(model_queue_size)))
        self.max_model_queue_size = model_queue_size

        self.models = []

        self.mean_ = None
        self.cov = None
    
    def multi_optimization(self, n_points=1, n_cores=1):
        """Method to query multiple points where the objective function
        shall be evaluated. The strategy which is used to query multiple 
        points is by using the :math:`f(x)\sim \mu(x)` strategy and and not changing
        the hyperparameters of the model. 
        
        This is done to increase speed since then the blockwise matrix 
        inversion lemma can be used to invert the K matrix. The optimization 
        for a single point is done using the :meth:`optimize_acq_func` method.

        Parameters
        ----------

        n_points : int, optional(default=1)
            Number of points returned by the optimize method
            If the value is 1, a single point to evaluate is returned.

            Otherwise a list of points to evaluate is returned of size
            n_points. This is useful if you can evaluate your objective
            in parallel, and thus obtain more objective function evaluations
            per unit of time.
        n_cores : int, optional (default=1)
            Number of available cores on the machine. If left as 1 a single
            core is used. otherwise the load of the optimizer is run in 
            parallel on multiple processors. If n_restarts_optimizer is
            set to 1 n_cores will be defaulted to 1.

        Returns
        -------

        X : numpy.ndarray, shape = (X_dim, n_points)
            The X values of the found optima
        fval : numpy.ndarray, shape = (n_points,)
            The values of the acquisition function at X_opt
        """

        # Check if n_points is positive
        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(
                "n_points should be int > 0, got " + str(n_points)
            )
        
        if not hasattr(self.surrogate_model, "X_train_"):
            raise AttributeError("The model which is given has not been fed "
                                 "any points. Please make sure, that the model "
                                 "already contains data when trying to optimize an "
                                 "acquisition function on it as optimizing priors is "
                                 "not supported yet.")
        
        # Initialize arrays for storing the optimized points
        X_opt = np.empty((n_points, 
                self.surrogate_model.X_train_.shape[1]))
        func = np.empty(n_points)

        # Copy the GP instance as it is modified durself.dimensionsing 
        # the optimization. The GP will be reset after the
        # Acquisition is done.
        _surrogate_model = deepcopy(self.surrogate_model)

        for i in range(n_points):
            # Optimize the acquisition function 
            X, f_val = self.optimize_acq_func(n_cores=n_cores)
            # Update the surrogate model with the new lie
            lie = self.surrogate_model.predict([X])
            # Take the mean of errors as supposed measurement error
            if np.iterable(self.surrogate_model.alpha):
                lie_alpha = np.array([np.mean(self.surrogate_model.alpha)])
                self.surrogate_model.append_to_data(np.array([X]), 
                                                    lie, 
                                                    alpha=lie_alpha,
                                                    fit=False)
            else:
                self.surrogate_model.append_to_data(np.array([X]), 
                                                    lie, 
                                                    fit=False)
            # Append the points found to the list...
            X_opt[i] = X
            func[i]  = f_val

        #Reset the surrogate model to the original state
        self.surrogate_model = _surrogate_model
        X_opt=np.array(X_opt)
        func = np.array(func)
        return X_opt, func


    def optimize_acq_func(self, n_cores=1):
        """Exposes the optimization method for the acquisition function.

        Parameters
        ----------

        n_cores : int, optional (default=1)
            Number of available cores on the machine. If left as 1 a
            single core is used. otherwise the load of the optimizer 
            is run in parallel on multiple processors. If n_restarts_optimizer
            is set to 1 n_cores will be defaulted to 1.

        Returns
        -------
        X_opt : numpy.ndarray, shape = (X_dim,)
            The X value of the found optimum 
        func : float
            The value of the acquisition function at X_opt
        """
        
        if self.n_restarts_optimizer == 0:
            n_cores = 1

        if not hasattr(self.surrogate_model, "X_train_"):
            raise AttributeError("The model which is given has not been fed "
                                 "any points. Please make sure, that the model "
                                 "already contains data when trying to optimize an "
                                 "acquisition function on it as optimizing priors is "
                                 "not supported yet.")
        
        # Turn of normalization of priors during acquisition if it
        # has been selected
        if self.bto_for_acquisition:
            self.surrogate_model.normalize_bounds = False
            self.dimensions = self.transformed_dimensions

        def obj_func(X, eval_gradient=False):

            # Check inputs
            X = np.asarray(X)
            X = np.expand_dims(X, axis=0)
            if X.ndim != 2:
                raise ValueError("X is {}-dimensional, however,"
                                " it must be 2-dimensional.".format(X.ndim))

            if eval_gradient:
                acq, grad = self.acq_func(X, self.surrogate_model,
                    eval_gradient=True)
                return -acq, -grad
            else:
                return -1 * self.acq_func(X, self.surrogate_model,
                    eval_gradient=False)

        optima_X = np.empty((self.n_restarts_optimizer, 
                           self.surrogate_model.X_train_.shape[1]))
        optima_acq_func = np.empty(self.n_restarts_optimizer)

        # Runs are performed from uniform chosen initial X's
        if self.n_restarts_optimizer > 0:
            if not np.isfinite(self.dimensions).all():
                raise ValueError(
                    "Multiple optimizer restarts (n_restarts_optimizer>0) "
                    "requires that all bounds are finite.")
            n_points = 10000
            X_initial = \
                np.random.uniform(self.dimensions[:, 0], self.dimensions[:, 1],
                                  size=(n_points, len(self.dimensions[:,0])))
            values = self.acq_func(X_initial, self.surrogate_model)
            x0 = X_initial[np.argsort(values)[-self.n_restarts_optimizer:]]
            for i, x_i in enumerate(x0):
                optima_X[i], optima_acq_func[i] = \
                    self._constrained_optimization(obj_func, x_i,
                                                    self.dimensions)
            # Select result from run with minimal objective function
            # (minimum/maximum acquisition function depending on the settings)
            max_pos = np.argmax(optima_acq_func)
            next_x = optima_X[max_pos]
            next_x = np.clip(next_x, self.dimensions[:, 0], self.dimensions[:, 1])

            # Inverse transform the point(s) and turn bto back on
            if self.bto_for_acquisition:
                self.surrogate_model.normalize_bounds = True
                self.dimensions = self._dimensions
                next_x = self.surrogate_model.bto.inverse_transform(next_x)

            return next_x, -1 * optima_acq_func[max_pos]

        # Inverse transform the point(s) and turn bto back on
        if self.bto_for_acquisition:
            self.surrogate_model.normalize_bounds = True
            self.dimensions = self._dimensions
            next_x = self.surrogate_model.bto.inverse_transform(next_x)

        return optima_X[0], optima_acq_func[0]

    def kl_divergence(self):
        """Calculate the Kullback-Liebler (KL) divergence between different steps of the GP acquisition.
        Here the KL divergence is used as a convergence criterion for the GP. The KL-Divergence assumes a 
        multivariate normal distribution as underlying likelihood. Thus it may perform strangely when
        applied to some sort of weird likelihood.
        
        This function approximates the KL divergence by using the training samples weighted by their
        Likelihood values to get an estimate for the mean and covariance matrix along each dimension. The
        training data is taken internally from the surrogate model.
    
        ..note::
            The KL divergence is the difference between the evaluations of the last call of this function
            and the current data.

        Returns
        -------

        KL_divergence : The value of the KL divergence
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('error') # Raise exception for all warnings to catch them.

            # First try to calculate the mean and covariance matrix
            try:
                # Save mean and cov for KL divergence
                last_mean = np.copy(self.mean_)
                last_cov = np.copy(self.cov)

                # Get training data from surrogate and preprocess if neccessary
                X_train = self.surrogate_model._X_train_
                y_train = self.surrogate_model.predict(X_train)
                y_train = np.exp(y_train - np.max(y_train)) # Turn into unnormalized probability

                # Calculate mean and cov for KL div and to fit the transformation
                self.mean_ = np.average(X_train, axis=0, weights=y_train)
                self.cov = np.cov(X_train.T, aweights=y_train)
                last_cov_inv = np.linalg.inv(last_cov)
                kl = 0.5 * (np.log(det(last_cov)) - np.log(det(self.cov)) - X_train.shape[-1]+\
                            tr(last_cov_inv@self.cov)+(last_mean-self.mean_).T @ last_cov_inv @ (last_mean-self.mean_))
                # self.cov=np.atleast_2d([self.cov])

            except Exception as e:
                print("KL divergence can't be calculated because:")
                print(e)
                kl = np.nan
            
            return kl
    
    def kl_divergence_alternative(self, n_points = 20000):
        """Calculate the Kullback-Liebler (KL) divergence between different steps of the GP acquisition.
        Here the KL divergence is used as a convergence criterion for the GP. The KL-Divergence assumes a 
        multivariate normal distribution as underlying likelihood. Thus it may perform strangely when
        applied to some sort of weird likelihood.
        
        This function approximates the KL divergence by drawing n samples and calculating the cov from this.
    
        ..note::
            The KL divergence is the difference between the evaluations of the last call of this function
            and the current data.

        Returns
        -------

        KL_divergence : The value of the KL divergence
        """

        with warnings.catch_warnings():
            warnings.filterwarnings('error') # Raise exception for all warnings to catch them.

            # First try to calculate the mean and covariance matrix
            try:
                # Save mean and cov for KL divergence
                last_mean = np.copy(self.mean_)
                last_cov = np.copy(self.cov)

                # Get training data from surrogate and preprocess if neccessary
                X_train = np.random.uniform(self.dimensions[:,0], self.dimensions[:,1], 
                                        (n_points,len(self.dimensions[:,0])))
                y_train = self.surrogate_model.predict(X_train)
                y_train = np.exp(y_train - np.max(y_train)) # Turn into unnormalized probability
                y_train = y_train / np.mean(y_train)

                # Calculate mean and cov for KL div and to fit the transformation
                self.mean_ = np.average(X_train, axis=0, weights=y_train)
                self.cov = np.cov(X_train.T, aweights=y_train)
                last_cov_inv = np.linalg.inv(last_cov)
                kl = 0.5 * (np.log(det(last_cov)) - np.log(det(self.cov)) - X_train.shape[-1]+\
                            tr(last_cov_inv@self.cov)+(last_mean-self.mean_).T @ last_cov_inv @ (last_mean-self.mean_))
                # self.cov=np.atleast_2d([self.cov])

            except Exception as e:
                print("KL divergence can't be calculated because:")
                print(e)
                kl = np.nan
            
            return kl

    def _constrained_optimization(self, obj_func, initial_theta, bounds):

        if self.acq_optimizer == "fmin_l_bfgs_b":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                opt_res = scipy.optimize.fmin_l_bfgs_b(
                    obj_func, initial_theta, args={"eval_gradient": True}, bounds=bounds,
                    approx_grad=False, maxiter=200) 
                theta_opt, func_min = opt_res[0], opt_res[1]
        elif self.acq_optimizer == "sampling":
            opt_res = scipy.optimize.minimize(
                obj_func, initial_theta, args=(False), method="Powell", bounds=bounds)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.acq_optimizer):
            theta_opt, func_min = \
                self.acq_optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.acq_optimizer)
        
        return theta_opt, func_min
