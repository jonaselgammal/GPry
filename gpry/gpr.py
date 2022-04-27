# general purpose stuff
import warnings
from copy import deepcopy
from operator import itemgetter

# numpy and scipy
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
import scipy.optimize

# gpry kernels and SVM
from gpry.kernels import RBF, Matern, ConstantKernel as C
from gpry.svm import SVM
from gpry.preprocessing import Normalize_bounds, Normalize_y

# sklearn GP and kernel utilities
from sklearn.gaussian_process import GaussianProcessRegressor \
    as sk_GaussianProcessRegressor
from sklearn.base import clone, BaseEstimator as BE

# sklearn utilities
from sklearn.utils import check_random_state
from sklearn.utils.optimize import _check_optimize_result
from sklearn.utils.validation import check_array


class GaussianProcessRegressor(sk_GaussianProcessRegressor, BE):
    """
    GaussianProcessRegressor (GPR) that allows dynamic expansion.

    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.

    In addition to standard scikit-learn estimator API,
    GaussianProcessRegressor:

       * allows prediction without prior fitting (based on the GP prior).
       * implements a pipeline for pretransforming data before fitting.
       * provides the method :meth:`append_to_data` which allows to append
         additional data points to an already existing GPR. This is done either
         by refitting the hyperparameters (theta) or alternatively by using the
         Matrix inversion Lemma to keep the hyperparameters fixed.
       * overwrites the (hidden) native deepcopy function. This enables copying
         the GPR as well as the sampled points it contains.

    Parameters
    ----------

    kernel : kernel object, optional (default: "RBF")
        The kernel specifying the covariance function of the GP. If
        "RBF"/"Matern" is passed, the asymmetric kernel::

            ConstantKernel(1, [0.001, 10000]) * RBF/Matern([0.01]*n_dim, "dynamic")

        is used as default where ``n_dim`` is the number of dimensions of the
        training space. In this case you will have to provide the prior bounds
        as the ``bounds`` parameter to the GP. For the Matern kernel
        :math:`\nu=3/2`. Note that the kernel's hyperparameters are optimized
        during fitting.

    noise_level : float or array-like, optional (default: 1e-5)
        Square-root of the value added to the diagonal of the kernel matrix
        during fitting. Larger values correspond to increased noise level in the
        observations and reduce potential numerical issue during fitting.
        If an array is passed, it must have the same number of entries as the
        data used for fitting and is used as datapoint-dependent noise level.
        Note that this is equivalent to adding a WhiteKernel with c=noise_level.

    optimizer : str or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer : int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.

    preprocessing_X : X-preprocessor, Pipeline_X, optional (default: None)
        Single preprocessor or pipeline of preprocessors for X. If None is
        passed the data is not preprocessed. The `fit` method of the
        preprocessor is only called when the GP's hyperparameters are refit.

    preprocessing_y : y-preprocessor or Pipeline_y, optional (default: None)
        Single preprocessor or pipeline of preprocessors for y. If None is
        passed the data is not preprocessed.The `fit` method of the preprocessor
        is only called when the GP's hyperparameters are refit.

    account_for_inf : SVM, None or "SVM" (default: "SVM")
        Uses a SVM (Support Vector Machine) to classify the data into finite and
        infinite values. This allows the GP to express values of -inf in the
        data (unphysical values). If all values are finite the SVM will just
        pass the data through itself and do nothing.

    bounds : array-like, shape=(n_dims,2), optional
        Array of bounds of the prior [lower, upper] along each dimension. Has
        to be provided when the kernel shall be built automatically by the GP.

    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : int or `numpy RandomState <https://numpy.org/doc/stable/reference/random/legacy.html?highlight=randomstate#numpy.random.RandomState>`_, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    verbose : 1, 2, 3, optional (default: 1)
        Level of verbosity of the GP. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise.

    Attributes
    ----------

    X_train : array-like, shape = (n_samples, n_features)
        Original (untransformed) feature values in training data. Intended to be
        used when one wants to access the training data for any purpose.

    y_train : array-like, shape = (n_samples, [n_output_dims])
        Original (untransformed) target values in training data. Intended to be
        used when one wants to access the training data for any purpose.

    X_train_ : array-like, shape = (n_samples, n_features)
        (Possibly transformed) feature values in training data
        (also required for prediction). Mostly intended for internal use.

    y_train_ : array-like, shape = (n_samples, [n_output_dims])
        (Possibly transformed) target values in training data
        (also required for prediction). Mostly intended for internal use.

    noise_level : array-like, shape = (n_samples, [n_output_dims]) or scalar
        The noise level (square-root of the variance) of the uncorrelated
        training data. This is un-transformed.

    noise_level_ : array-like, shape = (n_samples, [n_output_dims]) or scalar
        The transformed noise level (if y is preprocessed)

    alpha : array-like, shape = (n_samples, [n_output_dims]) or scalar
        The value which is added to the diagonal of the kernel. This is the
        square of `noise_level`.

    kernel_ : :mod:`kernels` object
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters.

    K_inv_ : array-like, shape = (n_samples, n_samples)
        The inverse of the Kernel matrix of the training data. Needed at
        prediction.

    alpha_ : array-like, shape = (n_samples, n_samples)
        **Not to be confused with alpha!** The inverse Kernel matrix of the
        training points multiplied with ``y_train_`` (Dual coefficients of
        training data points in kernel space). Needed at prediction.
        .

    L_ : array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``

        .. warning::

            ``L_`` is not recomputed when using the append_to_data method
            without refitting the hyperparameters. As only ``K_inv_`` and
            alpha_ are used at prediction this is not neccessary.

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``


    **Methods:**

    .. autosummary::
        :toctree: stubs

        append_to_data
        fit
        _update_model
        predict
    """

    def __init__(self, kernel="RBF", noise_level=1e-2,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 preprocessing_X=None, preprocessing_y=None,
                 account_for_inf="SVM", bounds=None,
                 copy_X_train=True, random_state=None,
                 verbose=1):
        self.newly_appended = 0

        self.preprocessing_X = preprocessing_X
        self.preprocessing_y = preprocessing_y

        self.noise_level = noise_level

        self.n_eval = 0

        self.y_max = -np.inf

        self.verbose = verbose

        # Initialize SVM if given
        if account_for_inf=="SVM":
            self.account_for_inf = SVM()
        else:
            self.account_for_inf = account_for_inf

        self.bounds = bounds

        # Auto-construct inbuilt kernels
        if isinstance(kernel, str):
            if self.bounds is None:
                raise ValueError("You selected used the automatically "
                                 f"constructed '{kernel}' kernel without "
                                 "specifying prior bounds.")
            # Transform prior bounds if neccessary
            if self.preprocessing_X is not None:
                self.bounds_ = self.preprocessing_X.transform_bounds(self.bounds)
            else:
                self.bounds_ = self.bounds
            # Check if it's a supported kernel
            if kernel not in ["RBF", "Matern"]:
                raise ValueError("Currently only 'RBF' and 'Matern' are "
                                 f"supported as standard kernels. Got {kernel}")
            # Build kernel
            elif kernel == "RBF":
                kernel = C(1.0, [0.001, 10000]) \
                    * RBF([0.01] * self.d, "dynamic",
                          prior_bounds=self.bounds_)
            elif kernel == "Matern":
                kernel = C(1.0, [0.001, 10000]) \
                    * Matern([0.01] * self.d, "dynamic",
                             prior_bounds=self.bounds_)

        super(GaussianProcessRegressor, self).__init__(
            kernel=kernel, alpha=noise_level**2., optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=False, copy_X_train=copy_X_train,
            random_state=random_state)

        if self.verbose == 3:
            print("Initializing GP with the following options:")
            print("===========================================")
            print("Kernel:")
            if kernel == "RBF":
                print("ConstantKernel(1, [0.001, 10000])"
                      " * RBF([0.01]*n_dim, 'dynamic')")
            else:
                print(self.kernel.hyperparameters)
            print("Optimizer restarts: %i" % n_restarts_optimizer)
            print("X-preprocessor: %s" % (preprocessing_X is not None))
            print("y-preprocessor: %s" % (preprocessing_y is not None))
            print("SVM to account for infinities: %s"
                  % (account_for_inf is not None))

    @property
    def n(self):
        """Number of points in the training set."""
        return self.X_train.shape[0]

    @property
    def d(self):
        """Dimension of the feature space."""
        if self.bounds is None:
            return self.X_train.shape[1]
        else:
            return self.bounds.shape[0]

    @property
    def n_total_evals(self):
        """
        Extracts the total number of posterior evaluations (finite AND infinite)
        from the gp.
        """
        if self.account_for_inf is None:
            if hasattr(self, "y_train"):
                n_evals = len(self.y_train)
            else:
                n_evals = 0
        else:
            if hasattr(self.account_for_inf, "y_train"):
                n_evals = len(self.account_for_inf.y_train)
            elif hasattr(self, "y_train"):
                n_evals = len(self.y_train)
            else:
                n_evals = 0
        return n_evals

    @property
    def n_accepted_evals(self):
        """
        Extracts the number of accepted posterior evaluations (accepted means being
        classified as *finite* by the GP).
        """
        if hasattr(self, "y_train"):
            n_evals = len(self.y_train)
        else:
            n_evals = 0
        return n_evals

    def append_to_data(self, X, y, noise_level=None, fit=True, simplified_fit=False):
        """Append newly acquired data to the GP Model and updates it.

        Here updating refers to the re-calculation of
        :math:`(K(X,X)+\sigma_n^2 I)^{-1}` which is needed for predictions. In
        most cases (except if :math:`f(x_*)=\mu(x_*)`) the hyperparameters
        :math:`\theta` need to be updated too though. Therefore the function
        offers two different methods of updating the GPR after the training data
        (``X_train, y_train``) has been updated:

           * Refit :math:`\\theta` using the
             internal ``fit`` method.
           * Keep :math:`\\theta` fixed and update
             :math:`(K(X,X)+\sigma_n^2 I)^{-1}` using the blockwise matrix
             inversion lemma.

        While the first method can always be applied it is considerably slower
        than the second one. Therefore it can be useful to use the second
        algorithm in cases where it is worth saving the computational expense
        such as when performing parallelized active sampling.

         .. note::

            The second method can only be used if the GPR has previously been
            trained.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data to append to the model.

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values to append to the data

        noise_level : array-like, shape = (n_samples, [n_output_dims])
            Uncorrelated standard deviations to add to the diagonal part of the
            covariance matrix. Needs to have the same number of entries as
            y. If None, the noise_level set in the instance is used. If you pass
            a single number the noise level will be overwritten. In this case it
            is advisable to refit the hyperparameters of the kernel.

        fit : Bool, optional (default: True)
            Whether the model is refit to new :math:`\\theta`-parameters
            or just updated using the blockwise matrix-inversion lemma.

        Returns
        -------
        self
            Returns an instance of self.
        """

        if self.copy_X_train:
            X = np.copy(X)
            y = np.copy(y)

        # Check if X_train and y_train exist to see if a model
        # has previously been fit to the data
        if not (hasattr(self, "X_train_") and hasattr(self, "y_train_")):
            if not fit and self.verbose > 1:
                warnings.warn("No model has previously been fit to the data, "\
                    "a model will be fit with X and y instead of just "\
                    "updating with the same kernel hyperparameters")

            self.fit(X, y, noise_level=noise_level)
            self.y_max = np.max(y)
            return self

        if self.account_for_inf is not None:
            finite = self.account_for_inf.append_to_data(X, y,
                fit_preprocessors=fit)
            # If all added values are infinite there's no need to refit the GP
            if np.all(~finite):
                return self
            print("gpr.py append :: X,y,finite = ",(["%.20e"%a for a in X[0]] if len(X)==1 else X),y,finite)
            X = X[finite]
            y = y[finite]
            if np.iterable(noise_level) and noise_level is not None:
                noise_level = noise_level[finite]

        # Update noise_level, this is a bit complicated because it can be
        # a number, iterable or None. The noise level is also transformed here
        # if the data is not refit.
        if np.iterable(noise_level):
            if noise_level.shape[0] != y.shape[0]:
                raise ValueError("noise_level must be an array with same "\
                    "number of entries as y.(%d != %d)"
                                 % (noise_level.shape[0], y.shape[0]))
            elif np.iterable(self.noise_level):
                self.noise_level = np.append(self.noise_level,
                    noise_level, axis=0)
                if not fit:
                    if self.preprocessing_y is not None:
                        noise_level_transformed = self.preprocessing_y.\
                            transform_noise_level(noise_level)
                    else:
                        noise_level_transformed = noise_level
                    self.noise_level_ = np.append(self.noise_level_,
                        noise_level_transformed, axis=0)
            else:
                if self.verbose > 1:
                    warnings.warn(
                        "A new noise level has been assigned to the "
                        "updated training set while the old training set has a"
                        " single scalar noise level: %s" % self.noise_level)
                noise_level = np.append(np.ones(self.y_train_.shape) * \
                    self.noise_level, noise_level, axis=0)
                self.noise_level = noise_level
                if not fit:
                    if self.preprocessing_y is not None:
                        noise_level_transformed = self.preprocessing_y.\
                            transform_noise_level(noise_level)
                    else:
                        noise_level_transformed = noise_level
                    self.noise_level_ = noise_level_transformed
        elif noise_level is None:
            if np.iterable(self.noise_level):
                raise ValueError("No value for the noise level given even "\
                    "though concrete values were given earlier. Please only "\
                    "give one scalar value or a different one for each "\
                    "training point.")
        elif isinstance(noise_level, int) or isinstance(noise_level, float):
            if not fit and self.verbose > 1:
                warnings.warn("Overwriting the noise level with a scalar "\
                    "without refitting the kernel's hyperparamters. This may "\
                    "cause unwanted behaviour.")
                if self.preprocessing_y is not None:
                    noise_level_transformed = \
                        self.preprocessing_y.transform_noise_level(noise_level)
                else:
                    noise_level_transformed = noise_level
                self.noise_level_ = noise_level_transformed
            self.noise_level = noise_level
        else:
            raise ValueError("noise_level needs to be an iterable, number or "\
                "None, not %s"%noise_level)

        self.X_train = np.append(self.X_train, X, axis=0)
        self.y_train = np.append(self.y_train, y)
        self.y_max = max(self.y_max, np.max(y))

        # The number of newly added points. Used for the update_model method
        self.newly_appended = y.shape[0]

        if fit:
            self.fit(simplified=simplified_fit)
        else:
            if self.preprocessing_X is not None:
                X = self.preprocessing_X.transform(X)
            if self.preprocessing_y is not None:
                y = self.preprocessing_y.transform(y)

            self.X_train_ = np.append(self.X_train_, X, axis=0)
            self.y_train_ = np.append(self.y_train_, y, axis=0)
            self.alpha = self.noise_level_**2.

            self._update_model()

        return self

    def remove_from_data(self, position, fit=True):
        """Removes data points from the GP model. Works very similarly to the
        :meth:`append_to_data` method with the only difference being that the
        position(s) of the training points to delete are given instead of
        values.

        Parameters
        ----------
        position : int or array-like, shape = (n_samples,)
            The position (index) at which to delete from the training data.
            If an array is given the data is deleted at multiple points.

        fit : Bool, optional (default: True)
            Whether the model is refit to new :math:`\\theta`-parameters
            or just updated.

        Returns
        -------
        self
            Returns an instance of self.
        """
        if not (hasattr(self, "X_train_") and hasattr(self, "y_train_")):
            raise ValueError("GP model contains no points. Cannot remove "\
                "points which do not exist.")

        if np.iterable(position):
            if np.max(position) >= len(self.y_train_):
                raise ValueError("Position index is higher than length of "\
                    "training points")

        else:
            if position >= len(self.y_train_):
                raise ValueError("Position index is higher than length of "\
                    "training points")

        self.X_train_ = np.delete(self._X_train_, position, axis=0)
        self.y_train_ = np.delete(self._y_train_, position)
        self.y_max = np.max(self._y_train_)
        if np.iterable(self.noise_level):
            self.noise_level = np.delete(self.noise_level, position)
            self.noise_level_ = np.delete(self.noise_level_, position)
            self.alpha = np.delete(self.alpha, position)

        if fit:
            self.fit(self.X_train_, self.y_train_)

        else:
            # Precompute quantities required for predictions which are
            # independent of actual query points
            K = self.kernel_(self.X_train_)
            K[np.diag_indices_from(K)] += self.alpha
            try:
                self.L_ = cholesky(K, lower=True)  # Line 2
                # self.L_ changed, self.K_inv_ needs to be recomputed
                L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
                self.K_inv_ = L_inv.dot(L_inv.T)
            except np.linalg.LinAlgError as exc:
                exc.args = ("The kernel, %s, is not returning a "\
                            "positive definite matrix. Try gradually "\
                            "increasing the 'noise_level' parameter of your "\
                            "GaussianProcessRegressor estimator."
                            % self.kernel_) + exc.args
                raise
            self.alpha_ = self.K_inv_ @ self.y_train_
            # Just here if stuff doesnt work, needs to be removed later...
            #self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3

        return self

    def fit(self, X=None, y=None, noise_level=None, simplified = False):
        """Optimizes the hyperparameters :math:`\\theta` for the training data
        given. The algorithm used to perform the optimization is very similar
        to the one provided by Scikit-learn. The only major difference is, that
        gradient information is used in addition to the values of the
        marginalized log-likelihood.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features), optional
            (default: None)
            Training data. If None is given X_train is
            taken from the instance. None should only be passed if it is called
            from the ``append to data`` method.

        y : array-like, shape = (n_samples, [n_output_dims]), optional
            (default: None)
            Target values. If None is given y_train is taken from the instance.
            None should only be passed if it is called from the
            ``append to data`` method.

        Returns
        -------
        self
        """

        if not hasattr(self, 'kernel_'):
            self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        # If X and y are not given and are taken from the model instance
        if X is None or y is None:
            # Check if X AND y are None, else raise ValueError
            if X is not None or y is not None:
                raise ValueError("X or y is None, while the other isn't. "\
                    "Either both need to be provided or both should be None")
            if noise_level is not None:
                raise ValueError("Cannot give a noise level if X and y are "\
                    "not given.")

            # Take X and y from model. We assume that all infinite values have
            # been removed by the append to data method
            X = np.copy(self.X_train)
            y = np.copy(self.y_train)
            noise_level = np.copy(self.noise_level)

        # If X and y are given
        else:

            # Set noise level
            if noise_level is None:
                if np.iterable(self.noise_level):
                    raise ValueError("If None is passed as noise level the "
                                     "internally saved noise level needs "
                                     "to be a scalar")
                noise_level = self.noise_level
            else:
                self.noise_level = noise_level
            if np.iterable(noise_level) and noise_level.shape[0] != y.shape[0]:
                raise ValueError("noise_level must be a scalar or an array "
                                 "with same number of entries as y.(%s "
                                 "!= %s)"
                                 % (noise_level.shape[0], y.shape[0]))

            # Account for infinite values
            if self.account_for_inf is not None:
                finite = self.account_for_inf.fit(X, y)
                # If there are no finite values there's no point
                # in fitting the GP.
                if np.all(~finite):
                    return self
                X = X[finite]
                y = y[finite]
                if np.iterable(noise_level):
                    noise_level = noise_level[finite]

            # Copy X and y for later use
            self.X_train = np.copy(X)
            self.y_train = np.copy(y)

        # Transform data and noise level
        if self.preprocessing_X is not None:
            self.preprocessing_X.fit(X, y)
            X_transformed = self.preprocessing_X.transform(X)
        else:
            X_transformed = X

        if self.preprocessing_y is not None:
            self.preprocessing_y.fit(X, y)
            y_transformed = self.preprocessing_y.transform(y)
            self.noise_level_ = self.preprocessing_y.\
                transform_noise_level(noise_level)
        else:
            y_transformed = y
            self.noise_level_ = noise_level

        self.X_train_ = X_transformed
        self.y_train_ = y_transformed
        # Set alpha for the inbuilt fit function
        self.alpha = self.noise_level_**2.

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta,
                                                         clone_kernel=False)

            print("FINDING FIRST OPTIMUM")
            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel_.theta,
                                                      self.kernel_.bounds))]

            print("FINDING ADDITIONAL OPTIMA",(self.n_restarts_optimizer if not simplified else 1))
            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0 and not simplified:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = np.copy(self.kernel_.bounds)
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            # self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta,
                                             clone_kernel=False)

        print("ALL OPTIMA FOUND")
        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self.K_inv_ needs to be recomputed
            L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
            self.K_inv_ = L_inv.dot(L_inv.T)
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = self.K_inv_ @ self.y_train_
        print("DONE FITTING")
        # leave this here if stuff doesnt work...
        # self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3

        # Reset newly_appended to 0
        self.newly_appended = 0

        return self

    def _update_model(self):
        """Updates a preexisting model using the matrix inversion lemma.

        This method is used when a refitting of the :math:`\\theta`-parameters
        is not needed. In this case only the Inverse of the Covariance matrix
        is updated. This method does not take X or y as inputs and should only
        be called from the append_to_data method.

        The X and y values used for training are taken internally from the
        instance. The method used to update the covariance matrix relies on
        updating it by using the blockwise matrix inversion lemma in order to
        reduce the computational complexity from :math:`n^3` to
        :math:`n^2\\cdot m`.

        Returns
        -------
        self
        """

        # Check if a model has previously been fit to the data, i.e. that a
        # K_inv_matrix, X_train_ and y_train_ exist. Furthermore check, that
        # newly_appended > 0.

        if self.newly_appended < 1:
            raise ValueError("No new points have been appended to the model. "
                "Please append points with the 'append_to_data'-method before "
                "trying to update.")

        if getattr(self, "X_train_", None) is None:
            raise ValueError("X_train_ is missing. Most probably the model "
                "hasn't been fit to the data previously.")

        if getattr(self, "y_train_", None) is None:
            raise ValueError("y_train_ is missing. Most probably the model "\
                "hasn't been fit to the data previously.")

        if getattr(self, "K_inv_", None) is None:
            raise ValueError("K_inv_ is missing. Most probably the model "\
                "hasn't been fit to the data previously.")

        if self.K_inv_.shape[0] != self.y_train_.size - self.newly_appended:
            raise ValueError("The number of added points doesn't match the "\
                "dimensions of the K_inv matrix. %s != %s"
                %(self.K_inv_.shape[0], self.y_train_.size-self.newly_appended))

        # Define all neccessary variables
        K_inv = self.K_inv_
        X_1 = self.X_train_[:-self.newly_appended]
        X_2 = self.X_train_[-self.newly_appended:]

        # Get the B, C and D matrices
        K_XY = self.kernel_(X_1, X_2)
        K_YY = self.kernel_(X_2)

        # Add the alpha value to the diagonal part of the matrix
        if np.iterable(self.alpha):
            K_YY[np.diag_indices_from(K_YY)] += \
                self.alpha[-self.newly_appended:]
        else:
            K_YY[np.diag_indices_from(K_YY)] += self.alpha

        # Inserting the new piece which uses the blockwise inversion lemma
        # C * A^{-1}
        gamma = K_XY.T @ K_inv
        # (D - C*A^{-1}*B)^{-1} in 1D
        alpha = np.linalg.inv(K_YY - gamma @ K_XY)
        # Off-Diag. Term
        beta = alpha @ gamma
        # Put all together
        self.K_inv_ = np.block([[(K_inv + K_inv @ K_XY @ beta), -1*beta.T],
                                [-1*beta                      , alpha]])

        # OVERWRITING THE ABOVE CODE , JUST FOR DEBUGGINGGGG!!!!!!!!
        k = self.kernel_(self.X_train_)
        k[np.diag_indices_from(k)] += self.alpha
        self.K_inv_ = np.linalg.inv(k)
        #self.K_inv_ = np.linalg.inv(self.kernel_(self.X_train_))

        # Also update alpha_ matrix
        self.alpha_ = self.K_inv_ @ self.y_train_

        # Reset newly_appended to 0
        self.newly_appended = 0
        return self

    def predict(self, X, return_std=False, return_cov=False,
                return_mean_grad=False, return_std_grad=False, do_check_array=True,doprint=False):
        """
        Predict output for X.

        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True) or covariance (return_cov=True),
        the gradient of the mean and the standard-deviation with respect to X
        can be optionally provided.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated.

        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_mean_grad : bool, default: False
            Whether or not to return the gradient of the mean.
            Only valid when X is a single point.

        return_std_grad : bool, default: False
            Whether or not to return the gradient of the std.
            Only valid when X is a single point.

        .. note::

            Note that in contrast to the sklearn GP Regressor our implementation
            cannot return the full covariance matrix. This is to save on some
            complexity and since the full covariance cannot be calculated if
            either values are infinite or a y-preprocessor is used.

        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points

        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.

        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.

        y_mean_grad : shape = (n_samples, n_features), optional
            The gradient of the predicted mean.

        y_std_grad : shape = (n_samples, n_features), optional
            The gradient of the predicted std.
        """
        self.n_eval += len(X)

        if return_std_grad and not (return_std and return_mean_grad):
            raise ValueError(
                "Not returning std_gradient without returning "
                "the std and the mean grad.")

        if X.shape[0] != 1 and (return_mean_grad or return_std_grad):
            raise ValueError("Mean grad and std grad not implemented \
                for n_samples > 1")

        if do_check_array and (self.kernel is None or self.kernel.requires_vector_input):
            X = check_array(X, ensure_2d=True, dtype="numeric")
        elif do_check_array:
            X = check_array(X, ensure_2d=False, dtype=None)

        if not hasattr(self, "X_train_"):  # Not fit; predict based on GP prior
            # we assume that since the GP has not been fit to data the SVM can
            # be ignored
            y_mean = np.zeros(X.shape[0])
            if return_std:
                y_var = self.kernel.diag(X)
                y_std = np.sqrt(y_var)
                if not return_mean_grad and not return_std_grad:
                    return y_mean, y_std
            if return_mean_grad:
                mean_grad = np.zeros_like(X)
                if return_std:
                    if return_std_grad:
                        std_grad = np.zeros_like(X)
                        return y_mean, y_std, mean_grad, std_grad
                    else:
                        return y_mean, y_std, mean_grad
                else:
                    return y_mean, mean_grad
            else:
                return y_mean

        # First check if the SVM says that the value should be -inf
        if self.account_for_inf is not None:
            # Every variable that ends in _full is the full (including infinite)
            # values
            X = np.copy(X)  # copy since we might change it
            n_samples = X.shape[0]
            n_dims = X.shape[1]
            # Initialize the full arrays for filling them later with infinite
            # and non-infinite values
            y_mean_full = np.ones(n_samples)
            y_std_full = np.zeros(n_samples)  # std is zero when mu is -inf
            grad_mean_full = np.ones((n_samples, n_dims))
            grad_std_full = np.zeros((n_samples, n_dims))
            finite = self.account_for_inf.predict(X)
            # If all values are infinite there's no point in running the
            # prediction through the GP
            if np.all(~finite):
                y_mean = y_mean_full * -np.inf
                if return_std:
                    y_std = np.zeros(n_samples)
                    if not return_mean_grad and not return_std_grad:
                        return y_mean, y_std
                if return_mean_grad:
                    grad_mean = np.ones((n_samples, n_dims)) * np.inf
                    if return_std:
                        if return_std_grad:
                            grad_std = np.zeros((n_samples, n_dims))
                            return y_mean, y_std, grad_mean, grad_std
                        else:
                            return y_mean, y_std, grad_mean
                    else:
                        return y_mean, grad_mean
                return y_mean

            y_mean_full[~finite] = -np.inf  # Set infinite values
            grad_mean_full[~finite] = np.inf  # the grad of inf values is +inf
            X = X[finite]  # only predict the finite samples

        if self.preprocessing_X is not None:
            X = self.preprocessing_X.transform(X)

        # Predict based on GP posterior
        K_trans = self.kernel_(X, self.X_train_)
        y_mean = K_trans.dot(self.alpha_)    # Line 4 (y_mean = f_star)
        # Undo normalization
        if self.preprocessing_y is not None:
            y_mean = self.preprocessing_y.inverse_transform(y_mean)
        # Put together with SVM predictions
        if self.account_for_inf is not None:
            y_mean_full[finite] = y_mean
            y_mean = y_mean_full

        if return_std:
            K_inv = self.K_inv_

            # Compute variance of predictive distribution
            y_var = self.kernel_.diag(X)
            y_var -= np.einsum("ki,kj,ij->k", K_trans, K_trans, K_inv)
            if doprint:
              print("K^T,K^-1,D_K = ",K_trans, K_inv, self.kernel_.diag(X))
              print("X-X_T, K^T*K^-1",np.where([np.allclose(X[0],self.X_train_[i]) for i in range(len(self.X_train_))]),np.where(np.abs(np.dot(K_trans,K_inv))>1e-5)[0],len(self.X_train_))
              print("THING = ",np.einsum("ki,kj,ij->k", K_trans, K_trans, K_inv),self.kernel_.diag(X),self.kernel_.diag(X)-np.einsum("ki,kj,ij->k", K_trans, K_trans, K_inv))
              #K_trans_inv = np.linalg.pinv(K_trans)
              #print("OTHERTHING = ",np.einsum("ki,ij,kj->k",K_trans,(np.einsum("ji,jk,kl->il",K_trans_inv,self.kernel_.diag(X),K_trans_inv)-K_inv),K_trans))
            # np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)
            # np.einsum("ki,kj,ij->k", K_trans, K_trans, K_inv)

            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                if self.verbose > 1:
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                y_var[y_var_negative] = 0.0
            y_std = np.sqrt(y_var)

            y_std_untransformed = np.copy(y_std)

            # Undo normalization
            if self.preprocessing_y is not None:
                y_std = self.preprocessing_y.\
                    inverse_transform_noise_level(y_std)
            # Add infinite values
            if self.account_for_inf is not None:
                y_std_full[finite] = y_std
                y_std = y_std_full

            if not return_mean_grad and not return_std_grad:
                return y_mean, y_std

        if return_mean_grad:
            grad = self.kernel_.gradient_x(X[0], self.X_train_)
            #print("grad:",grad)
            grad_mean = np.dot(grad.T, self.alpha_)
            # Undo normalization
            if self.preprocessing_y is not None:
                grad_mean = self.preprocessing_y.\
                    inverse_transform_noise_level(grad_mean)
            # Include infinite values
            if self.account_for_inf is not None:
                grad_mean_full[finite] = grad_mean
                grad_mean = grad_mean_full
            if return_std_grad:
                grad_std = np.zeros(X.shape[1])
                if not np.allclose(y_std, grad_std):
                    grad_std = -np.dot(K_trans,
                                       np.dot(K_inv, grad))[0] \
                                       / y_std_untransformed
                    # Undo normalization
                    if self.preprocessing_y is not None:
                        # Apply inverse transformation twice
                        grad_std = self.preprocessing_y.\
                            inverse_transform_noise_level(grad_std)
                        grad_std = self.preprocessing_y.\
                            inverse_transform_noise_level(grad_std)
                    # Include infinite values
                    if self.account_for_inf is not None:
                        grad_std_full[finite] = grad_std
                        grad_std = grad_std_full
                return y_mean, y_std, grad_mean, grad_std

            if return_std:
                return y_mean, y_std, grad_mean
            else:
                return y_mean, grad_mean

        return y_mean

    def __deepcopy__(self, memo):
        """
        Overwrites the internal deepcopy method of the class in order to
        also copy instance variables which are not defined in the init.
        """
        # Initialize the stuff specified in init
        c = GaussianProcessRegressor(
            kernel=self.kernel,
            noise_level=self.noise_level,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            preprocessing_X=self.preprocessing_X,
            preprocessing_y=self.preprocessing_y,
            copy_X_train=self.copy_X_train,
            bounds=self.bounds,
            random_state=self.random_state)

        # Initialize the X_train and y_train part
        if hasattr(self, "X_train"):
            c.X_train = self.X_train
        if hasattr(self, "y_train"):
            c.y_train = self.y_train
        if hasattr(self, "y_max"):
            c.y_max = self.y_max
        if hasattr(self, "X_train_"):
            c.X_train_ = self.X_train_
        if hasattr(self, "y_train_"):
            c.y_train_ = self.y_train_
        # Initialize noise levels
        if hasattr(self, "noise_level"):
            c.noise_level = self.noise_level
        if hasattr(self, "noise_level_"):
            c.noise_level_ = self.noise_level_
        if hasattr(self, "alpha"):
            c.alpha = self.alpha
        # Initialize kernel and inverse kernel
        if hasattr(self, "L_"):
            c.L_ = self.L_
        if hasattr(self, "alpha_"):
            c.alpha_ = self.alpha_
        if hasattr(self, "K_inv_"):
            c.K_inv_ = self.K_inv_
        if hasattr(self, "kernel_"):
            c.kernel_ = self.kernel_
        if hasattr(self, "account_for_inf"):
            c.account_for_inf = deepcopy(self.account_for_inf)
        return c

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.optimizer == "fmin_l_bfgs_b":
                opt_res = scipy.optimize.minimize(
                    obj_func, initial_theta, method="L-BFGS-B", jac=True,
                    bounds=bounds)
                # Temporarily disabled bc incompatibility between current sklearn+scipy
                # if self.verbose > 1:
                #     _check_optimize_result("lbfgs", opt_res)
                theta_opt, func_min = opt_res.x, opt_res.fun
            elif callable(self.optimizer):
                theta_opt, func_min = \
                    self.optimizer(obj_func, initial_theta, bounds=bounds)
            else:
                raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min
