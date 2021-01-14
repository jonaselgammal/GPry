
# general purpose stuff
import warnings
from operator import itemgetter

# numpy and scipy
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
import scipy.optimize

# gpry kernels and transformations
from gpry.kernels import RBF, ConstantKernel as C
from gpry.preprocessing import Whitening, Normalize_bounds

# sklearn GP and kernel utilities
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor as sk_GaussianProcessRegressor
from sklearn.base import clone, BaseEstimator as BE

# sklearn utilities
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

class GaussianProcessRegressor(sk_GaussianProcessRegressor, BE):
    """
    GaussianProcessRegressor (GPR) that allows dynamic expansion.

    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.

    In addition to standard scikit-learn estimator API,
    GaussianProcessRegressor:

       * allows prediction without prior fitting (based on the GP prior).
       * provides the method ``append_to_data(X,y)`` which allows to append additional
         data points to an already existing GPR. This is done either by refitting the
         hyperparameters (theta) or alternatively by using the Matrix inversion Lemma
         to keep the hyperparameters fixed.
       * overwrites the (hidden) native deepcopy function. This enables to copy the GPR
         as well as the sampled points it contains.

    Parameters
    ----------

    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel ``1.0 * RBF(1.0)`` is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.

    alpha : float or array-like, optional (default: 1e-10)
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations
        and reduce potential numerical issue during fitting. If an array is
        passed, it must have the same number of entries as the data used for
        fitting and is used as datapoint-dependent noise level. Note that this
        is equivalent to adding a WhiteKernel with c=alpha. Allowing to specify
        the noise level directly as a parameter is mainly for convenience and
        for consistency with Ridge.

    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
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

    normalize_y : boolean, optional (default: False)
        Whether the target values y are normalized, i.e., the mean of the
        observed target values become zero. This parameter should be set to
        True if the target values' mean is expected to differ considerable from
        zero. When enabled, the normalization effectively modifies the GP's
        prior based on the data, which contradicts the likelihood principle;
        normalization is thus disabled per default.

        ..warning::
            In addition to the aforementioned problem with the prior of the GP
            this function might interfere with the :meth:`append_to_data` method
            when using the matrix inversion lemma.
    
    whiten : boolean, optional (default: False)
        Whether the distribution of X-Values shall be transformed to resemble a unit
        multivariate normal distribution. This uses the internalPrincipal Component 
        Analysis (PCA) implementation of SKlearn.

    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    Attributes
    ----------

    X_train_ : array-like, shape = (n_samples, n_features)
        Feature values in training data (also required for prediction)

    y_train_ : array-like, shape = (n_samples, [n_output_dims])
        Target values in training data (also required for prediction)

    kernel_ : :mod:`kernels` object
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters

    K_inv_ : array-like, shape = (n_samples, n_samples)
        The inverse of the Kernel matrix of the training data. Needed at prediction

    alpha_ : array-like, shape = (n_samples, n_samples)
        **Not to be confused with alpha!** The inverse Kernel matrix of the
        training points multiplied with ``y_train_``. Needed at prediction.

    L_ : array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``

        .. warning::

            ``L_`` is not recomputed when using the append_to_data method without
            refitting the hyperparameters. As only K_inv_ and alpha_ are used at
            prediction this is not neccessary.
    
    alpha_ : array-like, shape = (n_samples,)
        Dual coefficients of training data points in kernel space

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

    def __init__(self, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, whiten=False, normalize_bounds=False, bounds=None,
                 copy_X_train=True, random_state=None):
        self.newly_appended = 0
        self.bounds = bounds

        self.normalize_bounds = normalize_bounds
        self.bto = Normalize_bounds(bounds)

        self.whiten = whiten
        self.pca = Whitening(bounds_normalized=normalize_bounds)

        super(GaussianProcessRegressor, self).__init__(
            kernel=kernel, alpha=alpha, optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y, copy_X_train=copy_X_train,
            random_state=random_state)

    def append_to_data(self, X, y, alpha=None, fit=True):
        """Append newly acquired data to the GP Model and updates it.

        Here updating refers to the re-calculation of :math:`(K(X,X)+\sigma_n^2 I)^{-1}`
        which is needed for predictions. In most cases (except if :math:`f(x_*)=\mu(x_*)`)
        the hyperparameters :math:`\theta` need to be updated too though. Therefore the function
        offers two different methods of updating the GPR after the training data (``X_train, y_train``)
        has been updated:

           * Refit :math:`\\theta` using the
             internal ``fit`` method.
           * Keep :math:`\\theta` fixed and update :math:`(K(X,X)+\sigma_n^2 I)^{-1}` using the blockwise
             matrix inversion lemma.
        
        While the first method can always be applied it is considerably slower than the second one. Therefore
        it can be useful to use the second algorithm in cases where it is worth saving the computational expense
        such as when performing parallelized active sampling.
         .. note::

            The second method can only be used if the GPR has previously been trained.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data to append to the model.

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values to append to the data
        
        alpha : array-like, shape = (n_samples, [n_output_dims])
            Uncorrelated variances to add to the diagonal part of the
            covariance matrix. Needs to have the same number of entries as
            y. If None, the alpha value set in the instance is used.

        fit : Bool, optional (default: True)
            Whether the model is refit to new :math:`\\theta`-parameters
            or just updated using the blockwise matrix-inversion lemma.

        Returns
        -------
        self
            Returns an instance of self.        
        """
        # Check if X_train and y_train exist to see if a model
        # has previously been fit to the data

        if self.copy_X_train:
            X = np.copy(X)
            y = np.copy(y)

        if not (hasattr(self, "X_train_") and hasattr(self, "y_train_")):
            if not fit:
                warnings.warn("No model has previously been fit to the data, "
                            "a model will be fit with X and y instead of just "
                            "updating with the same theta")
            self.fit(X, y)
            return self

        if self.kernel_.requires_vector_input:
            X, y = super()._validate_data(X, y, multi_output=True, y_numeric=True,
                                       ensure_2d=True, dtype="numeric")
        else:
            X, y = super()._validate_data(X, y, multi_output=True, y_numeric=True,
                                       ensure_2d=False, dtype=None)

        if np.iterable(alpha):
            if alpha.shape[0] != y.shape[0]:        
                raise ValueError("alpha must be an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (alpha.shape[0], y.shape[0]))
            elif np.iterable(self.alpha):
                self.alpha = np.append(self.alpha, alpha, axis=0)
            else:
                warnings.warn("A new alpha has been assigned to the updated training set"
                                " while the old training set has a single scalar alpha value:"
                                " %s"%self.alpha)
                self.alpha = np.append(np.ones(self.y_train_.shape)*self.alpha, alpha, axis=0)
        elif alpha is None:
            if np.iterable(self.alpha):
                raise ValueError("No value for alpha given even though concrete values"
                                 " were given earlier. Please only give one scalar value"
                                 " or a different one for each training point.")
        else:
            raise ValueError("alpha needs to be an iterable or None, not %s"%alpha)
            
        self._X_train_ = np.append(self._X_train_, X, axis=0)
        self._y_train_ = np.append(self._y_train_, y)
        
        # The number of newly added points. Used for the update_model method
        self.newly_appended = y.shape[0]

        if fit:
            self.fit()
        else:
            if self.normalize_bounds:
                X = self.bto.transform(X)
            if self.whiten:
                X = self.pca.transform(X)
            
            self.X_train_ = np.append(self.X_train_, X, axis=0)
            self.y_train_ = np.append(self.y_train_, y, axis=0)

            # Keep a copy of the original values around in case we want
            # to normalize again after adding or deleting points.
            self._update_model()

        return self
    
    def remove_from_data(self, position, fit=True):
        """Removes data points from the GP model. Works very similarly to the 
        :meth:`append_to_data` method with the only difference being that the 
        position(s) of the training points to delete are given instead of values.

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
            raise ValueError("GP model contains no points. Cannot remove points which " + \
                             "do not exist.")

        if np.iterable(position):
            if np.max(position) >= len(self.y_train_):
                raise ValueError("Position index is higher than length of training points")

        else:
            if position >= len(self.y_train_):
                raise ValueError("Position index is higher than length of training points")

        self.X_train_ = np.delete(self._X_train_, position, axis=0)
        self.y_train_ = np.delete(self._y_train_, position)
        if np.iterable(self.alpha):
            self.alpha = np.delete(self.alpha, position)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(self.y_train_, axis=0)
            self._y_train_std = np.std(self.y_train_, axis=0)

            # Remove mean and make unit variance
            self.y_train_ = (self._y_train_ - self._y_train_mean) / self._y_train_std

        else:
            self._y_train_mean = np.zeros(1)
            self._y_train_std = 1
        
        if fit:
            self.fit(self.X_train_, self.y_train_)

        else:
            # Precompute quantities required for predictions which are independent
            # of actual query points
            if self.whiten:
                self.X_train_ = self.pca.transform(self.X_train_)
            K = self.kernel_(self.X_train_)
            K[np.diag_indices_from(K)] += self.alpha
            try:
                self.L_ = cholesky(K, lower=True)  # Line 2
                # self.L_ changed, self._K_inv needs to be recomputed
                self._K_inv = None
            except np.linalg.LinAlgError as exc:
                exc.args = ("The kernel, %s, is not returning a "
                            "positive definite matrix. Try gradually "
                            "increasing the 'alpha' parameter of your "
                            "GaussianProcessRegressor estimator."
                            % self.kernel_,) + exc.args
                raise
            self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3

            # Precompute arrays needed at prediction
            L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
            self.K_inv_ = L_inv.dot(L_inv.T)

        # Fix deprecation warning #462
        if int(sklearn.__version__[2:4]) >= 19:
            self.y_train_mean_ = self._y_train_mean
        else:
            self.y_train_mean_ = self.y_train_mean
             
        return self

    def fit(self, X=None, y=None):
        """Optimizes the hyperparameters :math:`\\theta` for the training data given.
        The algorithm used to perform the optimization is very similar to the one provided
        by Scikit-learn. The only major difference is, that gradient information is used in addition
        to the values of the marginalized log-likelihood.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features), optional (default: None)
            Training data. If None is given X_train is
            taken from the instance.

        y : array-like, shape = (n_samples, [n_output_dims]), optional (default: None)
            Target values. If None is given y_train is taken from the instance.

        Returns
        -------
        self
            Returns an instance of self.
        """

        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds="fixed") \
                * RBF(1.0, length_scale_bounds="fixed")
        elif not hasattr(self, 'kernel_'):
            self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        # If X and y are not given and are taken from the model instance
        if X is None or y is None:
            # Check if X AND y are None, else raise ValueError
            if X is not None or y is not None:
                raise ValueError("X or y is None, while the other isn't."
                                 "Either both need to be provided or \
                                 both should be None")
            
            # Take X and y from model
            X = np.copy(self._X_train_) if self.copy_X_train else self._X_train_
            y = np.copy(self._y_train_) if self.copy_X_train else self._y_train_

            if self.normalize_bounds:
                X = self.bto.transform(X)

            if self.whiten:
                self.pca.fit(self)
                X = self.pca.transform(X)
            
            self.X_train_ = X
            self.y_train_ = y

        # If X and y are given
        else:

            # Copy X and y for later use
            self._X_train_ = np.copy(X) if self.copy_X_train else X
            self._y_train_ = np.copy(y) if self.copy_X_train else y

            # Copy X and y
            X = np.copy(X) if self.copy_X_train else X
            y = np.copy(y) if self.copy_X_train else y

            if self.normalize_bounds:
                X = self.bto.transform(X)
            
            if self.whiten:
                self.pca.fit(self)
                X = self.pca.transform(X)

            if np.iterable(self.alpha) \
            and self.alpha.shape[0] != y.shape[0]:
                if self.alpha.shape[0] == 1:
                    self.alpha = self.alpha[0]
                else:
                    raise ValueError("alpha must be a scalar or an array"
                                    " with same number of entries as y.(%d "
                                    "!= %d)"
                                    % (self.alpha.shape[0], y.shape[0]))

            self.X_train_ = X
            self.y_train_ = y

        super().fit(self.X_train_, self.y_train_)
        L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
        self.K_inv_ = L_inv.dot(L_inv.T)
        # Reset newly_appended to 0
        self.newly_appended = 0

        return self

    def _update_model(self):
        """Updates a preexisting model using the matrix inversion lemma

        This method is used when a refitting of the :math:`\\theta`-parameters is
        not needed. In this case only the Inverse of the Covariance matrix is updated.
        This method does not take X or y as inputs and should only be called from the
        append_to_data method.
        The X and y values used for training are taken internally from the instance.
        The method used to update the covariance matrix relies on updating it by using
        the blockwise matrix inversion lemma in order to reduce the computational complexity
        from :math:`n^3` to :math:`n^2\\cdot m`.

        Returns
        -------
        self
            Returns an instance of self.
        """

        # Check if a model has previously been fit to the data, i.e. that a K_inv_matrix,
        # X_train_ and y_train_ exist. Furthermore check, that newly_appended > 0.

        if self.newly_appended < 1:
            raise ValueError("No new points have been appended to the model. Please append"
                             " points with the 'append_to_data'-method before trying to update.")
        
        if getattr(self, "X_train_", None) is None:
            raise ValueError("X_train_ is missing. Most probably the model hasn't been"
                             " fit to the data previously.")

        if getattr(self, "y_train_", None) is None:
            raise ValueError("y_train_ is missing. Most probably the model hasn't been"
                             " fit to the data previously.")
        
        if getattr(self, "K_inv_", None) is None:
            raise ValueError("K_inv_ is missing. Most probably the model hasn't been"
                             " fit to the data previously.")
        
        if self.K_inv_.shape[0] != self.y_train_.size - self.newly_appended:
            raise ValueError("The number of added points doesn't match the dimensions of"
                             "the K_inv matrix. %s != %s"
                             %(self.K_inv_.shape[0], self.y_train_.size-self.newly_appended))

        # Normalize target value
        if self.normalize_y:
            # Remove mean and make unit variance
            self.y_train_[-self.newly_appended:] = (self._y_train_[-self.newly_appended:] - self._y_train_mean) / self._y_train_std

        # Define all neccessary variables
        K_inv = self.K_inv_
        X_1 = self.X_train_[:-self.newly_appended]
        X_2 = self.X_train_[-self.newly_appended:]

        # Get the B, C and D matrices
        K_XY = self.kernel_(X_1, X_2)
        K_YY = self.kernel_(X_2)
        
        # Add the alpha value to the diagonal part of the matrix
        if np.iterable(self.alpha):
            K_YY[np.diag_indices_from(K_YY)] += self.alpha[-self.newly_appended:]
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
        self.K_inv_ = np.block([[(K_inv + K_inv @ K_XY @ beta)       , -1*beta.T],
                                [-1*beta                             , alpha]])
        
        # Also update alpha_ matrix
        self.alpha_ = self.K_inv_ @ self.y_train_

        # Reset newly_appended to 0
        self.newly_appended = 0
        return self

    def predict(self, X, return_std=False, return_cov=False,
                return_mean_grad=False, return_std_grad=False):
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

        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.

        return_mean_grad : bool, default: False
            Whether or not to return the gradient of the mean.
            Only valid when X is a single point.

        return_std_grad : bool, default: False
            Whether or not to return the gradient of the std.
            Only valid when X is a single point.
        
        transform_bounds : bool, default: True
            Whether to sample in the true parameter space or the
            one where the priors are normalized to 1. This is needed 
            at training if the prior scales are vastly different.
            When running with the actual priors this should be set 
            to ``True``.

            .. note::

                This only has an effect if the ``normalize_bounds`` option
                is enabled in the GP Regressor

            .. warning::

                If this option is set to ``False`` please make sure to transform
                the bounds accordingly. The transformed bounds are stored in
                ``self.bto.transformed_bounds``.
        
        transform_PCA : bool, default: True
            Whether to sample in the true parameter space or the
            one where the priors are normalized to 1. This is needed 
            at training if the prior scales are vastly different.
            When running with the actual priors this should be set 
            to ``True``.

            .. note::

                This only has an effect if the ``whiten`` option
                is enabled in the GP Regressor.

            .. warning::

                If this option is set to ``False`` please make sure to transform
                the bounds accordingly. The transformation has not been implemented
                yet.

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
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        if return_std_grad and not return_std:
            raise ValueError(
                "Not returning std_gradient without returning "
                "the std.")

        X = check_array(X)

        if self.normalize_bounds:
            X = self.bto.transform(X)

        if self.whiten and hasattr(self.pca, "evals"):
            X = self.pca.transform(X)

        if X.shape[0] != 1 and (return_mean_grad or return_std_grad):
            raise ValueError("Mean grad and std grad not implemented for n_samples > 1")

        if not hasattr(self, "X_train_"):  # Not fit; predict based on GP prior
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = self.kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = self.kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

        else:  # Predict based on GP posterior
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha_)    # Line 4 (y_mean = f_star)

            # Undo normalization
            y_mean = self._y_train_std * y_mean + self._y_train_mean

            if return_cov:
                v = self.K_inv_ @ K_trans.T # Line 5
                y_cov = self.kernel_(X) - K_trans @ v # Line 6

                # Undo normalization
                y_cov = y_cov * self._y_train_std**2.
                
                if not return_mean_grad and not return_std_grad:
                    return y_mean, y_cov

            elif return_std:
                K_inv = self.K_inv_

                # Compute variance of predictive distribution
                y_var = self.kernel_.diag(X)
                y_var -= np.einsum("ki,kj,ij->k", K_trans, K_trans, K_inv)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0
                
                # Undo normalization
                y_std = np.sqrt(y_var) * self._y_train_std

                if not return_mean_grad and not return_std_grad:
                    return y_mean, np.sqrt(y_var)             

            if return_mean_grad:
                grad = self.kernel_.gradient_x(X[0], self.X_train_)
                grad_mean = np.dot(grad.T, self.alpha_)
                grad_mean = grad_mean * self._y_train_std
                if return_std_grad:
                    grad_std = np.zeros(X.shape[1])
                    if not np.allclose(y_std, grad_std):
                        grad_std = -np.dot(K_trans,
                                           np.dot(K_inv, grad))[0] / y_std
                        grad_std = grad_std * self._y_train_std**2
                    return y_mean, y_std, grad_mean, grad_std

                if return_std:
                    return y_mean, y_std, grad_mean
                else:
                    return y_mean, grad_mean

            else:
                if return_std:
                    return y_mean, y_std
                else:
                    return y_mean
    

    def __deepcopy__(self, memo):
        """
        Overwrites the internal deepcopy method of the class in order to
        also copy instance variables which are not defined in the init.

        
        """
        # Initialize the stuff specified in init
        c = GaussianProcessRegressor(kernel=self.kernel,
                                     alpha=self.alpha,
                                     optimizer=self.optimizer,
                                     n_restarts_optimizer=self.n_restarts_optimizer,
                                     normalize_y=self.normalize_y,
                                     whiten=self.whiten,
                                     copy_X_train=self.copy_X_train,
                                     random_state=self.random_state,
                                     bounds = self.bounds,
                                     normalize_bounds=self.normalize_bounds)
        
        # Initialize the X_train_ and y_train_ part
        if hasattr(self, "X_train_"):
            c.X_train_ = self.X_train_
        if hasattr(self, "y_train_"):
            c.y_train_ = self.y_train_
        if hasattr(self, "_X_train_"):
            c._X_train_ = self._X_train_
        if hasattr(self, "_y_train_"):
            c._y_train_ = self._y_train_
        if hasattr(self, "L_"):
            c.L_ = self.L_
        if hasattr(self, "alpha_"):
            c.alpha_ = self.alpha_
        if hasattr(self, "K_inv_"):
            c.K_inv_ = self.K_inv_
        if hasattr(self, "kernel_"):
            c.kernel_ = self.kernel_
        if hasattr(self, "whiten"):
            c.whiten = self.whiten
        if hasattr(self, "pca"):
            c.pca = self.pca
        if hasattr(self, "y_train_mean_"):
            c.y_train_mean_ = self.y_train_mean_
        if hasattr(self, "_y_train_mean"):
            c._y_train_mean = self._y_train_mean
        if hasattr(self, "_y_train_std"):
            c._y_train_std = self._y_train_std
        if hasattr(self, "y_train_std_"):
            c.y_train_std_ = self._y_train_std_
        if hasattr(self, "normalize_bounds"):
            c.normalize_bounds = self.normalize_bounds
        if hasattr(self, "bto"):
            c.bto = self.bto
        if hasattr(self, "bounds"):
            c.bounds = self.bounds
        if hasattr(self, "normalize_y"):
            c.normalize_y = self.normalize_y

        return c
