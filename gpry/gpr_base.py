"""Shared Gaussian-process regressor base for the sklearn and JAX backends."""

# Builtin
from abc import ABC
import warnings
from typing import Mapping

# External
import numpy as np
from scipy.linalg import cholesky, solve_triangular, cho_solve  # type: ignore
from scipy.linalg.blas import dtrmm as tri_mul  # type: ignore
from scipy.stats import qmc  # type: ignore
from sklearn.base import clone  # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor as sk_GPR  # type: ignore
try:
    from sklearn.utils.validation import validate_data  # type: ignore
except ImportError:
    def validate_data(estimator, *args, **kwargs):  # type: ignore
        return estimator._validate_data(*args, **kwargs)

# Local
from gpry.array_api import ArrayContract
from gpry.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from gpry.tools import check_random_state

GPR_CHOLESKY_LOWER = True
EPS_SQ_NOISE = 1e-6  # diagonal term to be added when WhiteKernel used as noise

class BaseGaussianProcessRegressor(sk_GPR, ABC):
    def __new__(cls, *args, **kwargs):
        if cls is BaseGaussianProcessRegressor:
            use_jax = kwargs.get("use_jax", True)
            if use_jax:
                try:
                    from gpry.gpr_jax import JaxGaussianProcessRegressor

                    return super().__new__(JaxGaussianProcessRegressor)
                except ImportError:
                    pass
            from gpry.gpr_sklearn import SklearnGaussianProcessRegressor

            return super().__new__(SklearnGaussianProcessRegressor)
        return super().__new__(cls)

    r"""
    Modified version of the GaussianProcessRegressor of sklearn.

    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.

    This modified interface provides, in addition to the sklearn-GPR:

       * Re-implements the ``fit`` method to allow for more control in the noise (alpha)
         update and the hyperparameter optimization.
       * Implements derivative return values in the ``predict`` method, as well as a
         ``predict_std`` method to return the standard deviation of the target only
         (useful for acquisition).
       * In the relevant methods, exposes flags to disable input data validation, for
         an additional speed boost.

    Parameters
    ----------
    kernel : kernel object, string, dict, optional (default: "RBF")
        The kernel specifying the covariance function of the GP. If
        "RBF"/"Matern" is passed, the asymmetric kernel::

            ConstantKernel() * RBF/Matern()

        is used as default, where the length correlation kernel is assumed anisotropic
        if a list of bounds is passed using the ``length_scale_prior`` argument.
        To pass different arguments to the kernel, e.g. ``nu=5/2`` for Matern, pass a
        single-key dict as ``{"Matern": {"nu": 2.5}}"``. Note that the kernel's
        hyperparameters are optimized during fitting.

    output_scale_prior : tuple as (min, max), optional (default: [1e-2, 1e3])
        Prior for the (non-squared) scale parameter, in normalised logp units.

    length_scale_prior : tuple as (min, max), optional (default: [1e-3, 1e1])
        Prior for the length parameters, as a fraction of the parameter priors sizes.

    noise_level : float or array-like, optional (default: 1e-2)
        Square-root of the value added to the diagonal of the kernel matrix
        during fitting. Larger values correspond to increased noise level in the
        observations and reduce potential numerical issue during fitting.
        If an array is passed, it must have the same number of entries as the
        data used for fitting and is used as datapoint-dependent noise level.
        Note that this is equivalent to adding a WhiteKernel with c=noise_level.

    noise_fixed : bool (default: True)
        Whether the noise is treated as a fixed diagonal offset in the kernel matrix, or
        an additive white noise term in the kernel. In the latter case, the value given
        as ``noise_level`` is set as the upper bound.

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

    random_state : int or numpy.random.Generator, optional
        The generator used to perform random operations of the GPR. If an integer is
        given, it is used as a seed for the default global numpy random number generator.

    Attributes
    ----------
    X_train_ : array-like, shape = (n_samples, n_features)
        (Possibly transformed) feature values in training data of the GPR (also required
        for prediction). Mostly intended for internal use.

    y_train_ : array-like, shape = (n_samples, [n_output_dims])
        (Possibly transformed) target values in training data of the GPR (also required
        for prediction). Mostly intended for internal use.

    alpha : array-like, shape = (n_samples, [n_output_dims]) or scalar
        The value which is added to the diagonal of the kernel. This is the
        square of `noise_level`.

    kernel_ : :mod:`kernels` object
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters.

    alpha_ : array-like, shape = (n_samples, n_samples)
        **Not to be confused with alpha!** The inverse Kernel matrix of the
        training points multiplied with ``y_train_`` (Dual coefficients of
        training data points in kernel space). Needed at prediction.

    V_ : array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the inverse kernel in ``X_train_``

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``

    scales : tuple
        Kernel scales as ``(output_scale, (length_scale_1, ...))``
    """

    def __init__(
        self,
        kernel="RBF",
        output_scale_prior=[1e-2, 1e3],
        length_scale_prior=[1e-2, 1e2],
        noise_level=1e-2,
        noise_fixed=True,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        random_state=None,
        use_jax=True,
    ):
        self.n_eval = 0
        self.n_eval_loglike = 0
        self._fitted = False
        self.kernel_ = None
        self.last_hyperopt_num_starts = None
        self.last_hyperopt_requested_restarts = None
        self.use_jax = bool(use_jax)
        self._runtime_enabled = self.use_jax
        # Auto-construct inbuilt kernels
        if isinstance(kernel, str):
            kernel = {kernel: {}}
        if isinstance(kernel, Mapping):
            if len(kernel) != 1:
                raise ValueError("'kernel' must be a single-key dict.")
            kernel_name = list(kernel)[0]
            kernel_args = kernel[kernel_name] or {}
            # Check if it's a supported kernel
            try:
                length_corr_kernel = {"rbf": RBF, "matern": Matern}[kernel_name.lower()]
            except KeyError as excpt:
                raise ValueError(
                    "Currently only 'RBF' and 'Matern' are "
                    f"supported as standard kernels. Got '{kernel_name}'."
                ) from excpt
            # Build kernel
            output_scale_init = np.sqrt(output_scale_prior[0] * output_scale_prior[1])
            # Guaranteed to be n-dimensional, if initialised from SurrogateModel
            length_scale_init = np.sqrt(
                length_scale_prior[:, 0] * length_scale_prior[:, 1]
            )
            # Noise treatment
            self.is_noise_in_kernel = not noise_fixed
            if hasattr(noise_level, "__len__"):
                raise TypeError(
                    "If noise is passed per training point, it needs to be fixed. i.e. "
                    "`noise_fixed=True`."
                )
            kernel_args = dict(kernel_args)
            kernel_args.setdefault("length_scale_bounds", length_scale_prior)
            kernel = C(
                output_scale_init**2,
                [output_scale_prior[0] ** 2, output_scale_prior[1] ** 2],
            ) * length_corr_kernel(
                length_scale_init,
                prior_bounds=length_scale_prior,
                **kernel_args,
            )
            # Use noise level as upper bound if added as additive kernel
            if self.is_noise_in_kernel:
                kernel += WhiteKernel(
                    noise_level=noise_level**2,
                    noise_level_bounds=(EPS_SQ_NOISE, noise_level**2),
                )
        sk_GPR.__init__(
            self,
            kernel=kernel,
            alpha=noise_level**2 if not self.is_noise_in_kernel else EPS_SQ_NOISE,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=False,
            copy_X_train=True,
            random_state=random_state,
        )

    @property
    def scales(self):
        """
        Kernel scales as ``(output_scale, (length_scale_1, ...))``.
        """
        if self.kernel_ is None:  # not fitted yet
            length_kernel = self.kernel
        else:
            length_kernel = self.kernel_
        if hasattr(length_kernel.k1, "k1"):  # there is a noise term
            length_kernel = length_kernel.k1
        return (
            np.sqrt(length_kernel.k1.constant_value),
            np.array(length_kernel.k2.length_scale),
        )

    @property
    def array_contract(self):
        return ArrayContract(
            accepted_inputs=frozenset({"numpy"}),
            preferred_input="numpy",
            output_kind="numpy",
        )

    @property
    def preferred_array_kind(self):
        return self.array_contract.preferred_input

    @property
    def native_backend_ready(self):
        return False

    def disable_native_acceleration(self):
        self._runtime_enabled = False

    def prime_hyperparameter_optimization(self):
        return None

    def fit_precompute_native(self):
        return None

    def refresh_native_state(self):
        return None

    def log_marginal_likelihood_with_grad_native(self, theta):
        raise NotImplementedError

    def optimize_hyperparameters_native(self, hyperparameter_bounds, selected_starts):
        raise NotImplementedError

    @property
    def supports_native_acquisition_optimization(self):
        return False

    def optimize_acquisition_native(self, initial_X, bounds, zeta, noise_var, baseline):
        raise NotImplementedError

    def make_ns_loglikelihood_builder(
        self, preprocessing_y, clip_factor, y_clip_min, y_clip_max
    ):
        return None

    def predict_native(self, X, return_std=False):
        if return_std:
            return self.predict(X, return_std=True, validate=False)
        return self.predict(X, validate=False)

    def predict_std_native(self, X):
        return self.predict_std(X, validate=False)

    @property
    def noise_level(self):
        """
        Kernel noise level (not squared).
        """
        if self.is_noise_in_kernel:
            if self.kernel_ is None:  # not fitted yet
                kernel = self.kernel
            else:
                kernel = self.kernel_
            return np.sqrt(kernel.k2.noise_level)
        else:
            return np.sqrt(self.alpha)

    def fit(self, X, y, noise_level=None, fit_hyperparameters=True, validate=True):
        r"""
        Re-implementation of the sk GPR fit method, that allows for updating the noise
        level (as alpha), and exposes flags for input validation and hyperparameter
        fitting.

        If hyperparameters are kept constant, fitting here refers to the re-calculation of
        the GPR inverse matrix :math:`(K(X,X)+\sigma_n^2 I)^{-1}` which is needed for
        predictions.

        The highest cost incurred by this method is the refitting of the GPR kernel
        hyperparameters :math:`\theta`. It can be useful to disable it
        (``fit_hyperparameters=False``) in cases where it is worth saving the
        computational expense in exchange for a loss of information, such as when
        performing parallelized active sampling (NB: this is only possible when the GPR
        hyperparameters have been fit at least once).

        If called with ``X=None, y=None``, it re-fits the model without adding new points.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features), or None
            Training data to append to the model.

        y : array-like, shape = (n_samples, [n_output_dims]), or None
            Target values to append to the data

        noise_level : number, array-like, shape = (n_samples, [n_output_dims])
            Uncorrelated standard deviation(s) to add to the diagonal part of the
            covariance matrix. If an array is passed, it needs to have the same number of
            entries as y. If None, the noise_level set in the instance is used (notice
            that passing None is not recommended if a preprocessor for the GPR training
            samples may have been refit after initialization). If a non-None value is
            passed, it is advisable to refit the hyperparameters of the kernel.

        fit_hyperparameters : Bool or dict (default: True)
            Whether the GPR :math:`\theta`-parameters should be optimised. It can be
            passed a dict, which is always interpreted as ``True``, containing arguments
            for the ``_fit_hyperparameters`` method, namely ``n_restarts`` of the
            optimizer, ``start_from_current`` if the first optimizer run should start
            from the last optimum, or different ``hyperparameters_bounds``. E.g. to
            perform a single GPR optimization run starting at the last hyperparameter
            optimum:
            ``fit_hyperparameters={'start_from_current': True, 'n_restarts': 1}``.

        validate : bool, default: True
            If False, ``X`` and ``y`` are assumed to be correctly formatted, and no
            checks are performed on them. Reduces overhead. Use only for repeated calls
            when the input is programmatically generated to be correct at each stage.

        Returns
        -------
        self : object
            GaussianProcessRegressor class instance.
        """
        if validate:
            kernel_for_validation = self.kernel_ if self.kernel_ is not None else self.kernel
            if kernel_for_validation is None or kernel_for_validation.requires_vector_input:
                dtype, ensure_2d = "numeric", True
            else:
                dtype, ensure_2d = None, False
            X, y = validate_data(
                self,
                X,
                y,
                multi_output=True,
                y_numeric=True,
                ensure_2d=ensure_2d,
                dtype=dtype,
            )
        if (X is None and y is not None) or (X is not None and y is None):
            raise ValueError("Pass neither or both of X, y, but not just one of them.")
        # If neither new points nor new noise nor hyperams fit, return
        if X is None and noise_level is None and fit_hyperparameters is None:
            return self
        if X is not None:
            self.X_train_ = np.copy(X) if self.copy_X_train else X
            self.y_train_ = np.copy(y) if self.copy_X_train else y
        if noise_level is not None:
            if validate:
                if (
                    np.iterable(noise_level)
                    and len(noise_level.shape[0]) != self.y_train_.shape[0]
                ):
                    if noise_level.shape[0] == 1:
                        noise_level = noise_level[0]
                    else:
                        raise ValueError(
                            "noise_level must be a scalar or an array with same number of"
                            f" entries as y. ({noise_level.shape[0]} != "
                            f"{self.y_train_.shape[0]})"
                        )
            if hasattr(noise_level, "__len__") and self.is_noise_in_kernel:
                raise TypeError(
                    "This GPR was initialized with a white noise kernel term. That is "
                    "incompatible with passing noise level per training point."
                )
            elif self.is_noise_in_kernel:
                self.alpha = EPS_SQ_NOISE
            else:
                # The following line causes sometimes noise larger than the one passed.
                self.alpha = np.maximum(np.array(noise_level) ** 2, EPS_SQ_NOISE)
        if fit_hyperparameters is not False:
            if self.is_noise_in_kernel:
                # Used passed noise as an upper bound
                k = self.kernel if self.kernel_ is None else self.kernel_
                k.k2.noise_level_bounds = (
                    min(noise_level**2, EPS_SQ_NOISE * 0.99),
                    noise_level**2,
                )
                # Also lower the current noise if needed, in case it's over the bound
                # after preprocessor is refit
                k.noise_level = min(k.k2.noise_level, noise_level**2)
            if fit_hyperparameters is True:
                fit_hyperparameters = {}
            elif not isinstance(fit_hyperparameters, Mapping):
                raise TypeError(
                    "'fit_hyperparameters' kwarg must be bool|dict, but was "
                    f"{fit_hyperparameters}"
                )
            # Refresh backend-native cached training inputs before hyperopt.
            if self.native_backend_ready:
                self.prime_hyperparameter_optimization()
            self.log_marginal_likelihood_value_ = self._fit_hyperparameters(
                **fit_hyperparameters
            )
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.kernel_.theta, clone_kernel=False
            )
        # Precompute quantities required for predictions which are independent
        # of actual query points
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        # NB: if we got here before returning, we *need* to do this.
        native_fit = self.fit_precompute_native()
        if native_fit is not None:
            try:
                L_native, V_native, alpha_native = native_fit
                self.L_ = np.asarray(L_native)
                self.V_ = np.asarray(V_native)
                self.alpha_ = np.asarray(alpha_native)
                return self
            except Exception:
                pass  # Fall back to numpy Cholesky below
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER,
                               check_finite=False)
            self.V_ = solve_triangular(
                self.L_, np.eye(self.L_.shape[0]), lower=True)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                (
                    f"The kernel, {self.kernel_}, is not returning a "
                    "positive definite matrix. Try gradually increasing "
                    "the 'alpha' parameter of your "
                    "GaussianProcessRegressor estimator."
                ),
            ) + exc.args
            raise
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train_,
            check_finite=False,
        )
        self.refresh_native_state()
        return self

    # Wrapper around log_marginal_likelihood to count the number of evaluations
    def log_marginal_likelihood(
        self, theta=None, eval_gradient=False, clone_kernel=True
    ):
        """
        Log-marginal likelihood of the kernel hyperparameters given the training data.

        Uses JAX automatic differentiation for gradient computation when available,
        falling back to sklearn's analytical gradients otherwise.
        """
        self.n_eval_loglike += 1
        if (eval_gradient
                and theta is not None
                and self.native_backend_ready
                and hasattr(self, "X_train_")):
            try:
                if not clone_kernel:
                    self.kernel_.theta = theta
                lml, grad = self.log_marginal_likelihood_with_grad_native(theta)
                return lml, grad
            except Exception:
                pass
        return super().log_marginal_likelihood(
            theta, eval_gradient=eval_gradient, clone_kernel=clone_kernel)

    def _fit_hyperparameters(
        self,
        start_from_current=True,
        n_restarts=None,
        hyperparameter_bounds=None,
        **kwargs,
    ):
        r"""Optimizes the hyperparameters :math:`\theta` for the current training data.
        The algorithm used to perform the optimization is very similar to the one provided
        by Scikit-learn. The only major difference is, that gradient information is used
        in addition to the values of the marginalized log-likelihood.

        NB: This function does *NOT* update the precomputed kernel matrices. Do not call
        outside self.fit

        Parameters
        ----------
        start_from_current : bool, default: True
            Starts the first optimization run from the current hyperparameters (ignored
            if not previously fitted).

        n_restarts : int, default None
            Number of restarts of the optimizer. If not defined, uses the one set at
            instantiation. ``1`` means a single optimizer run.

        hyperparameter_bounds : array-like, default: None
            Bounds for the hyperparameters, if different from those declared at init.

        Returns
        -------
        self
        """
        if not self._fitted:
            start_from_current = False
        if n_restarts is None:
            n_restarts = self.n_restarts_optimizer
        self.last_hyperopt_requested_restarts = int(n_restarts)
        no_optimizer = self.optimizer is None
        no_hyperparams = self.kernel.n_dims == 0
        no_restarts = n_restarts <= 0
        if no_optimizer or no_hyperparams or no_restarts:
            msg_reasons = []
            if no_optimizer:
                msg_reasons += ["no optimizer has been specified"]
            if no_hyperparams:
                msg_reasons += ["the kernel has no hyperparamenters"]
            if no_restarts:
                msg_reasons += ["the number of optimizer restarts requested is 0."]
            warnings.warn(
                f"Hyper-parameters not (re)fit. Reason(s): {'; '.join(msg_reasons)}."
            )
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.kernel_.theta, clone_kernel=False
            )
            self._update_model()
            return self
        # Choose hyperparameters based on maximizing the log-marginal
        # likelihood (potentially starting from several initial values)
        # We don't need to clone the kernel here, even if overwritten during optimization,
        # because it will be recomputed in the final `log_marginal_likelihood` call.

        def obj_func(theta, eval_gradient=True):
            if eval_gradient:
                lml, grad = self.log_marginal_likelihood(
                    theta, eval_gradient=True, clone_kernel=False
                )
                return -lml, -grad
            else:
                return -self.log_marginal_likelihood(theta, clone_kernel=False)

        if self.kernel_ is None:
            self.kernel_ = clone(self.kernel)
        if hyperparameter_bounds is None:
            hyperparameter_bounds = self.kernel_.bounds
        else:
            # TODO: validate dimensions!
            pass
        # If at least one run will be sampled from the prior, is has to be finite
        if n_restarts - int(start_from_current):
            if not np.isfinite(hyperparameter_bounds).all():
                raise ValueError(
                    "There is at least one optimizer run the requires sampling from the "
                    "hyperparameters' prior, but it has not finite density, because not "
                    "all bounds are finite. You can pass some finite bounds manually "
                    "using ``hyperparameter_bounds``."
                )
        self._rng = check_random_state(self.random_state)

        def _kernel_structure():
            kernel_ref = self.kernel_ if self.kernel_ is not None else self.kernel
            noise_kernel = None
            product_kernel = kernel_ref
            if hasattr(kernel_ref, "k1") and hasattr(kernel_ref, "k2"):
                if hasattr(kernel_ref.k2, "noise_level"):
                    noise_kernel = kernel_ref.k2
                    product_kernel = kernel_ref.k1
            if not (hasattr(product_kernel, "k1") and hasattr(product_kernel, "k2")):
                return None, None, noise_kernel
            return product_kernel.k1, product_kernel.k2, noise_kernel

        def _top_training_subset():
            if not hasattr(self, "X_train_") or self.X_train_ is None:
                return None, None
            n_train = len(self.X_train_)
            d_train = self.X_train_.shape[1]
            if n_train < max(6, d_train + 1):
                return None, None
            y_train = np.asarray(self.y_train_).reshape(-1)
            n_keep = min(
                n_train,
                max(3 * d_train, 12),
                max(6, int(np.ceil(0.35 * n_train))),
            )
            idx = np.argsort(y_train)[-n_keep:]
            return np.asarray(self.X_train_[idx], dtype=float), y_train[idx]

        def _build_theta_guess(length_scales=None, output_scale=None):
            constant_kernel, length_kernel, noise_kernel = _kernel_structure()
            if constant_kernel is None or length_kernel is None:
                return None
            theta_actual = []
            actual_bounds = []
            if output_scale is None:
                output_scale_sq = float(constant_kernel.constant_value)
            else:
                output_scale_sq = float(output_scale) ** 2
            output_bounds = np.exp(np.asarray(constant_kernel.bounds, dtype=float))
            output_scale_sq = float(np.clip(output_scale_sq, output_bounds[0, 0], output_bounds[0, 1]))
            theta_actual.append(output_scale_sq)
            actual_bounds.append(output_bounds[0])

            ls_bounds = np.exp(np.asarray(length_kernel.bounds, dtype=float))
            ls_current = np.atleast_1d(length_kernel.length_scale).astype(float)
            if length_scales is None:
                ls_guess = ls_current
            else:
                ls_guess = np.atleast_1d(length_scales).astype(float)
            if ls_guess.shape[0] != ls_current.shape[0]:
                return None
            ls_guess = np.clip(ls_guess, ls_bounds[:, 0], ls_bounds[:, 1])
            theta_actual.extend(ls_guess.tolist())
            actual_bounds.extend(ls_bounds.tolist())
            if noise_kernel is not None:
                noise_bounds = np.exp(np.asarray(noise_kernel.bounds, dtype=float))
                noise_level = float(noise_kernel.noise_level)
                noise_level = float(
                    np.clip(noise_level, noise_bounds[0, 0], noise_bounds[0, 1])
                )
                theta_actual.append(noise_level)
                actual_bounds.append(noise_bounds[0])
            theta = np.log(np.asarray(theta_actual, dtype=float))
            if theta.shape[0] != hyperparameter_bounds.shape[0]:
                return None
            theta = np.clip(theta, hyperparameter_bounds[:, 0], hyperparameter_bounds[:, 1])
            return theta

        def _estimate_output_scale(y_subset):
            if y_subset is None or len(y_subset) == 0:
                return None
            y_scale = float(np.std(y_subset))
            if not np.isfinite(y_scale) or y_scale <= 0:
                y_scale = float(np.std(np.asarray(self.y_train_).reshape(-1)))
            if not np.isfinite(y_scale) or y_scale <= 0:
                y_scale = 1.0
            return max(y_scale, 1e-3)

        def _theta_guess_from_local_cov():
            X_subset, y_subset = _top_training_subset()
            if X_subset is None or len(X_subset) < 3:
                return None
            if len(X_subset) == 1:
                diag_scales = np.full(X_subset.shape[1], 0.1)
            else:
                cov = np.cov(X_subset.T, ddof=0)
                cov = np.atleast_2d(cov)
                diag_scales = np.sqrt(np.maximum(np.diag(cov), 1e-6))
            diag_scales = np.maximum(diag_scales, 1e-3)
            return _build_theta_guess(
                length_scales=diag_scales,
                output_scale=_estimate_output_scale(y_subset),
            )

        def _theta_guess_from_local_quadratic():
            X_subset, y_subset = _top_training_subset()
            if X_subset is None:
                return None
            d = X_subset.shape[1]
            n_subset = len(X_subset)
            min_points = max(2 * d + 3, d + 4)
            if n_subset < min_points:
                return None
            x0 = X_subset[np.argmax(y_subset)]
            dx = X_subset - x0
            columns = [np.ones(n_subset)]
            columns.extend(dx[:, i] for i in range(d))
            quad_terms = []
            for i in range(d):
                for j in range(i, d):
                    quad_terms.append((i, j))
                    columns.append(dx[:, i] * dx[:, j])
            design = np.column_stack(columns)
            ridge = 1e-6 * np.eye(design.shape[1])
            ridge[0, 0] = 0.0
            try:
                beta = np.linalg.solve(design.T @ design + ridge, design.T @ y_subset)
            except np.linalg.LinAlgError:
                return None
            hessian = np.zeros((d, d), dtype=float)
            coeffs = beta[1 + d:]
            for coeff, (i, j) in zip(coeffs, quad_terms):
                if i == j:
                    hessian[i, i] = -2.0 * coeff
                else:
                    value = -coeff
                    hessian[i, j] = value
                    hessian[j, i] = value
            curvature = np.maximum(np.diag(hessian), 1e-6)
            output_scale = _estimate_output_scale(y_subset)
            length_scales = np.sqrt(max(output_scale, 1e-6) / curvature)
            return _build_theta_guess(
                length_scales=length_scales,
                output_scale=output_scale,
            )

        def _rng_uint32():
            if isinstance(self._rng, np.random.Generator):
                return int(self._rng.integers(0, 2**32 - 1))
            return int(self._rng.randint(0, 2**32 - 1))

        def _sample_restart_pool(n_samples):
            if n_samples <= 0:
                return []
            sampler = qmc.LatinHypercube(
                d=hyperparameter_bounds.shape[0],
                seed=_rng_uint32(),
            )
            unit = sampler.random(n=n_samples)
            return qmc.scale(
                unit, hyperparameter_bounds[:, 0], hyperparameter_bounds[:, 1]
            )

        def _boundary_penalty(theta):
            bounds_span = np.maximum(
                hyperparameter_bounds[:, 1] - hyperparameter_bounds[:, 0], 1e-12
            )
            lower_margin = (theta - hyperparameter_bounds[:, 0]) / bounds_span
            upper_margin = (hyperparameter_bounds[:, 1] - theta) / bounds_span
            margin = np.minimum(lower_margin, upper_margin)
            clipped = np.clip(0.02 - margin, 0.0, None)
            return float(np.sum(clipped / 0.02))

        def _is_pathological(theta):
            bounds_span = np.maximum(
                hyperparameter_bounds[:, 1] - hyperparameter_bounds[:, 0], 1e-12
            )
            lower_margin = (theta - hyperparameter_bounds[:, 0]) / bounds_span
            upper_margin = (hyperparameter_bounds[:, 1] - theta) / bounds_span
            margin = np.minimum(lower_margin, upper_margin)
            close_to_bounds = margin < 0.01
            return int(np.sum(close_to_bounds)) >= max(2, int(np.ceil(len(theta) / 2)))

        n_prior_candidates = max(
            n_restarts - int(start_from_current),
            min(max(4 * max(n_restarts, 1), 8), 24),
        )
        theta_candidates = []
        if start_from_current:
            theta_candidates.append(np.array(self.kernel_.theta, copy=True))
        for deterministic_guess in (
            _theta_guess_from_local_cov(),
            _theta_guess_from_local_quadratic(),
        ):
            if deterministic_guess is not None:
                theta_candidates.append(np.asarray(deterministic_guess, dtype=float))
        for theta_initial in _sample_restart_pool(n_prior_candidates):
            theta_candidates.append(np.asarray(theta_initial, dtype=float))

        scored_candidates = []
        for theta_initial in theta_candidates:
            try:
                eval_value = float(obj_func(theta_initial, eval_gradient=False))
            except Exception:
                continue
            if np.isfinite(eval_value):
                scored_candidates.append((eval_value, np.asarray(theta_initial)))

        if not scored_candidates:
            raise RuntimeError("Failed to produce any finite hyperparameter start.")

        scored_candidates.sort(
            key=lambda item: (item[0], _boundary_penalty(item[1]))
        )
        selected_starts = []
        seen = set()
        for _, theta_initial in scored_candidates:
            key = tuple(np.round(theta_initial, decimals=12))
            if key in seen:
                continue
            seen.add(key)
            selected_starts.append(theta_initial)
            target_runs = max(n_restarts, int(start_from_current)) + 2
            if len(selected_starts) >= target_runs:
                break
        self.last_hyperopt_num_starts = len(selected_starts)

        if self.native_backend_ready:
            try:
                theta_opt, neg_lml = self.optimize_hyperparameters_native(
                    hyperparameter_bounds, selected_starts
                )
                self.kernel_.theta = theta_opt
                self._fitted = True
                self.L_, self.V_, self.alpha_ = None, None, None
                return -neg_lml
            except Exception:
                pass
        optima = []
        for theta_initial in selected_starts:
            # Run the optimizer!
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                optima.append(
                    self._constrained_optimization(
                        obj_func, theta_initial, hyperparameter_bounds
                    )
                )
        # Select result from run with minimal (negative) log-marginal likelihood.
        valid_optima = [
            (theta, value)
            for theta, value in optima
            if not _is_pathological(np.asarray(theta))
        ]
        selected_optima = valid_optima if valid_optima else optima
        selected_idx = min(
            range(len(selected_optima)),
            key=lambda idx: (
                selected_optima[idx][1],
                _boundary_penalty(np.asarray(selected_optima[idx][0])),
            ),
        )
        self.kernel_.theta = selected_optima[selected_idx][0]
        self._fitted = True
        # Reset pre-computed matrices
        self.L_, self.V_, self.alpha_ = None, None, None
        return -selected_optima[selected_idx][1]

    def predict(
        self,
        X,
        return_std=False,
        return_mean_grad=False,
        return_std_grad=False,
        validate=True,
    ):
        """
        Predict output for X.

        Reimplementation of the sk-learn GPR method: in addition to the mean of the
        predictive distribution, also its standard deviation (return_std=True), the
        gradient of the mean and the standard-deviation with respect to X can be
        optionally provided.

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

        validate : bool, default: True
            If False, ``X`` and ``y`` are assumed to be correctly formatted, and no
            checks are performed on them. Reduces overhead. Use only for repeated calls
            when the input is programmatically generated to be correct at each stage.

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

        y_mean_grad : shape = (n_samples, n_features), optional
            The gradient of the predicted mean.

        y_std_grad : shape = (n_samples, n_features), optional
            The gradient of the predicted std.
        """
        self.n_eval += len(X)
        if return_std_grad and not (return_std and return_mean_grad):
            raise ValueError(
                "Not returning std_gradient without returning "
                "the std and the mean grad."
            )
        if X.shape[0] != 1 and (return_mean_grad or return_std_grad):
            raise ValueError(
                "Mean grad and std grad not implemented \
                for n_samples > 1"
            )
        if (self.native_backend_ready
                and hasattr(self, "X_train_")):
            if return_mean_grad:
                x = X[0]
                mean, std, grad_mean, grad_std = self.predict_with_grads(x)
                y_mean = np.array([mean])
                return_values = [y_mean]
                if return_std:
                    y_std = np.array([std])
                    return_values.append(y_std)
                return_values.append(grad_mean)
                if return_std_grad:
                    return_values.append(grad_std)
                return return_values
            elif return_std:
                y_mean_j, y_std_j = self.predict_native(X, return_std=True)
                return [np.asarray(y_mean_j), np.asarray(y_std_j)]
            else:
                y_mean_j = self.predict_native(X, return_std=False)
                return [np.asarray(y_mean_j)]
        if validate:
            if self.kernel is None or self.kernel.requires_vector_input:
                dtype, ensure_2d = "numeric", True
            else:
                dtype, ensure_2d = None, False
            X = validate_data(self, X, ensure_2d=ensure_2d, dtype=dtype, reset=False)
        # If not fit yet, predict based on GP prior
        if not hasattr(self, "X_train_"):
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
        # If already fit, use GP posterior to predict
        # Alg 2.1, page 19, line 4 -> f*_bar = K(X_test, X_train) . alpha
        # NB: there is no y-normalisation here: the data are passed normalised to the GPR
        K_trans = self.kernel_(X, self.X_train_)
        y_mean = K_trans.dot(self.alpha_)
        return_values = [y_mean]
        if return_std:
            # Compute variance of predictive distribution
            # Use einsum to avoid explicitly forming the large matrix
            # V^T @ V just to extract its diagonal afterward.
            M = tri_mul(1.0, self.V_, K_trans.T, lower=True)
            y_var = self.kernel_.diag(X).copy()
            y_var -= np.einsum("ji,ji->i", M, M, optimize=True)
            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn(
                    "Predicted variances smaller than 0. Setting those variances to 0."
                )
                y_var[y_var_negative] = 0.0
            y_std = np.sqrt(y_var)
            return_values.append(y_std)
        if return_mean_grad:
            grad = self.kernel_.gradient_x(X[0], self.X_train_)
            grad_mean = np.dot(grad.T, self.alpha_)
            return_values.append(grad_mean)
            if return_std_grad:
                if not np.any(y_std):  # do not compute if all stds null
                    grad_std = np.zeros(X.shape[1])
                else:
                    grad_std = (
                        -np.dot(K_trans, np.dot(self.V_.T.dot(self.V_), grad))[0]
                        / y_std
                    )
                return_values.append(grad_std)
        return return_values

    def predict_std(self, X, validate=True):
        """
        Predict output standart deviation for X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated.

        validate : bool, default: True
            If False, ``X`` and ``y`` are assumed to be correctly formatted, and no
            checks are performed on them. Reduces overhead. Use only for repeated calls
            when the input is programmatically generated to be correct at each stage.

        Returns
        -------
        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.
        """
        self.n_eval += len(X)
        if (self.native_backend_ready
                and hasattr(self, "X_train_")):
            return np.asarray(self.predict_std_native(X))
        if validate:
            if self.kernel is None or self.kernel.requires_vector_input:
                dtype, ensure_2d = "numeric", True
            else:
                dtype, ensure_2d = None, False
            X = validate_data(self, X, ensure_2d=ensure_2d, dtype=dtype, reset=False)
        # If not fit yet, predict based on GP prior
        if not hasattr(self, "X_train_"):  # Not fit; predict based on GP prior
            return np.sqrt(self.kernel.diag(X))
        # If already fit, use GP posterior to predict
        # Alg 2.1, page 19, line 4 -> f*_bar = K(X_test, X_train) . alpha
        # NB: there is no y-normalisation here: the data are passed normalised to the GPR
        K_trans = self.kernel_(X, self.X_train_)
        # Use einsum to avoid explicitly forming the large matrix
        # V^T @ V just to extract its diagonal afterward.
        M = tri_mul(1.0, self.V_, K_trans.T, lower=True)
        y_var = self.kernel_.diag(X).copy()
        y_var -= np.einsum("ji,ji->i", M, M, optimize=True)
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn(
                "Predicted variances smaller than 0. Setting those variances to 0."
            )
            y_var[y_var_negative] = 0.0
        return np.sqrt(y_var)
