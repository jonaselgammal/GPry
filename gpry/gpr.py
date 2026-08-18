"""
This module contains the Gaussia Process Regressor class. For now, the only implemented
one is a modified version of the ``sklearn`` implementation, with some tweaks to allow
disabling input validation and implementing more control at the hyperparameter
optimization stage.

At the moment, the GPR kernel is a product of a constant kernel and an anisotropic
length-correlation one. Noise is treated as uncorrelated standard deviations for inputs,
and thus simply added to the kernel matrix diagonal.
"""

# Builtin
import warnings
from operator import itemgetter
from typing import Mapping

# External
import numpy as np
from scipy.linalg import cholesky, solve_triangular, cho_solve  # type: ignore
from scipy.linalg.blas import dtrmm as tri_mul  # type: ignore
from sklearn.base import clone  # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor as sk_GPR  # type: ignore
from sklearn.utils.validation import validate_data  # type: ignore

# Local
from gpry.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from gpry.tools import check_random_state, mean_covmat_from_evals

GPR_CHOLESKY_LOWER = True
EPS_SQ_NOISE = 1e-6  # diagonal term to be added when WhiteKernel used as noise


class GaussianProcessRegressor(sk_GPR):
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

    length_scale_prior : tuple as (min, max), optional (default: [1e-3, 1e2])
        Prior for the length parameters, as a fraction of the parameter priors sizes.
        May also be given per-dimension, as an array of shape ``(n_dims, 2)``. Pass the
        string ``"dynamic"`` to derive the bounds from the prior size instead, which
        requires ``prior_bounds`` (see :class:`gpry.kernels.Hyperparameter`).

        The upper bound is deliberately well above 1: in the normalised input space a
        smooth (nearly quadratic) log-posterior is best described by a correlation
        length several times the prior width, so fitted length scales of O(10) are
        normal and a tighter bound would clamp them.

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

    restart_strategy : str, optional (default: "uniform")
        How the *random* restarts are drawn (the informed starts -- the current
        kernel and the covariance-based guess -- are unaffected). One of
        ``"uniform"`` (log-uniform over the full hyperparameter prior, the
        historical behaviour), ``"screen"``, ``"local"`` or ``"local_screen"``.
        See :meth:`_draw_restart_thetas`.

    prior_bounds : array-like, shape = (n_dims, 2), optional
        Bounds of the parameter space, **in the GPR's own (transformed) input space**,
        i.e. ``preprocessing_X.transform_bounds(bounds)``. Only needed when a length
        correlation kernel is created with ``length_scale_prior="dynamic"``, in which
        case the length-scale bounds are derived from the prior size. Supplied by
        :class:`gpry.surrogate.SurrogateModel`.

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
        length_scale_prior=[1e-3, 1e2],
        noise_level=1e-2,
        noise_fixed=True,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        restart_strategy="uniform",
        prior_bounds=None,
        random_state=None,
    ):
        self.restart_strategy = restart_strategy
        self.n_eval = 0
        self.n_eval_loglike = 0
        self._fitted = False
        self.kernel_ = None
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
            # Guaranteed to be n-dimensional, if initialised from SurrogateModel.
            # When "dynamic", derive an initial scale from the prior widths.
            if isinstance(length_scale_prior, str) and length_scale_prior == "dynamic":
                if prior_bounds is None:
                    raise TypeError(
                        "prior_bounds is required when length_scale_prior='dynamic'."
                    )
                _pb = np.asarray(prior_bounds)
                _widths = _pb[:, 1] - _pb[:, 0]
                length_scale_init = 0.1 * _widths
            else:
                length_scale_prior = np.asarray(length_scale_prior)
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
            kernel = C(
                output_scale_init**2,
                [output_scale_prior[0] ** 2, output_scale_prior[1] ** 2],
            ) * length_corr_kernel(
                length_scale_init,
                length_scale_prior=length_scale_prior,
                prior_bounds=prior_bounds,
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

    def __repr__(self):
        """Print the current kernel."""
        return str(self.kernel_)

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
            if self.kernel_.requires_vector_input:
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
                self.alpha = max(np.array(noise_level) ** 2, EPS_SQ_NOISE)
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
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
            self.V_ = solve_triangular(self.L_, np.eye(self.L_.shape[0]), lower=True)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                (
                    f"The kernel, {self.kernel_}, is not returning a positive "
                    "definite matrix. Try gradually increasing the 'alpha' "
                    "parameter of your GaussianProcessRegressor estimator."
                ),
            ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train_,
            check_finite=False,
        )
        return self

    # Wrapper around log_marginal_likelihood to count the number of evaluations
    def log_marginal_likelihood(self, *args, **kwargs):
        """
        Log-marginal likelihood of the kernel hyperparameters given the training data.
        """
        self.n_eval_loglike += 1
        return super().log_marginal_likelihood(*args, **kwargs)

    def _draw_restart_thetas(
        self, n, hyperparameter_bounds, obj_func, strategy="uniform", theta_cov=None,
        screen_factor=8, local_sigma_decades=1.0,
    ):
        """
        Produce the starting points for the *random* hyperparameter restarts.

        Measurements on real runs show that the accepted fit essentially always comes
        from one of the two informed starts (previous theta, covariance-based guess),
        while the tens of random restarts land far from the optimum and merely consume
        the fit budget. These strategies try to spend that budget better:

        - ``"uniform"``: the original behaviour -- log-uniform over the whole
          hyperparameter prior, rejecting draws with non-finite log-marginal likelihood.
        - ``"screen"``: draw ``screen_factor * n`` uniform candidates, evaluate the
          log-marginal likelihood once each (gradient-free: one Cholesky, i.e. the cost
          of a single L-BFGS iteration) and keep the best ``n``. Same number of
          expensive optimisations, but started from demonstrably better points.
        - ``"local"``: draw log-normal perturbations around ``theta_cov`` (width
          ``local_sigma_decades`` decades), i.e. search near the data-driven guess
          instead of across the whole prior. Falls back to ``"uniform"`` if no
          covariance-based guess is available.
        - ``"local_screen"``: ``"local"`` proposals, screened as in ``"screen"``.

        Returns a list of ``n`` theta vectors.
        """
        known = ("uniform", "screen", "local", "local_screen")
        if strategy not in known:
            raise ValueError(
                f"Unknown restart_strategy {strategy!r}; expected one of {known}."
            )
        if n <= 0:
            return []
        lo, hi = hyperparameter_bounds[:, 0], hyperparameter_bounds[:, 1]
        use_local = strategy in ("local", "local_screen") and theta_cov is not None
        n_draw = n * screen_factor if strategy in ("screen", "local_screen") else n

        def _propose():
            if use_local:
                # log-normal around the covariance-based guess, clipped to the prior
                width = local_sigma_decades * np.log(10.0)
                return np.clip(
                    theta_cov + self._rng.normal(scale=width, size=len(theta_cov)),
                    lo, hi,
                )
            return self._rng.uniform(lo, hi)

        candidates = []
        for _ in range(n_draw):
            # Reject non-finite draws, as the original implementation did.
            for _try in range(100):
                th = _propose()
                val = obj_func(th, eval_gradient=False)
                if np.isfinite(val):
                    break
            candidates.append((th, val))
        if strategy in ("screen", "local_screen"):
            # obj_func returns the NEGATIVE log-marginal likelihood: smaller is better.
            candidates.sort(key=lambda tv: tv[1])
            candidates = candidates[:n]
        return [th for th, _ in candidates[:n]]

    def _fit_hyperparameters(
        self,
        start_from_current=True,
        start_from_cov=False,
        n_restarts=None,
        hyperparameter_bounds=None,
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

        start_from_cov : bool, default: False
            Starts the first optimization (or the second one if
            ``start_from_current=True``) from the lenghts computed with the square root
            of the diagonal of the covariance of the training set.

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
        if n_restarts == 1 and start_from_current and start_from_cov:
            raise ValueError(
                "Cannot choose both 'start_from_current' and 'start_from_cov' if the "
                "number of restarts is 1."
            )
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
        optima = []
        self._rng = check_random_state(self.random_state)
        # The informed starts (previous theta, and the covariance-based guess) are the
        # ones that actually find the optimum in practice; the remaining restarts are
        # drawn randomly. ``restart_strategy`` controls how those random starts are
        # produced -- see ``_draw_restart_thetas``.
        strategy = getattr(self, "restart_strategy", "uniform") or "uniform"
        n_informed = int(bool(start_from_current)) + int(bool(start_from_cov))
        theta_cov = None
        if start_from_cov:
            theta_cov = self.kernel_.theta.copy()
            theta_cov[0] = (lambda b: b[1] - 0.1 * (b[1] - b[0]))(
                hyperparameter_bounds[0]
            )
            _, X_cov = mean_covmat_from_evals(self.X_train_, self.y_train_)
            theta_cov[1 : 1 + X_cov.shape[0]] = np.log(np.sqrt(np.diag(X_cov)))
        random_thetas = self._draw_restart_thetas(
            max(0, n_restarts - n_informed), hyperparameter_bounds, obj_func,
            strategy=strategy, theta_cov=theta_cov,
        )
        i_random = 0
        for iteration in range(n_restarts):
            if iteration == 0 and start_from_current:
                # self.kernel_ guaranteed to exist because self.fitted checked above
                theta_initial = self.kernel_.theta
            elif start_from_cov and (
                (iteration == 0 and not start_from_current)
                or (iteration == 1 and start_from_current)
            ):
                # Guess the length hyperparameters from the covariance matrix of the
                # training points (not from a sample from the GP mean, bc it may still be
                # overfitting; use a high value for the output scale
                theta_initial = theta_cov
            else:
                theta_initial = random_thetas[i_random]
                i_random += 1
            # Run the optimizer!
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                optima.append(
                    self._constrained_optimization(
                        obj_func, theta_initial, hyperparameter_bounds
                    )
                )
        # Select result from run with minimal (negative) log-marginal likelihood.
        lml_values = list(map(itemgetter(1), optima))
        self.kernel_.theta = optima[np.argmin(lml_values)][0]
        self._fitted = True
        # Reset pre-computed matrices
        self.L_, self.V_, self.alpha_ = None, None, None
        return -np.min(lml_values)

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
                if np.any(y_std):  # do not compute if all stds null
                    grad_std = np.zeros(X.shape[1])
                else:
                    # TODO: This can be made much more efficient,
                    #       but I don't think it's used currently
                    grad_std = (
                        -np.dot(K_trans, np.dot(self.V_.T.dot(self.V_), grad))[0]
                        / y_std
                    )
                return_values.append(grad_std)
        return return_values

    def predict_mean_grad_batch(self, X):
        """
        Vectorized GP posterior mean and its input-gradient for many points.

        Batched counterpart of ``predict(X, return_mean_grad=True)``, which is
        restricted to a single point. Intended for the gradient-based
        acquisition sampler (see :mod:`gpry.mc_interfaces`).

        ``X`` is expected in the regressor's own (pre-processed) input space,
        i.e. the space in which ``kernel_``, ``X_train_`` and ``alpha_`` live,
        and the returned mean/gradient are in the regressor's y-scale. No
        preprocessing, no infinities classifier and no clipping are applied
        here; use :meth:`gpry.surrogate.SurrogateModel.predict_mean_grad_batch`
        for the pre-processed, original-space counterpart.

        Parameters
        ----------
        X : array-like, shape=(n_points, n_features)

        Returns
        -------
        mean : array, shape=(n_points,)
        mean_grad : array, shape=(n_points, n_features)
        """
        X = np.atleast_2d(X)
        self.n_eval += X.shape[0]
        if not hasattr(self, "X_train_"):  # not fit: GP prior mean is zero
            return np.zeros(X.shape[0]), np.zeros_like(X)
        alpha = np.ravel(self.alpha_)
        K_trans = self.kernel_(X, self.X_train_)  # (n_points, n_train)
        y_mean = K_trans.dot(alpha)  # (n_points,)
        # (n_points, n_train, n_features)
        grad_K = self.kernel_.gradient_x_batch(X, self.X_train_)
        y_mean_grad = np.einsum("mnd,n->md", grad_K, alpha, optimize=True)
        return y_mean, y_mean_grad

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
