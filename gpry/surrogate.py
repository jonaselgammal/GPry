"""
Class holding the surrogate model.

Underscored-after attributes mean "preprocessed/transformed"
"""

# Builtin
import warnings
from copy import deepcopy
from typing import Mapping
from numbers import Number

# External
import numpy as np
import pandas as pd  # type: ignore
from sklearn.utils.validation import check_array  # type: ignore

# Local
from gpry.gpr import GaussianProcessRegressor
from gpry.svm import SVM
from gpry.preprocessing import DummyPreprocessor
from gpry.tools import (
    check_random_state,
    get_Xnumber,
    delta_logp_of_1d_nstd,
    generic_params_names,
    shrink_bounds,
    is_in_bounds,
)


class SurrogateModel:
    r"""
    Object holding the Gaussian Process Regressor, and, if applicable, the
    input/output preprocessing layer and the infinities classifier.

    Parameters
    ----------
    kernel : kernel object, string, dict, optional (default: "RBF")
        The kernel specifying the covariance function of the GP. If
        "RBF"/"Matern" is passed, the asymmetric kernel::

            ConstantKernel() * RBF/Matern()

        is used as default where ``n_dim`` is the number of dimensions of the
        training space. In this case you will have to provide the prior bounds
        as the ``bounds`` parameter to the GP. For the Matern kernel
        :math:`\nu=3/2`. To pass different arguments to the kernel, e.g. ``nu=5/2``
        for Matern, pass a single-key dict as ``{"Matern": {"nu": 2.5}}"``. Note
        that the kernel's hyperparameters are optimized during fitting.

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

    clip_factor : float, optional (default: 1.1)
        Factor for upper clipping of the GPR predictions, to avoid overshoots.

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
        passed the data is not preprocessed. The `fit` method of the preprocessor
        is only called when the GP's hyperparameters are refit.

    account_for_inf : SVM, None or "SVM" (default: "SVM")
        Uses a SVM (Support Vector Machine) to classify the data into finite and
        infinite values. This allows the GP to express values of -inf in the
        data (unphysical values). If all values are finite the SVM will just
        pass the data through itself and do nothing.

    inf_threshold : None, float or str
        Threshold for the infinities classifier to consider a value finite, understood as
        a positive difference with respect to the current maximum y. It can be given as a
        string formed of a number and ending in "s", meaning the distance to the mode
        which shall be considered finite in :math:`\sigma` using a :math:`\chi^2`
        distribution. Used only if account_for_inf is not None.
    keep_min_finite : int, optional (default: None)
        Minimum number of points to be considered finite, and this part of the GPR
        training set. Useful e.g. if a point with a much larger y-value than the rest is
        suddenly found. If no infinities classifier selected, it has no effect. Otherwise,
        if ``None``, is set to the dimensionality of the problem.

    bounds : array-like, shape=(n_dims,2), optional
        Array of bounds of the prior [lower, upper] along each dimension. Has
        to be provided when the kernel shall be built automatically by the GP.

    trust_region_factor : float, optional
        If defined as a positive float, it defines a trust region as a hypercube
        containing the GPR (finite) training set, enlarged by the given factor.
        The bounds of this trust region can be read from the ``trust_bounds`` attribute,
        and points outside these bounds will cause the ``predict`` method to return a
        negative infinity. Useful when the posterior can be expected to be much
        smaller than the prior (but in that case it would be preferable to simply reduce
        the prior bounds), noticeable e.g. by the acquisition module is taking too long
        to propose points.

    trust_region_nstd : float, optional
        If defined as a positive float, the definition of the trust region only takes into
        account training points corresponding to a significance (assuming a Gaussian
        posterior) equivalent to this value in 1d standard deviations.

    random_state : int or numpy.random.Generator, optional
        The generator used to perform random operations of the GPR. If an integer is
        given, it is used as a seed for the default global numpy random number generator.

    verbose : 1, 2, 3, optional (default: 1)
        Level of verbosity of the GP. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise.

    Attributes
    ----------
    n : int
        Number of points/features in the training set of the GPR.

    d : int
        Dimensionality of the training data.

    bounds : array
        The bounds with which the GPR was defined.

    trust_bounds : array or None
        The bounds of the trust region if ``trust_region_factor`` was defined, or ``None``
        otherwise.

    X_train : array-like, shape = (n_samples, n_features)
        Original (untransformed) feature values in training data of the GPR. Intended to
        be used when one wants to access the training data for any purpose.

    y_train : array-like, shape = (n_samples, [n_output_dims])
        Original (untransformed) target values in training data of the GPR. Intended to be
        used when one wants to access the training data for any purpose.

    X_train_ : array-like, shape = (n_samples, n_features)
        (Possibly transformed) feature values in training data of the GPR (also required
        for prediction). Mostly intended for internal use.

    y_train_ : array-like, shape = (n_samples, [n_output_dims])
        (Possibly transformed) target values in training data of the GPR (also required
        for prediction). Mostly intended for internal use.

    n_total : int
        Number of points/features in the training set of the model, including points with
        target values classified as infinite.

    X_train_all : array-like, shape = (n_samples, n_features)
        Original (untransformed) feature values in training data of the model, including
        points with target values classified as infinite, and thus not part of the
        training set of the GPR.

    y_train_all : array-like, shape = (n_samples, [n_output_dims])
        Original (untransformed) target values in training data of the model, including
        values classified as infinite, and thus not part of the training set of the GPR.

    X_train_all_ : array-like, shape = (n_samples, n_features)
        (Possibly transformed) feature values in training data of the model, including
        points with target values classified as infinite.

    y_train_all_ : array-like, shape = (n_samples, [n_output_dims])
        (Possibly transformed) target values in training data of the GPR model, including
        points with target values classified as infinite.

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

    alpha_ : array-like, shape = (n_samples, n_samples)
        **Not to be confused with alpha!** The inverse Kernel matrix of the
        training points multiplied with ``y_train_`` (Dual coefficients of
        training data points in kernel space). Needed at prediction.

    V_ : array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the inverse kernel in ``X_train_``

        .. warning::

            ``L_`` is not recomputed when using the append method
            without refitting the hyperparameters. As only ``K_inv_`` and
            ``alpha_`` are used at prediction this is not neccessary.

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.gpr.kernel_.theta``


    **Methods:**

    .. autosummary::
        :toctree: stubs

        append
        fit
        _update_model
        predict
    """

    def __init__(
        self,
        # gpr
        kernel="RBF",
        output_scale_prior=[1e-2, 1e3],
        length_scale_prior=[1e-3, 1e1],
        noise_level=1e-2,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        # maybe gpr (better not)
        clip_factor=1.1,
        # preprocessing
        preprocessing_X=None,
        preprocessing_y=None,
        # svm
        account_for_inf="SVM",
        inf_threshold="20s",
        keep_min_finite=None,
        # trust region
        trust_region_factor=None,
        trust_region_nstd=None,
        # prior/general
        bounds=None,
        random_state=None,
        verbose=1,
    ):
        self.preprocessing_X = (
            DummyPreprocessor if preprocessing_X is None else preprocessing_X
        )
        self.preprocessing_y = (
            DummyPreprocessor if preprocessing_y is None else preprocessing_y
        )
        self.noise_level = float(noise_level)  # at this point is a single number
        self.noise_level_ = None
        if clip_factor < 1:
            raise ValueError("'clip_factor' must be >= 1, or None for no clippling.")
        self.clip_factor = clip_factor
        self.n_eval = 0
        self.verbose = verbose
        self.inf_value = np.inf
        self.minus_inf_value = -np.inf
        self._fitted = False
        self.bounds = bounds
        # Arrays containing the evaluations used to train the model,
        # regardless of whether they are used and for what.
        self._X, self._y = (
            np.empty((0, len(bounds)), dtype=float),
            np.empty((0,), dtype=float),
        )
        self._X_, self._y_ = None, None
        # Initialize trust region
        self.trust_region = TrustRegion(
            bounds=self.bounds,
            factor=trust_region_factor,
            nstd=trust_region_nstd,
            n_min=self.d,
        )
        # Initialize SVM if necessary
        self.inf_threshold = inf_threshold
        self.keep_min_finite = (
            keep_min_finite if keep_min_finite is not None else max(2, self.d)
        )
        if isinstance(account_for_inf, str) and account_for_inf.lower() == "svm":
            self.infinities_classifier = SVM(random_state=random_state)
        elif account_for_inf is False:
            self.infinities_classifier = None
        else:
            self.infinities_classifier = account_for_inf
        if self.infinities_classifier is not None:
            y_preprocessor_guaranteed_linear = getattr(
                self.preprocessing_y, "is_linear", False
            )
            if not y_preprocessor_guaranteed_linear:
                warnings.warn(
                    "If using a standard classifier for infinities, the y-preprocessor "
                    "needs to be linear (declare an attr ``is_linear=True``). This may "
                    "lead to errors further in the pipeline."
                )
            if self.inf_threshold is None:
                raise ValueError(
                    "Specify 'inf_threshold' if using infinities classifier."
                )
            value, is_sigma_units, sigma_power = get_Xnumber(
                self.inf_threshold, "s", None, dtype=float, varname="inf_threshold"
            )
            if sigma_power is not None:
                raise ValueError("Power for sigma not supported.")
            if is_sigma_units:
                self._diff_threshold = self.compute_threshold_given_sigma(value, self.d)
            else:
                self._diff_threshold = value
        # Construct the Gaussian Process Regressor
        # NB: hyperparameter priors and noise level are understood in transformed space
        # Make sure the length scale prior is unfolded per-dimensionality, since the GPR
        # will read the dimensionality from there.
        length_scale_prior = np.atleast_2d(length_scale_prior)
        if len(length_scale_prior) == 1:
            length_scale_prior = np.array(list([length_scale_prior[0]]) * self.d)
        elif len(length_scale_prior) != self.d:
            raise TypeError(
                "length_scale_prior needs to define 1d bounds (common for all dimensions)"
                "or as many bounds as dimensions"
            )
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            output_scale_prior=output_scale_prior,
            length_scale_prior=length_scale_prior,
            noise_level=self.noise_level,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        # Iteration index, use if specified only
        self._i_iter = np.empty((0,), dtype=int)
        # Masks for being used at any time for which part of the model
        self._i_regress = np.empty((0,), dtype=bool)
        # Trackers for last-appended points
        self.n_last_appended = 0
        self.n_last_appended_finite = 0
        if self.verbose >= 3:
            print("Initializing GP with the following options:")
            print("===========================================")
            print(str(self))

    def __str__(self):
        return (
            "* Kernel:\n"
            + f"   {str(self.gpr.kernel)}\n"
            + "  with hyperparameters:\n"
            + "    -"
            + "\n    -".join(
                str(h) for h in self.gpr.kernel.hyperparameters
            )  # print in direct scale
            + "\n"
            + f"* Noise level: {self.noise_level}"
            # + f"* Optimizer: {self.optimizer}"
            # + f"* Optimizer restarts: {self.n_restarts_optimizer}"
            + f"* X-preprocessor: {self.preprocessing_X is not None}"
            + f"* y-preprocessor: {self.preprocessing_y is not None}"
            + f"* SVM to account for infinities: {bool(self.infinities_classifier)}"
        )

    @property
    def d(self):
        """Dimension of the feature space."""
        return self._X.shape[1]

    @property
    def X(self):
        """
        Coordinates of all the points used to train the model (returns a copy).
        """
        return np.copy(self._X)

    @property
    def y(self):
        """
        Log-posterior of all the points used to train the model (returns a copy).
        """
        return np.copy(self._y)

    @property
    def X_init(self):
        """
        Coordinates of the points passed at initialisaion (returns a copy).
        """
        return np.copy(self._X[np.where(self._i_iter==0)])

    @property
    def y_init(self):
        """
        Log-posterior of the points passed at initialisaion (returns a copy).
        """
        return np.copy(self._y[np.where(self._i_iter==0)])

    @property
    def X_regress(self):
        """
        Coordinates of the points used to train the GPR (returns a copy).
        """
        return np.copy(self._X[self._i_regress])

    @property
    def y_regress(self):
        """
        Log-posterior of the points used to train the GPR (returns a copy).
        """
        return np.copy(self._y[self._i_regress])

    @property
    def X_class(self):
        """
        Coordinates of the points used to train the infinities classifier (returns a
        copy).
        """
        return np.copy(self._X)

    @property
    def y_class(self):
        """
        Log-posterior of the points used to train the infinities classifier (returns a
        copy).
        """
        return np.copy(self._y)

    @property
    def X_infinite(self, ignore_trust_region=False):
        """
        Coordinates of the points classified as infinitely small (returns a copy).
        """
        return np.copy(
            self._X[
                ~self._i_finite(
                    X=self._X,
                    y=self._y,
                    X_=self._X_,
                    y_=self._y_,
                    validate=True,
                    ignore_trust_region=ignore_trust_region,
                )
            ]
        )

    @property
    def n_total(self):
        """
        Number of all the points used to train the model.
        """
        return len(self._y)

    @property
    def n_init(self):
        """
        Number of points passed at evaluation.
        """
        return len(np.where(self._i_iter==0)[0])

    @property
    def n_regress(self):
        """
        Number of points used to train the GPR.
        """
        return np.sum(self._i_regress)

    @property
    def n_class(self):
        """
        Number of points used to train the infinities classifier.
        """
        return len(self._y)

    @property
    def y_max(self):
        """
        The maximum log-posterior value in the training set.

        Returns -inf (regularized if applicable) if not points have been added.
        """
        try:
            return np.max(self._y)
        except ValueError:
            return self.minus_inf_value

    @property
    def trust_bounds(self):
        """
        Bounds of the trust region, i.e. the hyperrectable where the log-posterior is
        expected to be "finite", defined as being used by the GPR.
        """
        return np.copy(self.trust_region.bounds)

    @property
    def fitted(self):
        """Whether the GPR hyperparameters have been fitted at least once."""
        return self._fitted

    @property
    def last_appended(self):
        """
        Returns a copy of the last appended training points (finite/accepted or not),
        as (X, y).
        """
        if self.infinities_classifier is None:
            return self.last_appended_finite
        return (
            np.copy(self._X[-self.n_last_appended :]),
            np.copy(self._y[-self.n_last_appended :]),
        )

    @property
    def last_appended_finite(self):
        """Returns a copy of the last appended GPR (finite) training points, as (X, y)."""
        return (
            np.copy(self._X[self._i_regress][-self.n_last_appended_finite :]),
            np.copy(self._y[self._i_regress][-self.n_last_appended_finite :]),
        )

    @property
    def scales(self):
        """
        GPR Kernel scales as ``(output_scale, (length_scale_1, ...))`` in non-transformed
        coordinates.
        """
        return (
            self.preprocessing_y.inverse_transform_scale(
                np.sqrt(self.gpr.kernel_.k1.constant_value)
            ),
            tuple(
                self.preprocessing_X.inverse_transform_scale(
                    self.gpr.kernel_.k2.length_scale
                )
            ),
        )

    def training_set_as_df(self, param_names=None):
        """
        Returns the training set as a pandas DataFrame (created on-the-fly and not saved).

        If ``param_names`` are not passed, generic ones will be used.
        """
        use_X = self._X.copy()
        use_y = self._y.copy()
        if param_names is None:
            param_names = generic_params_names(self.d)
        data = dict(zip(param_names, use_X.T))
        data["y"] = use_y
        data["added_iter"] = self._i_iter
        data["is_regress"] = self._i_regress
        data["is_finite"] = self.is_finite(data["y"])
        return pd.DataFrame(data)

    @property
    def abs_finite_threshold(self):
        """
        Absolute threshold for ``y`` values to be considered finite.
        """
        return self.preprocessing_y.inverse_transform_scale(
            self.infinities_classifier.abs_threshold
        )

    @staticmethod
    def compute_threshold_given_sigma(n_sigma, n_dimensions):
        r"""
        Computes threshold value given a number of :math:`\sigma` away from the maximum,
        assuming a :math:`\chi^2` distribution.
        """
        return delta_logp_of_1d_nstd(n_sigma, n_dimensions)

    @staticmethod
    def _diff_threshold_if_keep_n_finite(y, n, reference_diff_threshold, epsilon=1e-6):
        """
        Recalculation of the relative threshold when imposing that at least ``n`` points
        must be kept finite (i.e. in the training set). The "at least" imposes that the
        minimum value returned is the given threshold.

        It requires sorting the ``y`` values, which can be costly in extreme cases.
        """
        # Keep fewer than n if there are less than n finite points, and some actual
        # infinities.
        # Open question: return -inf or the threshold just below the last finite <n-th
        # point? Maybe throw warning.
        if n is None or n <= 1:
            return reference_diff_threshold
        y_sorted = np.sort(y)
        difference_to_nth_point = y_sorted[-1] - y_sorted[-min(n, len(y_sorted))]
        return max(reference_diff_threshold, difference_to_nth_point + epsilon)

    def is_finite(self, y):
        """
        Returns the classification of y (target) values as finite (True) or not, by
        comparing them with the current threshold.

        Notes
        -----
        Use this method instead of the equivalent one of the 'infinities_classifier'
        attribute, since the arguments of that one may need to be transformed first.

        If calling with an argument which is not either the training set or a subset of it
        results may be inconsistent, since new values may modify the threshold.
        """
        if self.infinities_classifier is None:
            return np.full(shape=len(y), fill_value=True)
        return self.infinities_classifier.is_finite(self.preprocessing_y.transform(y))

    def predict_is_finite(self, X, validate=True):
        """
        Returns a prediction for the classification of the target value at some given
        parameters.

        Notes
        -----
        Use this method instead of the equivalent one of the 'infinities_classifier'
        attribute, since the arguments of that one may need to be transformed first.
        """
        if self.infinities_classifier is None:
            return np.full(shape=(len(self._y),), fill_value=True)
        return self.infinities_classifier.predict(
            np.ascontiguousarray(self.preprocessing_X.transform(X)), validate=validate
        )

    def _i_finite(
        self, X=None, y=None, X_=None, y_=None, validate=True, ignore_trust_region=False
    ):
        """
        Internal method to classify points as finite (True) or not (False) based on the
        infinities classifier and the Trust Region (if ``ignore_trust_region`` is False).

        It can be passed any combination of transformed or untransformed points.

        It performs no checks on the input.
        """
        if X is None and X_ is None:
            raise ValueError("Pass either X or X_ (transformed).")
        if X_ is None:
            X_ = self.preprocessing_X.transform(X)
        if self.infinities_classifier is None:
            if y_ is not None:
                finite = ~np.isinf(y_)
            else:  # no y passed: assume all finite
                finite = np.full(len(X_), True, dtype=bool)
        else:
            finite = self.infinities_classifier.predict(
                np.ascontiguousarray(X_), validate=validate
            )
        if not ignore_trust_region:
            if X is None:
                X = self.preprocessing_X.inverse_transform(X)
            # Since we are doing and & operation with SVM, we can pass -inf locations
            finite &= self.trust_region.predict(X, validate=validate)
        return finite

    def set_random_state(self, random_state):
        """
        (Re)sets the random state, including the SVM, if present.
        """
        self.random_state = random_state
        if self.infinities_classifier:
            # In the SVM case, since we have not wrapper the calls to the RNG,
            # (as we have for the GPR), we need to repackage the new numpy Generator
            # as a RandomState, which is achieved by gpry.tools.check_random_state
            self.infinities_classifier.random_state = check_random_state(
                random_state, convert_to_random_state=True
            )

    def _validate_noise_level(self, noise_level, n_train):
        """
        Checks for type and inconsistencies for the given noise level. Separated from the
        update method to be performed before updating class attributes.

        Returns a validated value that can be used directly.
        """
        if n_train == 0 and noise_level is not None:
            raise ValueError("noise_level must be None if not fitting to new points.")
        if np.iterable(noise_level):
            noise_level = np.atleast_1d(noise_level)
            if noise_level.shape[0] != n_train:
                raise ValueError(
                    "noise_level must be an array with same number of entries as y, but "
                    f"len(n)={noise_level.shape[0]} != len(y)={n_train})"
                )
        elif isinstance(noise_level, Number):
            if np.iterable(self.noise_level):
                noise_level = np.full(fill_value=noise_level, shape=(n_train,))
        elif noise_level is None:
            if np.iterable(self.noise_level):
                raise ValueError(
                    "Need to pass non-null noise_level (scalar or array) because concrete"
                    " values were given earlier for the training points."
                )
        else:
            raise ValueError(
                "noise_level needs to be an iterable, number or None. "
                f"Got type(noise_level)={type(noise_level)}"
            )
        return noise_level

    def _update_noise_level(self, noise_level):
        """
        Updates the noise level of the training set with the new values (or the lack
        thereof).

        Assumes possible inconsistencies dealt with by ``_validate_noise_level``.
        """
        if np.iterable(noise_level):
            if not np.iterable(self.noise_level):
                if self.verbose > 1:
                    warnings.warn(
                        "A new noise level has been assigned to the updated training set "
                        "while the old training set has a single scalar noise level: "
                        f"{self.noise_level}. Converting to individual levels!"
                    )
                self.noise_level = np.full(
                    fill_value=self.noise_level, shape=(len(self._y),)
                )
            self.noise_level = np.append(self.noise_level, noise_level, axis=0)
        elif isinstance(noise_level, Number):
            # NB at validation new=scalar has been converted to array if old=array
            assert not np.iterable(self.noise_level)
            if not np.isclose(noise_level, self.noise_level):
                if self.verbose > 1:
                    warnings.warn(
                        "Overwriting the noise level with a scalar. Make sure that "
                        "kernel's hyperparamters are refitted."
                    )
                self.noise_level = noise_level
        elif noise_level is None:
            pass  # keep old level for new points.

    def append(
        self,
        X,
        y,
        noise_level=None,
        fit_gpr=True,
        fit_classifier=True,
        validate=True,
        i_iter=None,
    ):
        r"""
        Append newly acquired data to the GPR and the infinities classifier, and updates
        them and the preprocessors.

        Here updating of the GPR refers to the re-calculation of the the GPR inverse
        matrix :math:`(K(X,X)+\sigma_n^2 I)^{-1}` which is needed for predictions.

        The highest cost incurred by this method is the refitting of the GPR kernel
        hyperparameters :math:`\theta`. It can be useful to disable it (``fit_gpr=False``)
        in cases where it is worth saving the computational expense in exchange for a loss
        of information, such as when performing parallelized active sampling (NB: this is
        only possible when the GPR hyperparameters have been fit at least once).

        An intermediate option is to perform a single GPR hyperparameter optimization run
        (instead of the default number of restarts) from the current hyperparameter
        values, using ``fit_gpr='simple'``.

        For an additional speed boost, the refitting of the infinities classifier (if
        present) can be disabled with ``fit_classifier=False`` (if a GPR refit is
        requested this value is overridden).

        If called with ``X=None, y=None``, it re-fits the model without adding new points.

        The following calls should then be equivalent:

        .. code-block:: python

           fit_gpr_kwargs = {"n_restarts": 10}
           # A
           gpr.append(new_X, new_y, fit_gpr=fit_gpr_kwargs)
           # B
           gpr.append(new_X, new_y, fit_gpr=False)
           gpr.fit_gpr_hyperparameters(**fit_gpr_kwargs)
           # C
           gpr.append(new_X, new_y, fit_gpr=False, fit_classifier=False)
           gpr.append(None, None, fit_gpr=fit_gpr_kwargs)


        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features), or None
            Training data to append to the model.

        y : array-like, shape = (n_samples, [n_output_dims]), or None
            Target values to append to the data

        noise_level : array-like, shape = (n_samples, [n_output_dims])
            Uncorrelated standard deviations to add to the diagonal part of the covariance
            matrix. Needs to have the same number of entries as y. If None, the
            noise_level set in the instance is used. If you pass a single number the noise
            level will be overwritten. In this case it is advisable to refit the
            hyperparameters of the kernel.

        fit_gpr : Bool or 'simple', dict, optional (default: True)
            Whether the GPR :math:`\theta`-parameters are optimised (``'simple'`` for a
            single run from last optimum; ``True`` for a more thorough search with
            multiple restarts), or a simple kernel matrix inversion is performed
            (``False``) with constant hyperparameters. Can also be passed a dict with
            arguments to be passed to the `fit_gpr_hyperparameters` method.

        fit_classifier: Bool, optional (default: True)
            Whether the infinities classifier is refit. Overridden to ``True`` if
            ``fit_gpr`` is not ``False``.

        i_iter : int or array-like, shape = (n_samples), optional
            An index to be assigned to the newly-appended points. If None, assigns the
            last entry +1.

        Returns
        -------
        self
            Returns an instance of self.
        """
        if validate and (
            self.gpr.kernel is None or self.gpr.kernel.requires_vector_input
        ):
            X = check_array(X, ensure_2d=True, dtype="numeric")
            y = check_array(y, ensure_2d=False, dtype="numeric")
        elif validate:
            X = check_array(X, ensure_2d=False, dtype=None)
            y = check_array(y, ensure_2d=False, dtype=None)
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                f"Different numbers of points in X (shape {X.shape}) and y (shape "
                f"{y.shape})."
            )
        # Ensure fit_gpr --> fit_classifier --> fit_preprocessors
        fit_preprocessors = False
        fit_gpr_kwargs = None
        if fit_gpr is True:
            fit_classifier = True
            fit_gpr_kwargs = {}
        elif str(fit_gpr) == "simple":
            fit_classifier = True
            fit_gpr_kwargs = {"simple": True}
            fit_gpr = True
        elif isinstance(fit_gpr, Mapping):
            fit_classifier = True
            fit_gpr_kwargs = deepcopy(fit_gpr)
            fit_gpr = True
        elif fit_gpr is not False:
            raise ValueError(
                "`fit_gpr` needs to be bool, 'simple', or a dict of args for the "
                f"`fit_gpr_hyperparameters` method. Got {fit_gpr}."
            )
        if fit_classifier:
            fit_preprocessors = True
        force_fit_gpr = False  # to avoid skipping fit if no points added if X,y = None
        if X is None and y is None:
            X, y = np.empty((0, self.d)), np.empty((0,))
            force_fit_gpr = fit_gpr
            if noise_level is not None:
                raise ValueError("Cannot give a noise level if X and y are not given.")
            # self._i_iter does not change
        elif X is None or y is None:  # (None, None) already excluded
            raise ValueError("If passing X, y needs to be passed too, and viceversa.")
        else:  # new points to be added
            if i_iter is None:
                if not len(self._i_iter):
                    i_iter = 0
                else:
                    i_iter = self._i_iter[-1] + 1
            self._i_iter = np.append(self._i_iter, len(y) * [i_iter])
        noise_level_valid = self._validate_noise_level(noise_level, len(y))
        # NB: if called with X, y = None, None, we could also have adopted the convention
        #     that the "last"-named variables refer to the last call with non-null X, y,
        #     but for now they are reset at every call, turning into 0 if no points given.
        self.n_last_appended = len(y)
        self._X = np.append(self._X, X, axis=0)
        self._y = np.append(self._y, y)
        self._update_noise_level(noise_level_valid)
        # 1. Fit preprocessors with finite points and select finite points in the process,
        #    and create transformed training set and noises.
        # NB: which points are finite does not change after SVM refit (as long as
        #     y-preprocessor is liner), so we can select them now.
        if self.infinities_classifier is None:
            self._i_regress = np.full(fill_value=True, shape=(len(self._y),))
        else:
            # Use the manual method for non-preprocessed input.
            # Make sure that the threshold is such that there is a min of finite ones.
            diff_threshold_keep_n = self._diff_threshold_if_keep_n_finite(
                self._y, self.keep_min_finite, self._diff_threshold
            )
            self._i_regress = self.infinities_classifier._is_finite_raw(
                self._y, diff_threshold_keep_n
            )
        # Apply the trust region to the *finite* points only (does nothing if not defined)
        self._i_regress[self._i_regress] &= self.trust_region.predict(
            self._X[self._i_regress]
        )
        if fit_preprocessors:
            self.preprocessing_X.fit(self._X[self._i_regress], self._y[self._i_regress])
            self.preprocessing_y.fit(self._X[self._i_regress], self._y[self._i_regress])
        self._X_ = self.preprocessing_X.transform(self._X)
        self._y_ = self.preprocessing_y.transform(self._y)
        # The transformed noise level is always an array.
        noise_level_array = (
            np.full(fill_value=self.noise_level, shape=(len(self._y),))
            if isinstance(self.noise_level, Number)
            else self.noise_level
        )
        self.noise_level_ = self.preprocessing_y.transform_scale(noise_level_array)
        # 2. Fit the classifiers: SVM in the transformed space, and trust region
        if self.infinities_classifier is None:
            is_finite_last_appended = np.full(
                fill_value=True, shape=(self.n_last_appended,)
            )
        else:
            if fit_classifier:
                # Again, make sure that we keep a min number of finite points
                diff_threshold_keep_n_ = self.preprocessing_y.transform_scale(
                    diff_threshold_keep_n
                )
                # The SVM lives in the preprocessed space, and the preprocessor may have
                # changed, so we need to pass all points every time
                is_finite_predict = self.infinities_classifier.fit(
                    self._X_, self._y_, diff_threshold_keep_n_
                )
                # The trust region lives in the original space with finite points only
                self.trust_region.fit(
                    self._X[self._i_regress], self._y[self._i_regress]
                )
                is_finite_predict &= self.trust_region.predict(self._X)
                assert np.array_equal(self._i_regress, is_finite_predict), (
                    "Infinities classifier miss-classified at least 1 point."
                )
            # Even if assert test fails, use the real classification
            is_finite_last_appended = self._i_regress[-self.n_last_appended :]
        # The number of newly added points. Used for the _update_model method
        self.n_last_appended_finite = sum(is_finite_last_appended)
        # If all added values are infinite there's no need to refit the GPR,
        # unless an explicit call for that with X, y = None was made
        if not self.n_last_appended_finite and not force_fit_gpr:
            self._fitted = True
            return self
        # 3. Re-fit the GPR in the transformed space, and maybe hyperparameters
        self.gpr.fit(
            X=self._X_[self._i_regress],
            y=self._y_[self._i_regress],
            noise_level=self.noise_level_[self._i_regress],
            fit_hyperparameters=fit_gpr_kwargs,
            validate=False,
        )
        self._fitted = True
        return self

    def predict(
        self,
        X,
        return_std=False,
        return_mean_grad=False,
        return_std_grad=False,
        validate=True,
        ignore_trust_region=False,
    ):
        """
        Predict output for X.

        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True),
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

        validate : bool, default: True
            If False, ``X`` is assumed to be correctly formatted (2-d float array, with
            points as rows and dimensions/features as columns, C-contiguous), and no
            checks are performed on it. Reduces overhead. Use only for repeated calls when
            the input is programmatically generated to be correct at each stage.

        ignore_trust_region : bool (default: False)
            If ``True`` and trust-region definition was used (``trust_region_factor``
            defined at initialisation), it ignores the trust region and does not return
            a negative infinity outside the trust region.

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
            raise ValueError("Mean grad and std grad not implemented for n_samples > 1")
        if validate and (
            self.gpr.kernel is None or self.gpr.kernel.requires_vector_input
        ):
            X = check_array(X, ensure_2d=True, dtype="numeric")
        elif validate:
            X = check_array(X, ensure_2d=False, dtype=None)
        # Create placeholders with default values
        X = np.copy(X)  # copy since preprocessors might change it
        n_samples = X.shape[0]
        X_ = self.preprocessing_X.transform(X)
        return_dict = {"mean": np.full(n_samples, self.minus_inf_value)}
        if return_std:
            return_dict["std"] = np.zeros(n_samples)  # std is zero when mu is -inf
        if return_mean_grad:
            return_dict["mean_grad"] = np.full((n_samples, self.d), self.inf_value)
        if return_std_grad:
            return_dict["std_grad"] = np.zeros((n_samples, self.d))
        # First check if either SVM or the trust region say that the value should be -inf
        finite = self._i_finite(
            X=X, X_=X_, validate=validate, ignore_trust_region=ignore_trust_region
        )
        # If all values are infinite no need to run the prediction through the GP
        if np.all(~finite):
            if len(return_dict) == 1:
                return return_dict["mean"]
            return list(return_dict.values())
        # else: at least some finite points --> GPR predict for them only
        return_gpr_ = self.gpr.predict(
            X_[finite],
            return_std=return_std,
            return_mean_grad=return_mean_grad,
            return_std_grad=return_std_grad,
            validate=validate,
        )
        return_gpr_ = self._regressor_output_to_dict(
            return_gpr_, return_std, return_mean_grad, return_std_grad
        )
        for k, v in return_gpr_.items():
            return_dict[k][finite] = self.preprocessing_y.inverse_transform(v)
        # Apply inverse transformation twice for std grad
        if "std_grad" in return_dict:
            return_dict["std_grad"][finite] = self.preprocessing_y.inverse_transform(
                return_dict["std_grad"][finite]
            )
        # Upper clipping to avoid overshoots
        if self.clip_factor is not None:
            upper = self.clip_factor * max(self._y[self._i_regress]) - (
                self.clip_factor - 1
            ) * min(self._y[self._i_regress])
            return_dict["mean"] = np.clip(return_dict["mean"], None, upper)
        if len(return_dict) == 1:
            return return_dict["mean"]
        return list(return_dict.values())

    @staticmethod
    def _regressor_output_to_dict(
        return_value, return_std, return_mean_grad, return_std_grad
    ):
        """
        Convert the return value of the regressor's ``predict`` method into a dict with
        with the requested products as keys.
        """
        return_dict = {"mean": return_value[0]}
        i = 1
        if return_std:
            return_dict["std"] = return_value[i]
            i += 1
        if return_mean_grad:
            return_dict["mean_grad"] = return_value[i]
            i += 1
        if return_std_grad:
            return_dict["std_grad"] = return_value[i]
        return return_dict

    def logp(
        self,
        X,
        validate=True,
        ignore_trust_region=False,
    ):
        """
        Returns the surrogate log-posterior.
        """
        if validate and (
            self.gpr.kernel is None or self.gpr.kernel.requires_vector_input
        ):
            X = check_array(X, ensure_2d=True, dtype="numeric")
        elif validate:
            X = check_array(X, ensure_2d=False, dtype=None)
        return self.predict(X, validate=False, ignore_trust_region=ignore_trust_region)

    def predict_std(self, X, validate=True, ignore_trust_region=False):
        """
        Predict output standart deviation for X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated.

        validate : bool, default: True
            If False, ``X`` is assumed to be correctly formatted (2-d float array, with
            points as rows and dimensions/features as columns, C-contiguous), and no
            checks are performed on it. Reduces overhead. Use only for repeated calls when
            the input is programmatically generated to be correct at each stage.

        Returns
        -------
        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.
        """
        self.n_eval += len(X)
        if validate and (
            self.gpr.kernel is None or self.gpr.kernel.requires_vector_input
        ):
            X = check_array(X, ensure_2d=True, dtype="numeric")
        elif validate:
            X = check_array(X, ensure_2d=False, dtype=None)
        # Create placeholders with default values
        X = np.copy(X)  # copy since preprocessors might change it
        n_samples = X.shape[0]
        X_ = self.preprocessing_X.transform(X)
        std = np.zeros(n_samples)  # std is zero when mu is -inf
        # First check if either SVM or the trust region say that the value should be -inf
        finite = self._i_finite(
            X=X, X_=X_, validate=validate, ignore_trust_region=ignore_trust_region
        )
        # If all values are infinite no need to run the prediction through the GP
        if np.all(~finite):
            return std
        # else: at least some finite points --> GPR predict for them only
        std_ = self.gpr.predict_std(X_[finite], validate=validate)
        std[finite] = self.preprocessing_y.inverse_transform(std_)
        return std


class TrustRegion:
    """
    Class managing the region within we trust the model to be finite, as a hyper-rectangle
    defined by the points with log-posterior over some value. That value is normally taken
    as a maximum difference between any point and the highest log-posterior in the
    surrogate model's training set. That maximum difference scales with dimensionality by
    means of being defined as the log-posteior value corresponding to some confidence
    level in a Gaussian approximation.

    Conceptually part of the "infinities classifier".

    Parameters
    ----------

    factor : float (positive), optional
        If defined as a positive float, enlarges the minimal hyper-rectangle by the
        given factor. If undefined, the bounds passed at initialization are kept.

    nstd : float, optional
        If defined as a positive float, the definition of the trust region only takes
        into account points corresponding to a significance (assuming a Gaussian
        posterior) equivalent to this value in 1d standard deviations, taken as a
        difference in log-posterior with respect to the highest point passed.

    n_min : int, optional
        If both this and ``nstd`` are defined, it ensures that at least ``n_min``
        points are within the hyper-rectangle if that many are available, temporarily
        lowering the ``nstd`` if necessary.
    """

    def __init__(self, bounds, factor=None, nstd=None, n_min=None):
        self._init_bounds = np.atleast_2d(bounds)
        self._trust_bounds = np.copy(self._init_bounds)
        if factor is not None and factor <= 0:
            raise TypeError("Please pass a positive 'factor' or None.")
        self.factor = factor
        if nstd is not None and nstd <= 0:
            raise TypeError("Please pass a positive 'nstd' or None.")
        self.nstd = nstd
        if (n_min is not None and not isinstance(n_min, int)) or n_min <= 0:
            raise TypeError("Please pass an integer, positive 'n_min' or None.")
        self.n_min = n_min

    @property
    def d(self):
        """Dimension of the space."""
        return self._init_bounds.shape[0]

    @property
    def bounds(self):
        """
        Returns a copy of the trust bounds (the original ones if nevet fit or
        ``factor=None``).
        """
        return np.copy(self._trust_bounds)

    @property
    def trivial(self):
        """
        Returns ``True`` if the trust region and the prior are by definition equivalent.
        """
        return self.factor is None

    def fit(self, X, y, validate=True):
        """
        Adjusts the boundaries of the trust region.

        All points are assumed to be of interest, i.e. no check for ``y`` being finite.
        """
        if self.trivial:
            return
        if validate:
            X = check_array(X, ensure_2d=True, dtype="numeric")
            y = check_array(y, ensure_2d=False, dtype="numeric")
            if X.shape[0] != y.shape[0]:
                raise TypeError(
                    f"Different numbers of points in X (shape {X.shape}) and y (shape "
                    f"{y.shape})."
                )
        if self.nstd is None:
            use_X = X
        elif (self.n_min, self.nstd) != (None, None) and len(X) <= self.n_min:
            use_X = X
        else:  # nstd used and more than enough points
            nstd_ = self.nstd
            use_X = np.empty(shape=(0, self.d))
            while len(use_X) < min(self.d, self.n_finite):
                delta_y = delta_logp_of_1d_nstd(self.nstd, self.d)
                use_X = X[np.where(max(y) - y < delta_y)]
                nstd_ = nstd_ + 0.1
        self._trust_bounds = shrink_bounds(self._init_bounds, use_X, factor=self.factor)

    def predict(self, X, validate=True):
        """
        Returns ``True`` for locations inside the trust region, and ``False`` otherwise.
        """
        return is_in_bounds(X, self._trust_bounds, check_shape=validate)
