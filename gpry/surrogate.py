"""
Class holding the surrogate model.

Underscored-after attributes mean "preprocessed/transformed"
"""

# Builtin
import warnings
from copy import deepcopy
from typing import Mapping, Sequence
from numbers import Number

# External
import numpy as np
import pandas as pd  # type: ignore
from sklearn.utils.validation import check_array  # type: ignore

# Local
from gpry.gpr import GaussianProcessRegressor
from gpry.preprocessing import DummyPreprocessor
from gpry.tools import delta_logp_of_1d_nstd, generic_params_names
from gpry.infinities_classifier import InfinitiesClassifiers


class SurrogateModel:
    r"""
    Object holding the Gaussian Process Regressor, and, if applicable, the
    input/output preprocessing layer and the infinities classifier.

    Parameters
    ----------

    bounds : array-like, shape=(n_dims,2)
        Array of bounds of the prior [lower, upper] along each dimension.

    preprocessing_X : X-preprocessor, Pipeline_X, optional (default: None)
        Single preprocessor or pipeline of preprocessors for X. If None is
        passed the data is not preprocessed.

    preprocessing_y : y-preprocessor or Pipeline_y, optional (default: None)
        Single preprocessor or pipeline of preprocessors for y. If None is
        passed the data is not preprocessed.

    regressor : dict
        Dictionary of options to iniitalise the :class:`gpr.GaussianProcessRegressor``.

    clip_factor : float, optional (default: 1.1)
        Factor for upper clipping of the regressor's predictions, to avoid overshoots.

    infinities_classifier : str, Sequence, dict
        Dictionary of options to initialise the
        :class:`infinities_classifier.InfinitiesClassifiers`, used for selecting a
        subset of points to be used to train the regressor, based on their target
        value. Alternative, the name of a single classifier, or a list of names, can
        be passed.

    random_state : int or numpy.random.Generator, optional
        The generator used to perform random operations of the GPR. If an integer is
        given, it is used as a seed for the default global numpy random number generator.

    verbose : 1, 2, 3, optional (default: 1)
        Level of verbosity of the GP. 3 prints Infos, Warnings and Errors, 2
        Warnings and Errors, and 1 only Errors. Should be set to 2 or 3 if
        problems arise.

    Attributes
    ----------
    d : int
        Dimensionality of the training data.

    bounds : array
        The bounds with which the GPR was defined.

    trust_bounds : array or None
        The bounds of a smaller trust region possibly defined by a classifier (in
        particular if :class:`infinities_classifer.TrustRegion` is being used.
        Otherwise returns the original prior bounds.

    fitted : bool
        True whenever the the surrogate model has already been fitted to some points.

    n_total : int
        Number of points/features in the training set of the model, including points with
        target values classified as infinite.

    X : array-like, shape = (n_samples, n_features)
        Original (untransformed) feature values in training data of the GPR. Intended to
        be used when one wants to access the training data for any purpose.

    y : array-like, shape = (n_samples)
        Original (untransformed) target values in training data of the GPR. Intended to be
        used when one wants to access the training data for any purpose.

    X_last_appended : array-like, shape = (n_samples, n_features)
        Original (untransformed) feature values added in the training set in the last
        call to :meth:`Surrogate.append`.

    y_last_appended : array-like, shape = (n_samples)
        Original (untransformed) targer values added in the training set in the last
        call to :meth:`Surrogate.append`.

    n_init : int
        Number of points/features provided as an initial training set.

    X_init : array-like, shape = (n_samples, n_features)
        Original (untransformed) feature values provided as an initial training set.

    y_init : array-like, shape = (n_samples)
        Original (untransformed) target values provided as an initial training set.

    n_regress : int
        Number of points/features currently in use to train the GP Regressor.

    X_regress : array-like, shape = (n_samples, n_features)
        Original (untransformed) feature values currently in use to train the GP
        Regressor.

    y_regress : array-like, shape = (n_samples)
        Original (untransformed) target values currently in use to train the GP
        Regressor.

    X_last_appended_regress : array-like, shape = (n_samples, n_features)
        Original (untransformed) feature values added in the training set in the last
        call to :meth:`Surrogate.append` and accepted into the Regressor.

    y_last_appended_regress : array-like, shape = (n_samples)
        Original (untransformed) targer values added in the training set in the last
        call to :meth:`Surrogate.append` and accepted into the Regressor.

    X_infinite : array-like, shape = (n_samples, n_features)
        Original (untransformed) feature values whose target log-posterior has been
        classified as non-finite.

    y_infinite : array-like, shape = (n_samples)
        Original (untransformed) target values that have been classified as non-finite.

    y_max : float
        Maximum target (log-posterior) value in the training set.

    noise_level : array-like, shape = (n_samples, [n_output_dims]) or scalar
        The noise level (square-root of the variance) of the uncorrelated
        training data, un-transformed. Returns a copy.

    scales : tuple(float, (float, float, ...))
        GPR kernel scales in untransformed space as
        ``(output_scale, (length_scale1, ...))``.

    abs_finite_threshold : float
        Absolute threshold above which log-posteriors are considered finite.

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
        bounds=None,
        preprocessing_X=None,
        preprocessing_y=None,
        regressor=None,
        infinities_classifier=None,
        clip_factor=1.1,
        random_state=None,
        verbose=1,
    ):
        self._bounds = np.atleast_2d(bounds)
        self.preprocessing_X = (
            DummyPreprocessor if preprocessing_X is None else preprocessing_X
        )
        self.preprocessing_y = (
            DummyPreprocessor if preprocessing_y is None else preprocessing_y
        )
        self.n_eval = 0
        self.verbose = verbose
        self._fitted = False
        # Arrays containing the evaluations used to train the model,
        # regardless of whether they are used and for what.
        self._X = np.empty((0, self.d), dtype=float)
        self._y = np.empty((0,), dtype=float)
        self._i_y_sorted = np.empty((0,), dtype=int)
        # Same, in the transformed space
        self._X_, self._y_ = None, None
        # Iteration index, use if specified only
        self._i_iter = np.empty((0,), dtype=int)
        # Masks for being used at any time for which part of the model
        self._i_regress = np.empty((0,), dtype=bool)
        # Trackers for last-appended points
        self.n_last_appended = 0
        self.n_last_appended_finite = 0
        # Dealing with infinities: upper and lower caps, and classifier(s)
        self.inf_value = np.inf
        self.minus_inf_value = -np.inf
        if not infinities_classifier:
            self.infinities_classifier = None
        else:
            if not getattr(self.preprocessing_y, "is_linear", False):
                warnings.warn(
                    "If using a standard classifier for infinities, the y-preprocessor "
                    "needs to be linear (declare an attr ``is_linear=True``). This may "
                    "lead to errors further in the pipeline."
                )
            if isinstance(infinities_classifier, str):
                infinities_classifier = {infinities_classifier.lower(): None}
            elif isinstance(infinities_classifier, Sequence):
                infinities_classifier = {k: None for k in infinities_classifier}
            elif not isinstance(infinities_classifier, Mapping):
                raise TypeError(
                    "'infinities_classifier': must be dict of '{classifier: {options}'."
                )
            # The infinities classifier lives in the transformed space:
            bounds_ = self.preprocessing_X.transform_bounds(self._bounds)
            # In this first call, if preprocessor not fitted, use dummy one
            if self.preprocessing_y.fitted:
                preprocessing_y = self.preprocessing_y
            else:
                preprocessing_y = DummyPreprocessor
            nstd_calculator = lambda nsigma: preprocessing_y.transform_scale(
                delta_logp_of_1d_nstd(nsigma, self.d)
            )
            try:
                self.infinities_classifier = InfinitiesClassifiers(
                    bounds=bounds_,
                    nstd_calculator=nstd_calculator,
                    random_state=random_state,
                    **infinities_classifier,
                )
            except Exception as excpt:
                raise TypeError(f"Error initialising infinities classifier: {excpt}")
        # Initialising the GP Regressor
        # NB: hyperparameter priors and noise level are understood in transformed space
        # Make sure the length scale prior is unfolded per-dimensionality, since the GPR
        # will read the dimensionality from there.
        length_scale_prior = np.atleast_2d(regressor["length_scale_prior"])
        if len(length_scale_prior) == 1:
            length_scale_prior = np.array(list([length_scale_prior[0]]) * self.d)
        elif len(length_scale_prior) != self.d:
            raise TypeError(
                "length_scale_prior needs to define 1d bounds (common for all dimensions)"
                "or as many bounds as dimensions"
            )
        regressor["length_scale_prior"] = length_scale_prior
        # This is the default "noise_level" of the regressor, understood as the common
        # sigma_y of the training samples. At this pint, it's a single number.
        self._noise_level = float(regressor["noise_level"])
        self._noise_level_ = None
        self.gpr = GaussianProcessRegressor(
            kernel=regressor["kernel"],
            output_scale_prior=regressor["output_scale_prior"],
            length_scale_prior=regressor["length_scale_prior"],
            noise_level=self._noise_level,
            optimizer=regressor["optimizer"],
            n_restarts_optimizer=regressor["n_restarts_optimizer"],
            random_state=random_state,
        )
        # Regressor post-processing: clip too high values
        self.clipper = Clipper(clip_factor)
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
            + f"* Noise level: {self._noise_level}\n"
            # + f"* Optimizer: {self.optimizer}"
            # + f"* Optimizer restarts: {self.n_restarts_optimizer}"
            + f"* X-preprocessor: {self.preprocessing_X is not None}\n"
            + f"* y-preprocessor: {self.preprocessing_y is not None}\n"
            + f"* Classifiers for infinities: {self.infinities_classifier}\n"
        )

    @property
    def d(self):
        """Dimension of the feature space."""
        return len(self._bounds)

    @property
    def bounds(self):
        """Copy of the problem's prior bounds."""
        return np.copy(self._bounds)

    @property
    def trust_bounds(self):
        """
        Bounds of the trust region, i.e. the hyperrectable where the log-posterior is
        expected to be "finite", defined as being used by the GPR.
        """
        if self.infinities_classifier is not None:
            return self.preprocessing_X.inverse_transform_bounds(
                self.infinities_classifier.trust_bounds
            )
        return np.copy(self._bounds)

    @property
    def fitted(self):
        """Whether the GPR hyperparameters have been fitted at least once."""
        return self._fitted

    @property
    def n_total(self):
        """
        Number of all the points used to train the model.
        """
        return len(self._y)

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
    def X_last_appended(self):
        """
        Returns a copy of the last appended training points, regardless of
        classification.
        """
        return np.copy(self._X[-self.n_last_appended :])

    @property
    def y_last_appended(self):
        """
        Returns a copy of the last appended training target values, regardless of
        classification.
        """
        return np.copy(self._y[-self.n_last_appended :])

    @property
    def n_init(self):
        """
        Number of points passed at evaluation.
        """
        return len(np.where(self._i_iter == 0)[0])

    @property
    def X_init(self):
        """
        Coordinates of the points passed at initialisaion (returns a copy).
        """
        return np.copy(self._X[np.where(self._i_iter == 0)])

    @property
    def y_init(self):
        """
        Log-posterior of the points passed at initialisaion (returns a copy).
        """
        return np.copy(self._y[np.where(self._i_iter == 0)])

    @property
    def n_regress(self):
        """
        Number of points used to train the GPR.
        """
        return np.sum(self._i_regress)

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
    def X_last_appended_regress(self):
        """
        Returns a copy of the last appended training points in the GP Regressor.
        """
        return np.copy(self._X[self._i_regress][-self.n_last_appended_finite :])

    @property
    def y_last_appended_regress(self):
        """
        Returns a copy of the last appended training target values in the GP Regressor.
        """
        return np.copy(self._y[self._i_regress][-self.n_last_appended_finite :])

    @property
    def _i_finite_y(self):
        """
        Indices of the elements in the training set whole log-posterior has been
        classified as non-finite.

        NB: Intentionally separated from ``self._i_regress`` since there may be finite
        points not in use to train the GP Regressor.
        """
        if self.infinities_classifier is None:
            return np.arange(self.n_total, dtype=int)
        else:
            return self.infinities_classifier.i_finite_y(
                self._y_, i_sorted=self._i_y_sorted, validate=False
            )

    @property
    def X_infinite(self, validate=True):
        """
        Coordinates of the points classified as infinitely small (returns a copy).
        """
        return np.copy(np.delete(self._X, self._i_finite_y, axis=0))

    @property
    def y_infinite(self, validate=True):
        """
        Log-posterior of the points classified as infinitely small (returns a copy).
        """
        return np.copy(np.delete(self._y, self._i_finite_y))

    @property
    def y_max(self):
        """
        The maximum log-posterior value in the training set (returns a copy).

        Returns -inf (regularized if applicable) if not points have been added.
        """
        try:
            return np.copy(self._y[self._i_y_sorted[-1]])
        except IndexError:
            return self.minus_inf_value

    @property
    def noise_level(self):
        """
        The noise level (square-root of the variance) of the uncorrelated training
        data, un-transformed.
        """
        return np.copy(self._noise_level)

    @property
    def scales(self):
        """
        GPR Kernel scales as ``(output_scale, (length_scale_1, ...))`` in non-transformed
        coordinates.
        """
        output_scale, length_scales = self.gpr.scales
        return (
            self.preprocessing_y.inverse_transform_scale(output_scale),
            tuple(self.preprocessing_X.inverse_transform_scale(length_scales)),
        )

    @property
    def abs_finite_threshold(self):
        """
        Absolute threshold(s) for ``y`` values to be considered finite.
        """
        if self.infinities_classifier is None:
            return self.minus_inf_value
        return self.y_max - self.preprocessing_y.inverse_transform_scale(
            self.infinities_classifier.get_highest_current_threshold()
        )

    def is_finite_y(self, y, validate=True):
        """
        Returns the classification of y (target) values as finite (True) or not, by
        comparing them with the current threshold(s).

        Notes
        -----
        Use this method instead of the equivalent one of the 'infinities_classifier'
        attribute, since the arguments of that one may need to be transformed first.

        If calling with an argument which is not either the training set or a subset of it
        results may be inconsistent, since new values may modify the threshold.
        """
        if self.infinities_classifier is None:
            return np.isfinite(y)
        return self.infinities_classifier.is_finite_y(
            self.preprocessing_y.transform(y), validate=validate
        )

    def is_finite_X(self, X, validate=True):
        """
        Returns a prediction for the classification of the target value at some given
        parameters.

        Notes
        -----
        Use this method instead of the equivalent one of the 'infinities_classifier'
        attribute, since the arguments of that one may need to be transformed first.
        """
        if self.infinities_classifier is None:
            return np.full(shape=(len(X),), fill_value=True)
        return self.infinities_classifier.predict(
            self.preprocessing_X.transform(X), validate=validate
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
        data["is_regress"] = np.full_like(use_y, False, dtype=bool)
        data["is_regress"][self._i_regress] = True
        if self.infinities_classifier is None:
            data["is_finite"] = np.isfinite(data["y"])
        else:
            data["is_finite"] = self.is_finite_y(self.y)
        return pd.DataFrame(data)

    def set_random_state(self, random_state):
        """
        (Re)sets the random state.
        """
        self.random_state = random_state
        if self.infinities_classifier is not None:
            self.infinities_classifer.set_random_state(random_state)

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
            if np.iterable(self._noise_level):
                noise_level = np.full(fill_value=noise_level, shape=(n_train,))
        elif noise_level is None:
            if np.iterable(self._noise_level):
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
            if not np.iterable(self._noise_level):
                if self.verbose > 1:
                    warnings.warn(
                        "A new noise level has been assigned to the updated training set "
                        "while the old training set has a single scalar noise level: "
                        f"{self._noise_level}. Converting to individual levels!"
                    )
                self._noise_level = np.full(
                    fill_value=self._noise_level, shape=(len(self._y),)
                )
            self._noise_level = np.append(self._noise_level, noise_level, axis=0)
        elif isinstance(noise_level, Number):
            # NB at validation new=scalar has been converted to array if old=array
            assert not np.iterable(self._noise_level)
            if not np.isclose(noise_level, self._noise_level):
                if self.verbose > 1:
                    warnings.warn(
                        "Overwriting the noise level with a scalar. Make sure that "
                        "kernel's hyperparamters are refitted."
                    )
                self._noise_level = noise_level
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
        self._i_y_sorted = None
        self._update_noise_level(noise_level_valid)
        # 1. Fit preprocessors with finite points and select finite points in the process,
        #    and create transformed training set and noises.
        # NB: which points are finite does not change after classifier is refit (as long
        #     as the y-preprocessor is a linear transf.), so we can select them now.
        if self.infinities_classifier is None:
            self._i_regress = np.full(fill_value=True, shape=(len(self._y),))
        else:
            # The point of this: select points early, even before fitting preprocessors!
            # precisely because we need this selection to fit the preprocessors!
            # Use the manual method for non-preprocessed input.
            # Make sure that the threshold is such that there is a min of finite ones.
            if self.preprocessing_y.fitted:
                use_y = self.preprocessing_y.transform(self._y)
            else:  # first call, preprocessors not fit yet.
                use_y = self._y
            self._i_y_sorted = np.argsort(use_y)
            self._i_regress = self.infinities_classifier._i_finite_y_prefit(
                use_y, i_sorted=self._i_y_sorted, validate=False
            )
        nstd_calculator = None
        if fit_preprocessors:
            self.preprocessing_X.fit(self._X[self._i_regress], self._y[self._i_regress])
            self.preprocessing_y.fit(self._X[self._i_regress], self._y[self._i_regress])
            # We need to update the thresholds even if not fitting the inf class
            if self.infinities_classifier is not None:
                nstd_calculator = lambda nsigma: self.preprocessing_y.transform_scale(
                    delta_logp_of_1d_nstd(nsigma, self.d)
                )
                self.infinities_classifier.update_threshold_definition(nstd_calculator)
        self._X_ = self.preprocessing_X.transform(self._X)
        self._y_ = self.preprocessing_y.transform(self._y)
        # The transformed noise level is always an array.
        noise_level_array = (
            np.full(fill_value=self._noise_level, shape=(len(self._y),))
            if isinstance(self._noise_level, Number)
            else self._noise_level
        )
        self._noise_level_ = self.preprocessing_y.transform_scale(noise_level_array)
        # 2. Fit the classifiers: SVM in the transformed space, and trust region
        if self.infinities_classifier is None:
            i_finite_predict = np.arange(len(self._y))
        else:
            if fit_classifier:
                i_finite_predict = self.infinities_classifier.fit(
                    self._X_,
                    self._y_,
                    nstd_calculator=nstd_calculator,
                    i_sorted=self._i_y_sorted,
                    validate=False,
                )
            else:  # there is a classifier, but are not fitting it
                i_finite_predict = self.infinities_classifier.i_finite_y(
                    self._y_, i_sorted=self._i_y_sorted, validate=False
                )
            assert set(self._i_regress) == set(i_finite_predict), (
                "Infinities classifier miss-classified at least 1 point."
            )
        # Get number of newly added finite points -- not trivial
        self.n_last_appended_finite = len(self._i_regress) - np.searchsorted(
            i_finite_predict, self.n_total - self.n_last_appended
        )
        # 3. Re-fit the GPR in the transformed space, and maybe hyperparameters
        # If all added values are infinite there's no need to refit the GPR,
        # unless an explicit call for that with X, y = None was made
        if self.n_last_appended_finite != 0 or force_fit_gpr:
            self.gpr.fit(
                X=self._X_[self._i_regress],
                y=self._y_[self._i_regress],
                noise_level=self._noise_level_[self._i_regress],
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
        ignore_classifier=None,
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

        ignore_classifier : list, optional (default: None)
            If defined as a list, the classifiers with names on that list are not used
            to discard the given points whenever all inputs are predicted to be negative
            infinity.

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
        if validate:
            if self.gpr.kernel is None or self.gpr.kernel.requires_vector_input:
                X = check_array(X, ensure_2d=True, dtype="numeric")
            else:
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
        # TODO: At the moment is is not checked if the point is within prior bounds.
        #       Could be checked even before calling classifier, if validate=True.
        #       Idem for predict_std
        # First check if either SVM or the trust region say that the value should be -inf
        finite = self.infinities_classifier.is_finite_X(
            X_, ignore=ignore_classifier, validate=validate
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
        # The 'trivial' check avoids wasting computation in finding the y extremes
        if not self.clipper.trivial:
            return_dict["mean"] = self.clipper(
                return_dict["mean"],
                self._y[self._i_regress].min(),
                self._y[self._i_regress].max(),
            )
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
        return self.predict(X, validate=False, ignore_classifier=ignore_trust_region)

    def predict_std(self, X, ignore_classifier=None, validate=True):
        """
        Predict output standart deviation for X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated.

        ignore_classifier : list, optional (default: None)
            If defined as a list, the classifiers with names on that list are not used
            to discard the given points whenever all inputs are predicted to be negative
            infinity.

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
        finite = self.infinities_classifier.is_finite_X(
            X_, ignore=ignore_classifier, validate=validate
        )
        # If all values are infinite no need to run the prediction through the GP
        if np.all(~finite):
            return std
        # else: at least some finite points --> GPR predict for them only
        std_ = self.gpr.predict_std(X_[finite], validate=validate)
        std[finite] = self.preprocessing_y.inverse_transform(std_)
        return std


class Clipper:
    """
    Handles the upper clipping of the y-output of the regressor, as a factor of the
    difference in its range.
    """

    def __init__(self, clip_factor):
        if clip_factor is not None and clip_factor < 1:
            raise ValueError("'clip_factor' must be >= 1, or None for no clippling.")
        self.clip_factor = clip_factor

    @property
    def trivial(self):
        return self.clip_factor is None

    def __call__(self, y, y_min, y_max=None):
        if self.trivial:
            return y
        upper = self.clip_factor * y_max - (self.clip_factor - 1) * y_min
        return np.clip(y, None, upper)
