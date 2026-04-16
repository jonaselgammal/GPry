"""
Class holding the surrogate model.

The surrogate model consists of different parts, some of them optional, acting in
descending order:

- Preprocessors: defined using kwargs ``preprocessing_X|y``, they define the
  transformation between the input space and the space in which the GP Regressor and the
  infinities classifiers are trained on. See the :doc:`documentation of the preprocessing
  module<module_preprocessing>`.

- Infinities classifier(s): a class handling one or more classifiers that divide the
  space into regions where the log-posterior is expected to be finite or not. Its purpose
  is both to focus the GP Regressor into areas where the posterior is more regular, and to
  save time by avoiding evaluating the GP Regressor in uninsteresting regions. See
  :doc:`its documentation<module_infinities_classifier>`.

- GP Regressor: It is the heart of the surrogate model and thus the only non-optionally
  trivial part. It is described in more detail in :doc:`the documentation of its
  module<module_gpr>`.

- Clipper: an optional post-processor that clips the output of the GP Regressor whenever
  it is above some factor of the maximum log-posterior difference in the training set.

The :class:`surrogate.SurrogateModel` class provides the :meth:`SurrogateModel.append`
method to add new data to the model and optionally fit its different components (including
their hyperparameters if present).

Surrogate model predictions can be obtained using the :meth:`SurrogateModel.logp` method,
or more generally the :meth:`SurrogateModel.predict` method. Classification of finite
vs infinite log-posterior values can be obtained with :meth:`SurrogateModel.is_finite_X`
(for predicting with the classifier(s) based on some input) or
:meth:`SurrogateModel.is_finite_y` (comparison against a set threshold). Do not call
methods of the GP Regressor or the classifiers directly, since they need to be passed
preprocessed input.

To extract the current training set there are a number of class properties defined,
depending on whether one wants all the training points, just the last ones added, just the
ones used (or not used) by the GP Regressor, etc. Alternatively, one can get a list of
all the points and their properties with :meth:`SurrogateModel.training_set_as_df`.
"""

# Builtin
import warnings
from copy import deepcopy
from typing import Mapping, Sequence
from numbers import Number

# External
import numpy as np
import pandas as pd  # type: ignore
from sklearn.utils.validation import check_array as _sk_check_array  # type: ignore

# Local
from gpry.gpr import GaussianProcessRegressor
from gpry.preprocessing import DummyPreprocessor
from gpry.tools import delta_logp_of_1d_nstd, generic_params_names
from gpry.infinities_classifier import InfinitiesClassifiers


def check_array(*args, **kwargs):
    if "ensure_all_finite" in kwargs:
        try:
            return _sk_check_array(*args, **kwargs)
        except TypeError:
            kwargs["force_all_finite"] = kwargs.pop("ensure_all_finite")
    return _sk_check_array(*args, **kwargs)


class SurrogateModel:
    r"""
    Object holding the Gaussian Process Regressor, and, if applicable, the
    input/output preprocessing layer and the infinities classifier.

    Attributes ending with an underscore possess values in the respective preprocessed
    spaces.

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
    """

    def __init__(
        self,
        bounds=None,
        preprocessing_X=None,
        preprocessing_y=None,
        regressor=None,
        infinities_classifier=None,
        clip_factor=1.1,
        adaptive_noise=None,
        noise_floor=None,
        max_training_points=None,
        fallback=None,
        turbo=None,
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
        self.verbose = verbose if verbose is not None else 3
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
        # Custom properties for the input points
        self._properties = {}
        # Masks for being used at any time for which part of the model
        self._i_regress = np.empty((0,), dtype=bool)
        # Trackers for last-appended points
        self.n_last_appended = 0
        self.n_last_appended_finite = 0
        self._i_last_appended_regress = np.empty((0,), dtype=int)
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
        # sigma_y of the training samples.
        self._noise_level = float(regressor["noise_level"])
        # For now, the y preprocessor is not fitted
        self._noise_level_ = preprocessing_y.transform_scale(self._noise_level)
        kwargs_regressor = deepcopy(regressor)
        kwargs_regressor["noise_level"] = self._noise_level_
        kwargs_regressor["random_state"] = random_state
        backend = kwargs_regressor.pop("backend", "standard")
        # Non-standard backends handle optimization internally; strip
        # kwargs that only the standard GPR understands.
        if backend != "standard":
            for _std_only_key in ("optimizer", "copy_X_train", "use_jax"):
                kwargs_regressor.pop(_std_only_key, None)
        if backend == "standard":
            self.gpr = GaussianProcessRegressor(**kwargs_regressor)
        elif backend == "random_features":
            from gpry.experimental_random_feature_gp import RandomFeatureGP
            self.gpr = RandomFeatureGP(**kwargs_regressor)
        elif backend == "gibbs":
            from gpry.experimental_gibbs_gpr import GibbsGPR
            self.gpr = GibbsGPR(**kwargs_regressor)
        elif backend == "learned_features":
            from gpry.experimental_learned_feature_gp import LearnedFeatureGP
            self.gpr = LearnedFeatureGP(**kwargs_regressor)
        else:
            raise ValueError(
                f"Unknown GP backend '{backend}'. Supported: "
                "'standard', 'random_features', 'gibbs', 'learned_features'"
            )
        # Regressor post-processing: clip too high values
        self.clipper = Clipper(clip_factor)
        # Adaptive noise: scale noise per-sample based on distance from mode
        self._adaptive_noise = self._parse_adaptive_noise(adaptive_noise)
        self._base_noise_level = float(self._noise_level)  # original scalar
        # Noise floor: estimate minimum noise from k-NN local variance
        self._noise_floor = self._parse_noise_floor(noise_floor)
        # Max training points: limit GP training set size to avoid O(n³) bottleneck
        self._max_training_points = max_training_points
        # Ensemble fallback: train a simpler model alongside the GP for robustness
        self._fallback = self._parse_fallback(fallback)
        self._using_fallback = False  # True when GP fit failed and fallback is active
        # TuRBO: trust region Bayesian optimization
        self._turbo = self._parse_turbo(turbo)
        # Store regressor config for backend switching
        self._regressor_config = deepcopy(regressor)
        self._backend = backend

    def switch_backend(self, new_backend, **extra_kwargs):
        """Switch the GP backend (e.g. from 'standard' to 'gibbs').

        Creates a new GPR instance and retrains it on the current training data.
        The surrogate model state (training points, preprocessors, classifiers) is
        preserved.

        Parameters
        ----------
        new_backend : str
            One of 'standard', 'random_features', 'gibbs', or
            'learned_features'. Non-standard backends are experimental and
            implemented in ``gpry.experimental_*`` modules.
        **extra_kwargs
            Additional keyword arguments passed to the new GPR constructor,
            overriding the stored regressor config.
        """
        if new_backend == self._backend:
            return  # already on this backend
        # Build constructor kwargs from stored config
        kwargs = deepcopy(self._regressor_config)
        kwargs["noise_level"] = self._noise_level_
        kwargs["random_state"] = None  # will be set from surrogate
        kwargs.pop("backend", None)
        kwargs.update(extra_kwargs)
        if new_backend != "standard":
            for _std_only_key in ("optimizer", "copy_X_train", "use_jax"):
                kwargs.pop(_std_only_key, None)
        if new_backend == "standard":
            self.gpr = GaussianProcessRegressor(**kwargs)
        elif new_backend == "random_features":
            from gpry.experimental_random_feature_gp import RandomFeatureGP
            self.gpr = RandomFeatureGP(**kwargs)
        elif new_backend == "gibbs":
            from gpry.experimental_gibbs_gpr import GibbsGPR
            self.gpr = GibbsGPR(**kwargs)
        elif new_backend == "learned_features":
            from gpry.experimental_learned_feature_gp import LearnedFeatureGP
            self.gpr = LearnedFeatureGP(**kwargs)
        else:
            raise ValueError(
                f"Unknown GP backend '{new_backend}'. Supported: "
                "'standard', 'random_features', 'gibbs', 'learned_features'"
            )
        old_backend = self._backend
        self._backend = new_backend
        # Retrain on existing data
        if self.n_regress > 0:
            i_gpr = np.where(self._i_regress)[0]
            if self._adaptive_noise is not None and not self.gpr.is_noise_in_kernel:
                noise_level_adaptive = self._compute_adaptive_noise(self._y_)
                noise_level_passed = noise_level_adaptive[i_gpr]
            elif self.gpr.is_noise_in_kernel or not np.iterable(self._noise_level_):
                noise_level_passed = self._noise_level_
            else:
                noise_level_passed = self._noise_level_[i_gpr]
            if self._noise_floor is not None and not self.gpr.is_noise_in_kernel:
                floor = self._estimate_noise_floor(
                    self._X_[i_gpr], self._y_[i_gpr], k=self._noise_floor["k"]
                )
                if floor > 0:
                    noise_level_passed = np.maximum(noise_level_passed, floor)
            self.gpr.fit(
                X=self._X_[i_gpr],
                y=self._y_[i_gpr],
                noise_level=noise_level_passed,
                fit_hyperparameters=True,
                validate=False,
            )
        return old_backend

    @staticmethod
    def _parse_adaptive_noise(adaptive_noise):
        """Parse adaptive_noise configuration.

        Returns None if disabled, or a dict with keys:
            amplification: max factor by which noise grows away from mode (default: 10)
            scale: logp units at which noise reaches ~63% of max amplification (default: 5)
        """
        if adaptive_noise is None or adaptive_noise is False:
            return None
        if adaptive_noise is True:
            adaptive_noise = {}
        if not isinstance(adaptive_noise, dict):
            raise TypeError(
                "'adaptive_noise' must be None, True/False, or a dict. "
                f"Got {type(adaptive_noise)}"
            )
        return {
            "amplification": adaptive_noise.get("amplification", 10.0),
            "scale": adaptive_noise.get("scale", 5.0),
        }

    @staticmethod
    def _parse_fallback(fallback):
        """Parse fallback configuration.

        Returns None if disabled, or an experimental RandomForestFallback instance.

        Parameters
        ----------
        fallback : None, bool, dict, or RandomForestFallback instance
            - None or False: no fallback
            - True: enable with defaults
            - dict: enable with custom RandomForestFallback kwargs
            - RandomForestFallback instance: use directly
        """
        if fallback is None or fallback is False:
            return None
        from gpry.experimental_fallback import RandomForestFallback
        if isinstance(fallback, RandomForestFallback):
            return fallback
        if fallback is True:
            fallback = {}
        if not isinstance(fallback, dict):
            raise TypeError(
                "'fallback' must be None, True/False, a dict, or a "
                f"RandomForestFallback instance. Got {type(fallback)}"
            )
        return RandomForestFallback(**fallback)

    @staticmethod
    def _parse_noise_floor(noise_floor):
        """Parse noise_floor configuration.

        Returns None if disabled, or a dict with configuration.

        Parameters
        ----------
        noise_floor : None, bool, dict
            - None or False: no noise floor estimation
            - True: enable with defaults (k=5 nearest neighbors)
            - dict: with optional 'k' key (number of neighbors)
        """
        if noise_floor is None or noise_floor is False:
            return None
        if noise_floor is True:
            noise_floor = {}
        if not isinstance(noise_floor, dict):
            raise TypeError(
                f"'noise_floor' must be None, True/False, or a dict. "
                f"Got {type(noise_floor)}"
            )
        return {"k": noise_floor.get("k", 5)}

    def _parse_turbo(self, turbo):
        """Parse turbo (TuRBO trust region) configuration.

        Returns None if disabled, or an experimental TuRBOManager instance.

        Parameters
        ----------
        turbo : None, bool, dict, or TuRBOManager instance
            - None or False: no TuRBO
            - True: enable with defaults
            - dict: enable with custom TuRBOManager kwargs
              (L_init, n_regions, n_success, n_failure)
            - TuRBOManager instance: use directly
        """
        if turbo is None or turbo is False:
            return None
        from gpry.experimental_turbo import TuRBOManager
        if isinstance(turbo, TuRBOManager):
            return turbo
        if turbo is True:
            turbo = {}
        if not isinstance(turbo, dict):
            raise TypeError(
                "'turbo' must be None, True/False, a dict, or a "
                f"TuRBOManager instance. Got {type(turbo)}"
            )
        return TuRBOManager(bounds=self._bounds, **turbo)

    @staticmethod
    def _estimate_noise_floor(X, y, k=5):
        """Estimate minimum noise level from k-nearest-neighbor variance.

        For each training point, computes the variance of y among its k nearest
        neighbors. The median of these local variances provides a robust
        estimate of the noise floor.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,)
        k : int, default=5

        Returns
        -------
        noise_floor : float
            Estimated noise standard deviation.
        """
        from sklearn.neighbors import NearestNeighbors
        n = len(y)
        if n <= k:
            return 0.0
        finite = np.isfinite(y)
        X_f, y_f = X[finite], y[finite]
        if len(y_f) <= k:
            return 0.0
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(y_f)))
        nn.fit(X_f)
        _, indices = nn.kneighbors(X_f)
        # Local variance for each point (exclude self, which is index 0)
        local_vars = np.array([np.var(y_f[idx[1:]]) for idx in indices])
        # Median is robust to outliers
        return float(np.sqrt(np.median(local_vars)))

    def _compute_adaptive_noise(self, y):
        """Compute per-sample noise levels based on distance from mode.

        Points near the mode get the base noise level; points far from it get
        up to ``base_noise * (1 + amplification)``.

        Parameters
        ----------
        y : array, shape (n_samples,)
            Log-likelihood values (in transformed space).

        Returns
        -------
        noise : array, shape (n_samples,)
            Per-sample noise levels (in transformed space).
        """
        if self._adaptive_noise is None:
            return self._noise_level_
        amp = self._adaptive_noise["amplification"]
        scale = self._adaptive_noise["scale"]
        # Always use the original scalar base noise, transformed to y-space
        base = self.preprocessing_y.transform_scale(self._base_noise_level)
        y_finite = y[np.isfinite(y)]
        if len(y_finite) == 0:
            return np.full(len(y), base)
        y_max = np.max(y_finite)
        # Distance from mode (always >= 0)
        delta = np.where(np.isfinite(y), y_max - y, y_max - np.min(y_finite))
        # Noise increases exponentially with distance, saturating at amplification
        noise = base * (1.0 + amp * (1.0 - np.exp(-delta / max(scale, 1e-10))))
        return noise

    def __str__(self):
        kernel = self.gpr.kernel
        if kernel is not None and hasattr(kernel, 'hyperparameters'):
            kernel_str = (
                f"   {str(kernel)}\n"
                + "  with hyperparameters (in transformed scale):\n"
                + "    -"
                + "\n    -".join(str(h) for h in kernel.hyperparameters)
            )
        else:
            kernel_str = f"   {str(kernel)}"
        return (
            f"* X-preprocessor: {self.preprocessing_X.__class__.__name__}\n"
            + f"* y-preprocessor: {self.preprocessing_y.__class__.__name__}\n"
            + "* GPR kernel:\n"
            + kernel_str
            + "\n"
            + (
                f"* Noise level: {self._noise_level}\n"
                if not self.gpr.is_noise_in_kernel
                else ""
            )
            + f"* Classifiers for infinities: {self.infinities_classifier}"
        )

    @property
    def d(self):
        """Dimensionality of the training data."""
        return len(self._bounds)

    @property
    def bounds(self):
        """Copy of the problem's prior bounds."""
        return np.copy(self._bounds)

    @property
    def trust_bounds(self):
        """
        The bounds of a smaller trust region where the log-posterior is expected to be
        finite, possibly defined by a classifier (in particular if
        :class:`infinities_classifier.TrustRegion` is being used) and/or by TuRBO.

        When both a classifier trust region and TuRBO are active, returns the
        intersection (tightest bounds).

        Otherwise returns a copy of the original prior bounds.
        """
        if self.infinities_classifier is not None:
            bounds = self.preprocessing_X.inverse_transform_bounds(
                self.infinities_classifier.trust_bounds
            )
        else:
            bounds = np.copy(self._bounds)
        if self._turbo is not None:
            turbo_bounds = self._turbo.get_trust_bounds()
            # Intersect: take the tighter of the two
            bounds[:, 0] = np.maximum(bounds[:, 0], turbo_bounds[:, 0])
            bounds[:, 1] = np.minimum(bounds[:, 1], turbo_bounds[:, 1])
        return bounds

    @property
    def fitted(self):
        """
        True whenever the the surrogate model has already been fitted to some points.
        """
        return self._fitted

    @property
    def n_total(self):
        """
        Number of points/features in the training set of the model, including points with
        target values classified as infinite.
        """
        return len(self._y)

    @property
    def X(self):
        """
        Coordinates of all the points used to train the model (returns a copy).
        Intended to be used when one wants to access the training data for any purpose.
        """
        return np.copy(self._X)

    @property
    def y(self):
        """
        Log-posterior of all the points used to train the model (returns a copy).
        Intended to be used when one wants to access the training data for any purpose.
        """
        return np.copy(self._y)

    @property
    def X_last_appended(self):
        """
        Coordinates of the training points added to the surrogate model in the last call
        to :meth:`Surrogate.append`, regardless of classification (returns a copy).
        """
        return np.copy(self._X[-self.n_last_appended :])

    @property
    def y_last_appended(self):
        """
        Log-posterior of the training points added to the surrogate model in the last call
        to :meth:`Surrogate.append`, regardless of classification (returns a copy).
        """
        return np.copy(self._y[-self.n_last_appended :])

    @property
    def n_init(self):
        """
        Number of points passed at initialisation.
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
        Number of points currently in use to train the GP Regressor.
        """
        return len(self._i_regress)

    @property
    def X_regress(self):
        """
        Coordinates of the points currently in use to train the GP Regressor (returns a
        copy).
        """
        return np.copy(self._X[self._i_regress])

    @property
    def y_regress(self):
        """
        Log-posterior of the points currently in use to train the GP Regressor (returns a
        copy).
        """
        return np.copy(self._y[self._i_regress])

    @property
    def X_last_appended_regress(self):
        """
        Coordinates of the training points added to the GP Regressor in the last call
        to :meth:`Surrogate.append` (returns a copy).
        """
        return np.copy(self._X[self._i_last_appended_regress])

    @property
    def y_last_appended_regress(self):
        """
        Log-posterior of the training points added to the GP Regressor in the last call
        to :meth:`Surrogate.append` (returns a copy).
        """
        return np.copy(self._y[self._i_last_appended_regress])

    def _indices_added_to_gpr_on_last_append(self, i_gpr):
        """
        Return newly appended indices that are included in the actual GP fit.

        Classifier output is not guaranteed to be sorted by training-set index. For
        example, threshold-based classifiers often return finite points in y-sorted
        order. Therefore this must be a set-membership check, not a ``searchsorted``
        computation. Using ``searchsorted`` here can incorrectly report zero newly
        finite points and skip the GP refit, leaving the surrogate stale.
        """
        if self.n_last_appended == 0:
            return np.empty((0,), dtype=int)
        first_new = self.n_total - self.n_last_appended
        new_indices = np.arange(first_new, self.n_total, dtype=int)
        if len(new_indices) == 0:
            return new_indices
        in_gpr = np.isin(new_indices, np.asarray(i_gpr, dtype=int), assume_unique=False)
        return new_indices[in_gpr]

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
        Coordinates of the training points classified as infinitely small (returns a
        copy).
        """
        return np.copy(np.delete(self._X, self._i_finite_y, axis=0))

    @property
    def y_infinite(self, validate=True):
        """
        Log-posterior of the training points classified as infinitely small (returns a
        copy).
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
    def scales(self):
        """
        GPR Kernel scales as ``(output_scale, (length_scale_1, ...))`` in non-transformed
        coordinates.
        """
        output_scale, length_scales = self.gpr.scales
        if self.preprocessing_y.fitted:
            return (
                self.preprocessing_y.inverse_transform_scale(output_scale),
                tuple(self.preprocessing_X.inverse_transform_scale(length_scales)),
            )
        else:
            return (output_scale, length_scales)

    @property
    def noise_level(self):
        """
        The noise level (square-root of the variance) of the uncorrelated training
        data, in non-transformed coordinates.
        """
        if self.preprocessing_y.fitted:
            return self.preprocessing_y.inverse_transform_scale(self.gpr.noise_level)
        else:
            return self.gpr.noise_level

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
        for k, v in self._properties.items():
            data[f"property:{k}"] = v
        return pd.DataFrame(data)

    def set_random_state(self, random_state):
        """
        (Re)sets the random state.
        """
        self.random_state = random_state
        self.gpr.random_state = random_state
        if self.infinities_classifier is not None:
            self.infinities_classifier.set_random_state(random_state)

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

    def _update_properties(self, properties):
        if properties is None:
            return
        err_msg = (
            "`properties` must be a dict of {property: [values1, value2, ...]}, with as "
            "many values as points passed to `append`."
        )
        if not isinstance(properties, Mapping):
            raise TypeError(f"{err_msg} Got {properties}.")
        for k, v in properties.items():
            if not hasattr(v, "__len__"):
                raise TypeError(f"{err_msg} Got {v} (not a Sequence) for property {k}.")
            if len(v) != self.n_last_appended:
                raise TypeError(
                    f"{err_msg} Wrong number of values in {k}:{v}: should be "
                    f"{self.n_last_appended}."
                )
            # New property -> assign NaN to previous entries
            if k not in self._properties:
                self._properties[k] = [np.nan] * (self.n_total - self.n_last_appended)
            self._properties[k] += list(v)

    def append(
        self,
        X,
        y,
        noise_level=None,
        fit_gpr=True,
        fit_classifier=True,
        validate=True,
        i_iter=None,
        properties=None,
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

        y : array-like, shape = (n_samples), or None
            Target values to append to the data

        noise_level : number, array-like, shape = (n_samples)
            Uncorrelated standard deviation(s) to add to the diagonal part of the
            covariance matrix. If an array is passed, it needs to have the same number of
            entries as y. If None, the noise_level set in the instance is used (notice
            that passing None is not recommended if a preprocessor for the GPR training
            samples may have been refit after initialization). If a non-None value is
            passed, it is advisable to refit the hyperparameters of the kernel.

        fit_gpr : Bool or dict (default: True)
            Whether the GPR :math:`\theta`-parameters should be optimised. It can be
            passed a dict, which is always interpreted as ``True``, containing arguments
            for the ``_fit_hyperparameters`` method, namely ``n_restarts`` of the
            optimizer, ``start_from_current`` if the first optimizer run should start
            from the last optimum, or different ``hyperparameters_bounds``. E.g. to
            perform a single GPR optimization run starting at the last hyperparameter
            optimum: ``fit_gpr={'start_from_current': True, 'n_restarts': 1}``.

        fit_classifier: Bool (default: True)
            Whether the infinities classifier is refit. Overridden to ``True`` if
            ``fit_gpr`` is not ``False``.

        i_iter : int or array-like, shape = (n_samples), optional
            An index to be assigned to the newly-appended points. If None, assigns the
            last entry +1.

        properies : dict {str: Sequence}, optional
            A dictionary of property names, with a list of the same length as ``X`` as
            values. These properties have no effect, beyond being kept track of.

        Returns
        -------
        self
            Returns an instance of self.
        """
        if validate and (
            self.gpr.kernel is None or self.gpr.kernel.requires_vector_input
        ):
            X = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=False)
            y = check_array(
                y, ensure_2d=False, dtype="numeric", ensure_all_finite=False
            )
        elif validate:
            X = check_array(X, ensure_2d=False, dtype=None, ensure_all_finite=False)
            y = check_array(y, ensure_2d=False, dtype=None, ensure_all_finite=False)
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                f"Different numbers of points in X (shape {X.shape}) and y (shape "
                f"{y.shape})."
            )
        # Ensure fit_gpr --> fit_classifier --> fit_preprocessors
        fit_preprocessors = False
        if not isinstance(fit_gpr, (bool, Mapping)):
            raise TypeError(
                f"'fit_gpr' kwarg must be bool|dict|'simple', but was {fit_gpr}"
            )
        if fit_gpr is not False:
            fit_classifier = True
        if fit_classifier:
            fit_preprocessors = True
        force_fit_gpr = False  # to avoid skipping fit if no points added if X,y = None
        if X is None and y is None:
            X, y = np.empty((0, self.d)), np.empty((0,))
            force_fit_gpr = fit_gpr is not False
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
        self._update_properties(properties)
        # 1. Fit preprocessors with finite points and select finite points in the process,
        #    and create transformed training set and noises.
        # NB: which points are finite does not change after classifier is refit (as long
        #     as the y-preprocessor is a linear transf.), so we can select them now.
        if self.infinities_classifier is None:
            self._i_regress = self._i_y_sorted.copy()
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
        # Recompute sorted indices after transformation (needed for non-linear y-preprocessors)
        if not getattr(self.preprocessing_y, 'is_linear', True):
            self._i_y_sorted = np.argsort(self._y_)
            # Recompute _i_regress with transformed values for consistency
            if self.infinities_classifier is not None:
                self._i_regress = self.infinities_classifier._i_finite_y_prefit(
                    self._y_, i_sorted=self._i_y_sorted, validate=False
                )
        # The transformed noise level is always an array if noise is not in the kernel
        noise_level_upd = self._noise_level
        self._noise_level_ = self.preprocessing_y.transform_scale(self._noise_level)
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
            # For non-linear y-preprocessors, the pre-fit classification may differ
            # from the post-fit one because the transform changed. Use the post-fit
            # classification in that case.
            if not getattr(self.preprocessing_y, 'is_linear', True):
                self._i_regress = i_finite_predict
            else:
                assert set(self._i_regress) == set(i_finite_predict), (
                    "Infinities classifier miss-classified at least 1 point."
                )
        # 3. Re-fit the GPR in the transformed space, and maybe hyperparameters
        # If all added values are infinite there's no need to refit the GPR,
        # unless an explicit call for that with X, y = None was made
        # Determine which finite points to use for GP training
        i_gpr = self._i_regress
        # Apply max_training_points limit: keep highest-y points for GP accuracy
        if (self._max_training_points is not None
                and len(i_gpr) > self._max_training_points):
            # Select the top-m points by transformed y value (highest = near mode)
            y_at_regress = self._y_[i_gpr]
            top_m = np.argsort(y_at_regress)[-self._max_training_points:]
            i_gpr = i_gpr[np.sort(top_m)]  # maintain original ordering
        self._i_last_appended_regress = self._indices_added_to_gpr_on_last_append(i_gpr)
        self.n_last_appended_finite = len(self._i_last_appended_regress)
        # Apply adaptive noise if configured (per-sample noise based on y distance)
        if self._adaptive_noise is not None and not self.gpr.is_noise_in_kernel:
            noise_level_adaptive = self._compute_adaptive_noise(self._y_)
            noise_level_passed = noise_level_adaptive[i_gpr]
        elif self.gpr.is_noise_in_kernel or not np.iterable(self._noise_level_):
            noise_level_passed = self._noise_level_
        else:
            noise_level_passed = self._noise_level_[i_gpr]
        # Apply noise floor: estimate minimum noise from k-NN local variance
        # Computed in transformed space (where the GP trains)
        if self._noise_floor is not None and not self.gpr.is_noise_in_kernel:
            floor = self._estimate_noise_floor(
                self._X_[i_gpr], self._y_[i_gpr], k=self._noise_floor["k"]
            )
            if floor > 0:
                noise_level_passed = np.maximum(noise_level_passed, floor)
        if self.n_last_appended_finite != 0 or force_fit_gpr:
            gpr_failed = False
            try:
                self.gpr.fit(
                    X=self._X_[i_gpr],
                    y=self._y_[i_gpr],
                    noise_level=noise_level_passed,
                    fit_hyperparameters=fit_gpr,
                    validate=False,
                )
            except Exception as e:
                if self._fallback is not None:
                    gpr_failed = True
                    warnings.warn(
                        f"GP fit failed ({type(e).__name__}: {e}). "
                        "Falling back to RandomForest."
                    )
                else:
                    raise
            # Train fallback alongside GP (or as replacement if GP failed)
            if self._fallback is not None:
                self._fallback.fit(self._X_[i_gpr], self._y_[i_gpr])
                self._using_fallback = gpr_failed
        self._fitted = True
        # Update TuRBO trust regions if active
        if self._turbo is not None and self.n_last_appended > 0:
            new_X = self._X[-self.n_last_appended:]
            new_y = self._y[-self.n_last_appended:]
            self._turbo.update(new_X, new_y, all_X=self._X, all_y=self._y)
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
        y_mean : array, shape = (n_samples)
            Mean of GPR's predictive distribution a query points

        y_std : array, shape = (n_samples,), optional
            Standard deviation of GPR's predictive distribution at query points.
            Only returned when return_std is True.

        y_mean_grad : shape = (n_samples, n_features), optional
            The gradient of GPR's the predicted mean.

        y_std_grad : shape = (n_samples, n_features), optional
            The gradient of GPR's the predicted std.
        """
        self.n_eval += len(X)
        if return_std_grad and not (return_std and return_mean_grad):
            raise ValueError(
                "Not returning std_gradient without returning "
                "the std and the mean grad."
            )
        if X.shape[0] != 1 and (return_mean_grad or return_std_grad):
            raise ValueError("Mean grad and std grad not implemented for n_samples > 1")
        # JAX fast path: skip classifier, dict wrapping, and np.copy when
        # the GP has a ready JAX accelerator and no gradients are requested.
        # The classifier is skipped because the acquisition optimizer already
        # respects bounds, and classifier overhead is significant.
        jax_accel = getattr(self.gpr, '_jax_accel', None)
        if (not validate
                and not return_mean_grad and not return_std_grad
                and not (self._using_fallback and self._fallback is not None)
                and self.clipper.trivial
                and jax_accel is not None and jax_accel.ready):
            X_ = self.preprocessing_X.transform(X)
            if return_std:
                y_mean_j, y_std_j = jax_accel.predict_mean_std_jax(X_)
                y_mean = self.preprocessing_y.inverse_transform(
                    np.asarray(y_mean_j))
                y_std = self.preprocessing_y.inverse_transform_scale(
                    np.asarray(y_std_j))
                return [y_mean, y_std]
            else:
                y_mean_j = jax_accel.predict_mean_jax(X_)
                return self.preprocessing_y.inverse_transform(
                    np.asarray(y_mean_j))
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
        # else: at least some finite points --> regressor predict for them only
        # Use fallback if GP fit failed and fallback is available
        if self._using_fallback and self._fallback is not None:
            return_gpr_ = self._fallback.predict(
                X_[finite],
                return_std=return_std,
                return_mean_grad=return_mean_grad,
                return_std_grad=return_std_grad,
                validate=validate,
            )
        else:
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
            if k == "mean":
                return_dict[k][finite] = self.preprocessing_y.inverse_transform(v)
            else:
                return_dict[k][finite] = self.preprocessing_y.inverse_transform_scale(v)
        # Apply inverse transformation twice for std grad
        if "std_grad" in return_dict:
            return_dict["std_grad"][finite] = (
                self.preprocessing_y.inverse_transform_scale(
                    return_dict["std_grad"][finite]
                )
            )
        # Chain rule correction for nonlinear X-preprocessors (e.g. InputWarping).
        # GPR gradients are w.r.t. warped x; we need gradients w.r.t. original x.
        # d(f)/d(x_orig) = d(f)/d(x_warped) * d(x_warped)/d(x_orig)
        if (return_mean_grad or return_std_grad) and hasattr(
            self.preprocessing_X, 'has_nonlinear'
        ) and self.preprocessing_X.has_nonlinear:
            # Jacobian at the pre-warped (NormalizeBounds-transformed) query points
            jac = self.preprocessing_X.jacobian_diagonal(X[finite])
            if "mean_grad" in return_dict:
                return_dict["mean_grad"][finite] *= jac
            if "std_grad" in return_dict:
                return_dict["std_grad"][finite] *= jac
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

    def predict_transformed(
        self,
        X_,
        return_std=False,
        return_mean_grad=False,
        return_std_grad=False,
        validate=True,
        ignore_classifier=None,
    ):
        """
        Predict output for inputs that are already in the surrogate's transformed X-space.
        """
        X_ = np.atleast_2d(X_)
        self.n_eval += len(X_)
        if return_std_grad and not (return_std and return_mean_grad):
            raise ValueError(
                "Not returning std_gradient without returning "
                "the std and the mean grad."
            )
        if X_.shape[0] != 1 and (return_mean_grad or return_std_grad):
            raise ValueError("Mean grad and std grad not implemented for n_samples > 1")
        if validate:
            if self.gpr.kernel is None or self.gpr.kernel.requires_vector_input:
                X_ = check_array(X_, ensure_2d=True, dtype="numeric")
            else:
                X_ = check_array(X_, ensure_2d=False, dtype=None)
        else:
            X_ = np.asarray(X_)
        return_dict = {"mean": np.full(X_.shape[0], self.minus_inf_value)}
        if return_std:
            return_dict["std"] = np.zeros(X_.shape[0])
        if return_mean_grad:
            return_dict["mean_grad"] = np.full((X_.shape[0], self.d), self.inf_value)
        if return_std_grad:
            return_dict["std_grad"] = np.zeros((X_.shape[0], self.d))
        finite = self.infinities_classifier.is_finite_X(
            X_, ignore=ignore_classifier, validate=validate
        )
        if np.all(~finite):
            if len(return_dict) == 1:
                return return_dict["mean"]
            return list(return_dict.values())
        if self._using_fallback and self._fallback is not None:
            return_gpr_ = self._fallback.predict(
                X_[finite],
                return_std=return_std,
                return_mean_grad=return_mean_grad,
                return_std_grad=return_std_grad,
                validate=validate,
            )
        else:
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
            if k == "mean":
                return_dict[k][finite] = self.preprocessing_y.inverse_transform(v)
            else:
                return_dict[k][finite] = self.preprocessing_y.inverse_transform_scale(v)
        if "std_grad" in return_dict:
            return_dict["std_grad"][finite] = (
                self.preprocessing_y.inverse_transform_scale(
                    return_dict["std_grad"][finite]
                )
            )
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
        ignore_classifier=None,
    ):
        """
        Returns the surrogate log-posterior. Alias of :meth:`Surrogate.predict` for the
        GP Regressor's mean only.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated.

        validate : bool, default: True
            If False, ``X`` is assumed to be correctly formatted (2-d float array, with
            points as rows and dimensions/features as columns, C-contiguous), and no
            checks are performed on it. Reduces overhead. Use only for repeated calls when
            the input is programmatically generated to be correct at each stage.

        ignore_classifier : list, optional (default: None)
            If defined as a list, the classifiers with names on that list are not used
            to discard the given points whenever all inputs are predicted to be negative
            infinity.

        Returns
        -------
        log-posterior : array, shape = (n_samples)
            Predicted log-posterior.
        """
        if validate:
            X = np.atleast_2d(X)
            if self.gpr.kernel is None or self.gpr.kernel.requires_vector_input:
                X = check_array(X, ensure_2d=True, dtype="numeric")
            else:
                X = check_array(X, ensure_2d=False, dtype=None)
        return self.predict(X, validate=False, ignore_classifier=ignore_classifier)

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
        if validate:
            X = np.atleast_2d(X)
            if self.gpr.kernel is None or self.gpr.kernel.requires_vector_input:
                X = check_array(X, ensure_2d=True, dtype="numeric")
            else:
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
        # else: at least some finite points --> regressor predict for them only
        if self._using_fallback and self._fallback is not None:
            std_ = self._fallback.predict_std(X_[finite], validate=validate)
        else:
            std_ = self.gpr.predict_std(X_[finite], validate=validate)
        std[finite] = self.preprocessing_y.inverse_transform_scale(std_)
        return std

    def predict_std_transformed(self, X_, ignore_classifier=None, validate=True):
        """
        Predict output standard deviation for inputs already in transformed X-space.
        """
        X_ = np.atleast_2d(X_)
        self.n_eval += len(X_)
        if validate:
            if self.gpr.kernel is None or self.gpr.kernel.requires_vector_input:
                X_ = check_array(X_, ensure_2d=True, dtype="numeric")
            else:
                X_ = check_array(X_, ensure_2d=False, dtype=None)
        else:
            X_ = np.asarray(X_)
        std = np.zeros(X_.shape[0])
        finite = self.infinities_classifier.is_finite_X(
            X_, ignore=ignore_classifier, validate=validate
        )
        if np.all(~finite):
            return std
        if self._using_fallback and self._fallback is not None:
            std_ = self._fallback.predict_std(X_[finite], validate=validate)
        else:
            std_ = self.gpr.predict_std(X_[finite], validate=validate)
        std[finite] = self.preprocessing_y.inverse_transform_scale(std_)
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
