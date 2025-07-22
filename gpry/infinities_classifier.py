r"""
This module handles classifiers for training points that categorize them as either
"finite" (interesting) or "infinite" (non-interesting). This classification defines
regions which are "safe" to explore in contrast to regions which are "unsafe"
to explore since they are infinite. This is done in an attempt to hinder the
exploration of parts of the parameter space which have a :math:`-\infty` log-posterior
value. These values need to be filtered out since feeding them to the GP
Regressor will break it. Nevertheless this is important information that we
shouldn't throw away.

We also make use of these classifiers when doing the MC runs of the surrogate log-p
to tell the model which regions it shouldn't visit. In essence our process
shrinks the prior to a region where the model thinks that all values of the
log-posterior distribution are finite.

At the moment, it impements a Support vector machine (SVM) with an RBF kernel and a
hyper-rectangular constraint fitted to a fraction of the highest log-p samples.
"""

import numpy as np
from sklearn.svm import SVC  # type: ignore
from sklearn.utils.validation import check_array  # type: ignore

from gpry.tools import check_random_state, get_Xnumber, shrink_bounds, is_in_bounds

k_trust, k_svm = "trust_region", "svm"


class InfinitiesClassifiers:
    f"""
    Wrapper class for a collection of infinities classifiers.

    Parameters
    ----------
    bounds : array-like, shape = (dimensionality, 2)
        Bounds of the problem

    nstd_calculator : callable
        Function able to translate threshold values from sigma units to the space in which
        the classifiers are defined (including preprocessing, if present).

    keep_min : int, optional
        If passed, and there are fewer than ``keep_min`` points above the threshold,
        the threshold is ignored and the ``keep_min`` largest points are returned.
        Exception: fewer than ``keep_min`` points in ``y`` that are not ``-np.inf``.
        If none passed, uses by default ``max(2, dimensionality)``.

    random_state : int or numpy.random.Generator, optional
        The generator used to perform random operations of the GPR. If an integer is
        given, it is used as a seed for the default global numpy random number generator.

    classifiers : dict {{str: {{str: any}}}}
        Dictionary containg the names of classifier classes as keys and a dict of their
        initialisation options as values. At the moment, the supported classifiers are
        {k_trust}, {k_svm}.
    """

    def __init__(
        self, bounds, nstd_calculator, keep_min=None, random_state=None, **classifiers
    ):
        self.bounds = bounds
        if keep_min is None:
            dim = len(bounds)
            self.keep_min = max(2, dim)
        self.classifiers = {}
        for k, v in classifiers.items():
            # Process common parameters (min n finite, threshold) before initialising
            k = k.lower()
            v = v or {}
            v["nstd_calculator"] = nstd_calculator
            if k == k_trust:
                self.classifiers[k] = TrustRegion(bounds=bounds, **v)
            elif k == k_svm:
                self.classifiers[k] = SVM(random_state=random_state, **v)
            else:
                unknown = [unkn for unkn in classifiers if unkn not in [k_trust, k_svm]]
                raise ValueError(
                    f"'infinities_classifier': unknown classifiers {unknown}."
                )

    def __str__(self):
        return ", ".join(str(c) for c in self.classifiers.values())

    @property
    def trust_bounds(self):
        """
        The bounds of a smaller trust region possibly defined by a classifier (in
        particular if :class:`infinities_classifer.TrustRegion` is being used.
        Otherwise returns the original prior bounds.
        """
        # Assuming there is only one class with trust bounds.
        # Extensible to more than one by taking intersection of them.
        try:
            return next(
                infclass.trust_bounds
                for infclass in self.classifiers.values()
                if hasattr(infclass, "trust_bounds")
            )
        except StopIteration:
            return self.bounds

    def update_threshold_definition(self, nstd_calculator):
        """
        Updates the classifier's thresholds after a change in the transformation
        defining the space of the y's.

        Parameters
        ----------
        nstd_calculator : callable
            Function able to translate threshold values from sigma units to the space in
            which the classifiers are defined (including preprocessing, if present.
        """
        for infclass in self.classifiers.values():
            infclass.update_threshold_definition(nstd_calculator)

    def fit(
        self, X, y, nstd_calculator=None, keep_min=None, i_sorted=None, validate=True
    ):
        r"""
        Fits the classifiers.

        If the representation space of the ``y``'s has changed, it is necessary to pass an
        updated ``nstd_calculator`` to update the thresholds.

        Parameters
        ----------
        X : array-like, shape = (n_samples, dimensionality)
            Training data.

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values. They can have any value smaller than positive infinity,
            including negative infinity.

        nstd_calculator : callable
            Function able to translate threshold values from sigma units to the space in
            which the classifiers are defined (including preprocessing, if present.

        keep_min : int, optional
            If passed, and there are fewer than ``keep_min`` points above the threshold,
            the threshold is ignored and the ``keep_min`` largest points are returned.
            Exception: fewer than ``keep_min`` points in ``y`` that are not ``-np.inf``.
            If none passed, uses by default ``max(2, dimensionality)``.

        i_sorted : bool or array-like, int, 1-dimensional, optional
            Sorting indices for ``y``. If True passed, ``y`` is assumed sorted.
            Passing sorting indices (or True) greatly improves the efficiency of this
            function, saving a sort of ``y``.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        i_finite : array-like int
            Indices of elements in ``y`` classified as "finite" according to the
            threshold.
        """
        i_finite = None
        for k, infclass in self.classifiers.items():
            if i_sorted is None:
                i_sorted = np.argsort(y)
            this_i_finite = infclass.fit(
                X,
                y,
                nstd_calculator=nstd_calculator,
                keep_min=keep_min if keep_min is not None else self.keep_min,
                i_sorted=i_sorted,
                validate=validate,
            )
            if i_finite is None:
                i_finite = this_i_finite
            elif this_i_finite is not None:
                i_finite = np.intersect1d(i_finite, this_i_finite, assume_unique=True)
        return i_finite

    def get_classifier_min_threshold(self, ignore=None):
        """
        Returns the classifier with the minimum threshold, or None of none present or all
        ignored.

        Parameters
        ----------
        ignore : list(str), optional
            List of classifiers to ignore in this call.

        Returns
        -------
        name_or_strictest_classifier : str
        """
        if ignore is None:
            ignore = []
        elif isinstance(ignore, str):
            ignore = [ignore]
        # If there are no classifiers, simply pass whether they are finite or not
        if not self.classifiers or set(ignore) == set(self.classifiers):
            return None
        # Otherwise, just use the one with the smallest threshold:
        thresholds = [
            (infclass._diff_threshold if k not in ignore else np.inf)
            for k, infclass in self.classifiers.items()
        ]
        return list(self.classifiers.values())[np.argmin(thresholds)]

    def get_highest_current_threshold(self, ignore=None):
        """
        Returns the minimum differential threshold among all classifiers.

        Parameters
        ----------
        ignore : list(str), optional
            List of classifiers to ignore in this call.

        Returns
        -------
        threshold : float (positive)
        """
        if ignore is None:
            ignore = []
        elif isinstance(ignore, str):
            ignore = [ignore]
        # If there are no classifiers, return -inf
        if not self.classifiers or set(ignore) == set(self.classifiers):
            return -np.inf
        # Otherwise, return the smallest threshold:
        thresholds = [
            (infclass._diff_threshold if k not in ignore else np.inf)
            for k, infclass in self.classifiers.items()
        ]
        return min(thresholds)

    def _i_finite_y_prefit(
        self, y, keep_min=None, i_sorted=None, ignore=None, validate=True
    ):
        """
        Method to classify points with non-fitted preprocessors, based on the current
        threshold only.

        Parameters
        ----------
        keep_min : int, optional
            If passed, and there are fewer than ``keep_min`` points above the threshold,
            the threshold is ignored and the ``keep_min`` largest points are returned.
            Exception: fewer than ``keep_min`` points in ``y`` that are not ``-np.inf``.
            If none passed, uses by default ``max(2, dimensionality)``.

        i_sorted : bool or array-like, int, 1-dimensional, optional
            Sorting indices for ``y``. If True passed, ``y`` is assumed sorted.
            Passing sorting indices (or True) greatly improves the efficiency of this
            function, saving a sort of ``y``.

        ignore : list(str), optional
            List of classifiers to ignore in this call.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        i_finite : array-like int
            Indices of elements in ``y`` classified as "finite" according to the
            threshold.
        """
        use_infclass = self.get_classifier_min_threshold(ignore)
        if use_infclass is None:
            return np.argwhere(np.isfinite(y)).T[0]
        if i_sorted is None:
            i_sorted = np.argsort(y)
        return use_infclass.i_finite_threshold(
            y,
            use_infclass._diff_threshold,
            i_sorted=i_sorted,
            validate=validate,
        )[0]

    def i_finite_y(self, y, i_sorted=None, ignore=None, validate=True):
        """
        Returns the indices of the finite points, depending on the current fit of the
        classifiers.

        Parameters
        ----------
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values. They can have any value smaller than positive infinity,
            including negative infinity.

        i_sorted : bool or array-like, int, 1-dimensional, optional
            Sorting indices for ``y``. If True passed, ``y`` is assumed sorted.
            Passing sorting indices (or True) greatly improves the efficiency of this
            function, saving a sort of ``y``.

        ignore : list(str), optional
            List of classifiers to ignore in this call.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        i_finite : array-like int
            Indices of elements in ``y`` classified as "finite" according to the
            threshold.
        """
        use_infclass = self.get_classifier_min_threshold(ignore)
        if use_infclass is None:
            return np.argwhere(np.isfinite(y)).T[0]
        if i_sorted is None:
            i_sorted = np.argsort(y)
        return use_infclass.i_finite_y(y, i_sorted=i_sorted, validate=validate)

    def is_finite_y(self, y, i_sorted=None, ignore=None, validate=True):
        """
        Returns True for the finite indices of the input array, depending on the current
        threshold and maximum ``y`` value in the training set (not the input).

        Parameters
        ----------
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values. They can have any value smaller than positive infinity,
            including negative infinity.

        i_sorted : bool or array-like, int, 1-dimensional, optional
            Sorting indices for ``y``. If True passed, ``y`` is assumed sorted.
            Passing sorting indices (or True) greatly improves the efficiency of this
            function, saving a sort of ``y``.

        ignore : list(str), optional
            List of classifiers to ignore in this call.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        y_finite : array-like bool
            Classification of given points according to the threshold.
        """
        use_infclass = self.get_classifier_min_threshold(ignore)
        if use_infclass is None:
            return ~np.isinf(y)
        if i_sorted is None:
            i_sorted = np.argsort(y)
        return use_infclass.is_finite_y(y, i_sorted=i_sorted, validate=validate)

    # TODO: this function could be made more efficient by operating each classifier
    #       not in the whole y set but on the output of the previous one.
    #       But it is not used so often, so left for later.
    def is_finite_X(self, X, ignore=None, validate=True):
        """
        Returns ``True`` for locations predicted to have a finite posterior probability
        density, based on the output of the classifiers' ``predict`` methods.

        Parameters
        ----------
        X : array-like, shape = (n_samples, dimensionality)
            Training data.

        ignore : list(str), optional
            List of classifiers to ignore in this call.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        X_finite : array-like bool
            Whether each of the points are located inside the trust region.
        """
        # If there are no classifiers, assume all points are finite
        if not self.classifiers or ignore == "all":
            return np.full(len(X), True, dtype=bool)
        i_finite = None
        if ignore is None:
            ignore = []
        for k, infclass in self.classifiers.items():
            if k in ignore:
                continue
            this_i_finite = self.classifiers[k].is_finite_X(
                np.ascontiguousarray(X) if k == k_svm else X, validate=validate
            )
            if i_finite is None:
                i_finite = this_i_finite
            else:
                i_finite &= this_i_finite
        if i_finite is None:  # all classifiers were ignored.
            return self._i_finite_X(X, validate=validate, ignore="all")
        return i_finite

    def set_random_state(self, random_state):
        for classifier in self.classifiers.values():
            if hasattr(classifier, "set_random_state"):
                classifier.set_random_state(random_state)


class ThresholdClassifier:
    """
    Parent class for classifiers that separate the target values according to some
    given threshold.

    Parameters
    ----------
    threshold : numeric
        Differential threshold below the maximum element in ``y`` over which points
        are selected as "finite". Must be positive, can be ``np.inf``.

    nstd_calculator : callable
        Function able to translate threshold values from sigma units to the space in which
        the classifiers are defined (including preprocessing, if present).
    """

    def __init__(self, threshold, nstd_calculator):
        self.threshold_definition = threshold
        # Attr to hold the differential threshold in the ``y`` space
        self._diff_threshold = None
        # Attr to hold the actual threshold used,
        # possibly dep. on keep_min in the last self.fit call.
        self._current_threshold = None
        self.update_threshold_definition(nstd_calculator)

    def __str__(self):
        return f"{self.__class__.__name__} (threshold: {self.threshold_definition})"

    def update_threshold_definition(self, nstd_calculator):
        """
        Converts the units of sigma in the theshold definition and transforms it into
        the right ``y`` space.

        Call every time the representation space of the ``y``'s changes, or alternatively
        pass the ``nstd_calculator`` in the corresponding call of the ``fit`` method.

        Parameters
        ----------
        nstd_calculator : callable
            Function able to translate threshold values from sigma units to the space in
            which the classifiers are defined (including preprocessing, if present).
        """
        if self.threshold_definition == np.inf:
            self._diff_threshold = np.inf
            return
        thr, is_sigma_units, sigma_power = get_Xnumber(
            self.threshold_definition, "s", None, dtype=float, varname="threshold"
        )
        if sigma_power is not None:
            raise TypeError("Power for sigma in threshold not supported.")
        if is_sigma_units:
            thr = nstd_calculator(thr)
        self._diff_threshold = thr

    @property
    def fitted(self):
        return self._current_threshold is not None

    def i_finite_y(self, y, i_sorted=None, validate=True):
        """
        Returns the indices of the finite points, depending on the current threshold and
        maximum ``y`` value in the training set (not the input).

        Parameters
        ----------
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values. They can have any value smaller than positive infinity,
            including negative infinity.

        i_sorted : bool or array-like, int, 1-dimensional, optional
            Sorting indices for ``y``. If True passed, ``y`` is assumed sorted.
            Passing sorting indices (or True) greatly improves the efficiency of this
            function, saving a sort of ``y``.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        i_finite : array-like int
            Indices of elements in ``y`` classified as "finite" according to the
            threshold, in sorting order.
        """
        if not self.fitted:
            raise ValueError("This classifier has not been trained yet!")
        return self.i_finite_threshold(
            y,
            self._current_threshold,
            i_sorted=i_sorted,
            validate=validate,
        )[0]

    def is_finite_y(self, y, i_sorted=None, validate=True):
        """
        Returns True for the finite indices of the input array, depending on the current
        threshold and maximum ``y`` value in the training set (not the input).

        Parameters
        ----------
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values. They can have any value smaller than positive infinity,
            including negative infinity.

        i_sorted : bool or array-like, int, 1-dimensional, optional
            Sorting indices for ``y``. If True passed, ``y`` is assumed sorted.
            Passing sorting indices (or True) greatly improves the efficiency of this
            function, saving a sort of ``y``.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        y_finite : array-like bool
            Classification of given points according to the threshold.
        """
        if not self.fitted:
            raise ValueError("This classifier has not been trained yet!")
        y_finite = np.full_like(y, False, dtype=bool)
        y_finite[self.i_finite_y(y, i_sorted=i_sorted, validate=validate)] = True
        return y_finite

    # Alias for Classifier.predict, which should be implemented per class.
    def is_finite_X(self, X, validate=True):
        """
        Returns ``True`` for locations predicted to have a finite posterior probability
        density, based on the output of the classifier's ``predict`` method.

        Parameters
        ----------
        X : array-like, shape = (n_samples, dimensionality)
            Training data.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        X_finite : array-like bool
            Whether each of the points are located inside the trust region.
        """
        if not self.fitted:
            raise ValueError("This classifier has not been trained yet!")
        return self.predict(X, validate=validate)

    def fit(
        self, X, y, nstd_calculator=None, keep_min=None, i_sorted=None, validate=True
    ):
        """
        Fits the classifier to some X, y, given a threshold (differential, see below).

        If the representation space of the ``y``'s has changed, it is necessary to pass an
        updated ``nstd_calculator`` to update the thresholds.

        Parameters
        ----------
        X : array-like, shape = (n_samples, dimensionality)
            Training data.

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values. They can have any value smaller than positive infinity,
            including negative infinity.

        nstd_calculator : callable
            Function able to translate threshold values from sigma units to the space in
            which the classifiers are defined (including preprocessing, if present).

        keep_min : int, optional
            If passed, and there are fewer than ``keep_min`` points above the threshold,
            the threshold is ignored and the ``keep_min`` largest points are returned.
            Exception: fewer than ``keep_min`` points in ``y`` that are not ``-np.inf``.
            If none passed, uses by default ``max(2, dimensionality)``.

        i_sorted : bool or array-like, int, 1-dimensional, optional
            Sorting indices for ``y``. If True passed, ``y`` is assumed sorted.
            Passing sorting indices (or True) greatly improves the efficiency of this
            function, saving a sort of ``y``.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        i_finite : array-like int
            Indices of elements in ``y`` classified as "finite" according to the
            threshold, in sorting order.
        """
        # This function needs to call i_finite_threshold and store the 2nd returned
        # value as self._current_threshold
        pass

    @staticmethod
    def i_finite_threshold(
        y, threshold, is_abs=False, keep_min=None, i_sorted=None, validate=True
    ):
        """
        Selects indices of finite y's. More efficient if sorting indiced for y are
        passed.

        Parameters
        ----------
        y : array-like, 1-dimensional
            Array of input values. They can have any value smaller than positive infinity,
            including negative infinity.

        threshold : numeric
            Differential threshold below the maximum element in ``y`` over which points
            are selected as "finite". Must be positive, can be ``np.inf``.

        is_abs : bool (default: False)
            If True, the threshold is understood as absolute, instead of a delte with
            respect to the max value in ``y``.

        keep_min : int, optional
           If passed, and there are fewer than ``keep_min`` points above the threshold,
            the threshold is ignored and the ``keep_min`` largest points are returned.
            Exception: fewer than ``keep_min`` points in ``y`` that are not ``-np.inf``.

        i_sorted : bool or array-like, int, 1-dimensional, optional
            Sorting indices for ``y``. If True passed, ``y`` is assumed sorted.
            Passing sorting indices (or True) greatly improves the efficiency of this
            function, saving a sort of ``y``.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        (i_sorted_finite, threshold)
            Indices of elements in ``y`` classified as "finite" according to the
            threshold, in sorting order, and actual threshold used, which is different
            from the given one only if ``keep_min`` needed to be used to enlarge the set
            of finite points.
        """
        # The structure of this function looks quite complicated, but it is implemented
        # so that it takes advantage of the sorting as much as possible before calling
        # an array operation (np.searchsorted).
        # Here j's are indices of i_sorted, and i's are indices of y
        assert threshold >= 0, "Pass a positive threshold."
        assert keep_min is None or keep_min >= 0, "Pass a positive keep_min."
        if validate:
            y = check_array(y, ensure_2d=False, dtype="numeric")
            if np.any(np.isnan(y)):
                raise TypeError("y cannot contain nan's,")
        if not len(y):
            return [], threshold
        # Since points can actually be -inf and we should never return those, we cannot
        # simply return all points if there are fewer than keep_min, and checking for
        # -inf's is more expensive overall than using the sorted list.
        if i_sorted is None or i_sorted is False:
            i_sorted = np.argsort(y)
        elif i_sorted is True:
            i_sorted = np.arange(len(y))
        if validate:
            if i_sorted.dtype != int or i_sorted.shape != y.shape:
                raise TypeError("i_sorted needs to be sorting indices for y")
        if is_abs:
            threshold = y[i_sorted[-1]] - threshold
        # Quick corner case: if threshold is inf, just return whether elements are finite.
        # (assumes y != positive infinity)
        if threshold == np.inf:
            return i_sorted[np.argwhere(np.isfinite(y[i_sorted])).T[0]], threshold
        if y[i_sorted[-1]] == -np.inf:
            return []
        min_accepted_y = y[i_sorted[-1]] - threshold
        # Let's try to avoid calling searchsorted
        if y[i_sorted[0]] > min_accepted_y:
            return i_sorted, threshold
        if keep_min is not None and keep_min > 0:
            j_first_or_keep_min = -min(
                keep_min, len(y)
            )  # account for len(y) < keep_min
            # Corner case: needs to start on the first non-infinite value.
            if y[i_sorted[j_first_or_keep_min]] == -np.inf:
                j_first_or_keep_min = -1 + next(
                    -(j + 1)
                    for j in range(-j_first_or_keep_min)
                    if y[i_sorted[-j + 1]] > -np.inf
                )
            y_first_or_keep_min = y[i_sorted[j_first_or_keep_min]]
            if y_first_or_keep_min < min_accepted_y:
                enlarged_threshold = max(
                    threshold, y[i_sorted[-1]] - y_first_or_keep_min
                )
                return i_sorted[j_first_or_keep_min:], enlarged_threshold
        j_first_finite = np.searchsorted(
            y, min_accepted_y, side="left", sorter=i_sorted
        )
        return i_sorted[j_first_finite:], threshold


class SVM(SVC, ThresholdClassifier):
    r"""
    Wrapper for the sklearn `RBF kernel SVM <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.

    Classifies points as finite of non-finite, in order to exclude the latter from the
    training set of a parent GPR. It keeps track of the full training set, including
    classified-infinite points.

    The classification is performed using a threshold understood as a positive difference
    against the current maximum ``y`` in the training set. The threshold is passed at
    fitting time and not at initialisation, in case the classifier is defined in a
    transformed coordinate space, with a transformation that changes through the training
    of the parent GPR. (NB: passing the threshold every time is a compromise that allows
    to keep the full training set stored in this object with non-reduced ``y`` values
    while avoiding preprocessing overhead.)

    Also in case there is a coordinate transformation, the training set of this object
    should not be obtained directly, but via properties of the parent GP instead that will
    undo the transformation. The same applying to calling any method directly.

    Parameters
    ----------
    threshold : numeric
        Differential threshold below the maximum element in ``y`` over which points
        are selected as "finite". Must be positive, can be ``np.inf``.

    nstd_calculator : callable
        Function able to translate threshold values from sigma units to the space in which
        the classifiers are defined (including preprocessing, if present).


    C : float, default=1e7
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
        a callable.
        If none is given, 'rbf' will be used. If a callable is given it is
        used to pre-compute the kernel matrix from data matrices; that matrix
        should be an array of shape ``(n_samples, n_samples)``.

    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        * if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        * if 'auto', uses 1 / n_features.


    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.

    probability : bool, default=False
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, will slow down that method as it internally uses
        5-fold cross-validation, and `predict_proba` may be inconsistent with
        `predict`.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    class_weight : dict or 'balanced', default=None
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    decision_function_shape : {'ovo', 'ovr'}, default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one
        ('ovo') is always used as multi-class strategy. The parameter is
        ignored for binary classification.

    break_ties : bool, default=False
        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
        predict will break ties according to the confidence values of
        decision_function; otherwise the first class among the tied
        classes is returned. Please note that breaking ties comes at a
        relatively high computational cost compared to a simple predict.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when `probability` is False.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------

    all_finite : bool
        Is true when all posterior values which have been sampled are finite
        which removes the need for fitting the SVM.

    class_weight_ : ndarray of shape (n_classes,)
        Multipliers of parameter C for each class.
        Computed based on the ``class_weight`` parameter.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    coef_ : ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.
        `coef_` is a readonly property derived from `dual_coef_` and
        `support_vectors_`.

    dual_coef_ : ndarray of shape (n_classes -1, n_SV)
        Dual coefficients of the support vector in the decision
        function, multiplied by
        their targets.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial.

    fit_status_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)

    intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
        Constants in decision function.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during fit. Defined only when `X`
        has feature names that are all strings.

    support_ : ndarray of shape (n_SV)
        Indices of support vectors.

    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.

    n_support_ : ndarray of shape (n_classes,), dtype=int32
        Number of support vectors for each class.

    probA_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
    probB_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
        If `probability=True`, it corresponds to the parameters learned in
        Platt scaling to produce probability estimates from decision values.
        If `probability=False`, it's an empty array. Platt scaling uses the
        logistic function
        ``1 / (1 + exp(decision_value * probA_ + probB_))``
        where ``probA_`` and ``probB_`` are learned from the dataset..

    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        Array dimensions of training vector ``X``.
    """

    def __init__(
        self,
        threshold,
        nstd_calculator,
        C=1e7,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        ThresholdClassifier.__init__(self, threshold, nstd_calculator)
        self.all_finite = None
        # In the SVM, since we have not wrapper the calls to the RNG,
        # (as we have for the GPR), we need to repackage the new numpy Generator
        # as a RandomState, which is achieved by gpry.tools.check_random_state
        random_state = check_random_state(random_state, convert_to_random_state=True)
        SVC.__init__(
            self,
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

    def set_random_state(self, random_state):
        self.random_state = check_random_state(
            random_state, convert_to_random_state=True
        )

    def fit(
        self, X, y, nstd_calculator=None, keep_min=None, i_sorted=None, validate=True
    ):
        r"""
        Fits the SVM with two categorial classes:

        * :math:`\tilde{y}=True` Finite points
        * :math:`\tilde{y}=False` Infinite points

        where :math:`\tilde{y}` is produced after checking the input ``y``'s against
        an internal threshold value, which may also be adjusted at this step.

        If the representation space of the ``y``'s has changed, it is necessary to pass an
        updated ``nstd_calculator`` to update the thresholds.

        Parameters
        ----------
        X : array-like, shape = (n_samples, dimensionality)
            Training data.

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values. They can have any value smaller than positive infinity,
            including negative infinity.

        nstd_calculator : callable
            Function able to translate threshold values from sigma units to the space in
            which the classifiers are defined (including preprocessing, if present).

        keep_min : int, optional
            If passed, and there are fewer than ``keep_min`` points above the threshold,
            the threshold is ignored and the ``keep_min`` largest points are returned.
            Exception: fewer than ``keep_min`` points in ``y`` that are not ``-np.inf``.
            If none passed, uses by default ``max(2, dimensionality)``.

        i_sorted : bool or array-like, int, 1-dimensional, optional
            Sorting indices for ``y``. If True passed, ``y`` is assumed sorted.
            Passing sorting indices (or True) greatly improves the efficiency of this
            function, saving a sort of ``y``.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        i_finite : array-like int
            Indices of elements in ``y`` classified as "finite" according to the
            threshold, in sorting order.
        """
        if nstd_calculator is not None:
            self.update_threshold_definition(nstd_calculator)
        if keep_min is None:
            keep_min = max(2, len(X))
        if validate:  # make sure X.shape[1] is the dimensionality
            X = check_array(X, ensure_2d=True, dtype="numeric")
        i_finite, current_threshold = self.i_finite_threshold(
            y,
            self._diff_threshold,
            keep_min=keep_min,
            i_sorted=i_sorted,
            validate=validate,
        )
        if len(i_finite) == 0:
            self._current_threshold = None
            return
        self._current_threshold = current_threshold
        if validate:
            if X.shape[0] != y.shape[0]:
                raise TypeError(
                    f"Different numbers of points in X (shape {X.shape}) and y (shape "
                    f"{y.shape})."
                )
        # If no value below the threshold, nothing to do. Save test for faster checks.
        if len(i_finite) == len(y):
            self.all_finite = True
            return i_finite
        self.all_finite = False
        y_finite = np.full_like(y, False, dtype=bool)
        y_finite[i_finite] = True
        # Stupid attribute assignment bc sklearn tries to be too clever at validation
        attrs = ["nstd_calculator", "threshold"]
        for attr in attrs:
            setattr(self, attr, None)
        super(SVC, self).fit(X, y_finite)
        for attr in attrs:
            delattr(self, attr)
        return i_finite

    def predict(self, X, validate=True):
        """
        Wrapper for the predict method of the SVM. Returns a boolean array which is true
        at locations where the SVM predicts a finite posterior distribution and False
        where it predicts infinite values.

        Parameters
        ----------
        X : array-like, shape = (n_samples, dimensionality)
            Training data.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        X_finite : array-like bool
            A boolean array which is True at locations predicted finite posterior
            and False at locations with predicted infinite posterior.

        Raises
        ------
        ValueError: "ndarray is not C-contiguous"
           May be raised if ``validate`` is False. Call ``numpy.ascontiguousarray()`` on
           the input before the call.
        """
        if not self.fitted:
            raise ValueError("This classifier has not been trained yet!")
        if validate:
            X = np.ascontiguousarray(check_array(X, ensure_2d=True, dtype="numeric"))
        if self.all_finite:
            return np.full(len(X), True)
        if validate:
            return SVC.predict(self, X)
        else:  # valid for our use only (dense, 2 classes), when input is guaranteed valid
            y = self._dense_predict(X)
            return self.classes_.take(np.asarray(y, dtype=np.intp))


class TrustRegion(ThresholdClassifier):
    """
    Class managing the region within we trust the model to be finite, as a hyper-rectangle
    defined by the points with log-posterior over some value, expressed as a different
    with the current maximum value.

    Parameters
    ----------
    threshold : numeric
        Differential threshold below the maximum element in ``y`` over which points
        are selected as "finite". Must be positive, can be ``np.inf``.

    nstd_calculator : callable
        Function able to translate threshold values from sigma units to the space in which
        the classifiers are defined (including preprocessing, if present).

    bounds : array-like, shape = (dimensionality, 2)
        Bounds of the problem

    factor : float (positive), optional
        If defined as a positive float, enlarges the size of the minimal hyper-rectangle
        by the given factor. If undefined, it keeps the boundaries defined by the
        threshold.
    """

    def __init__(self, threshold, nstd_calculator, bounds, factor=None):
        self._init_bounds = np.atleast_2d(bounds)
        self._trust_bounds = np.copy(self._init_bounds)
        if factor is not None and factor <= 0:
            raise TypeError("Please pass a positive 'factor' or None (alias for 1).")
        elif factor is None:
            factor = 1
        self.factor = factor
        ThresholdClassifier.__init__(self, threshold, nstd_calculator)

    @property
    def d(self):
        """Dimension of the space."""
        return self._init_bounds.shape[0]

    @property
    def trust_bounds(self):
        """
        Returns a copy of the trust bounds (the original ones if never fit or
        ``factor=None``).
        """
        return np.copy(self._trust_bounds)

    def fit(
        self, X, y, nstd_calculator=None, keep_min=None, i_sorted=None, validate=True
    ):
        """
        Adjusts the boundaries of the trust region so that it contains the points over
        the theshold. More efficient if sorting indiced for y are passed.
        Indices are returned as sorting indices.

        If the representation space of the ``y``'s has changed, it is necessary to pass an
        updated ``nstd_calculator`` to update the thresholds.

        Parameters
        ----------
        X : array-like, shape = (n_samples, dimensionality)
            Training data.

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values. They can have any value smaller than positive infinity,
            including negative infinity.

        nstd_calculator : callable
            Function able to translate threshold values from sigma units to the space in
            which the classifiers are defined (including preprocessing, if present).

        keep_min : int, optional
            If passed, and there are fewer than ``keep_min`` points above the threshold,
            the threshold is ignored and the ``keep_min`` largest points are returned.
            Exception: fewer than ``keep_min`` points in ``y`` that are not ``-np.inf``.
            If none passed, uses by default ``max(2, dimensionality)``.

        i_sorted : bool or array-like, int, 1-dimensional, optional
            Sorting indices for ``y``. If True passed, ``y`` is assumed sorted.
            Passing sorting indices (or True) greatly improves the efficiency of this
            function, saving a sort of ``y``.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        i_finite : array-like int
            Indices of elements in ``y`` classified as "finite" according to the
            threshold, in sorting order.
        """
        if nstd_calculator is not None:
            self.update_threshold_definition(nstd_calculator)
        if keep_min is None:
            keep_min = max(2, self.d)
        i_finite, current_threshold = self.i_finite_threshold(
            y,
            self._diff_threshold,
            keep_min=keep_min,
            i_sorted=i_sorted,
            validate=validate,
        )
        if len(i_finite) == 0:
            self._current_threshold = None
            return
        self._current_threshold = current_threshold
        if validate:
            X = check_array(X, ensure_2d=True, dtype="numeric")
            if X.shape[0] != y.shape[0]:
                raise TypeError(
                    f"Different numbers of points in X (shape {X.shape}) and y (shape "
                    f"{y.shape})."
                )
        use_X = X[i_finite]
        self._trust_bounds = shrink_bounds(self._init_bounds, use_X, factor=self.factor)
        return i_finite

    def predict(self, X, validate=True):
        """
        Returns ``True`` for locations inside the trust region, and ``False`` otherwise.

        Parameters
        ----------
        X : array-like, shape = (n_samples, dimensionality)
            Training data.

        validate : bool (default: True)
            If passed, check consistency of input arrays.

        Returns
        -------
        X_finite : array-like bool
            Whether each of the points are located inside the trust region.
        """
        if not self.fitted:
            raise ValueError("This classifier has not been trained yet!")
        return is_in_bounds(X, self._trust_bounds, validate=validate)
