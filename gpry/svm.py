r"""
This module uses a Support vector machine (SVM) with an RBF kernel to classify
regions which are "safe" to explore in contrast to regions which are "unsafe"
to explore since they are infinite. This is done in an attempt to hinder the
exploration of parts of the parameter space which have a :math:`-\infty` log-posterior
value. These values need to be filtered out since feeding them to the GP
Regressor will break it. Nevertheless this is important information that we
shouldn't throw away. We will also need the SVM later when doing the MCMC run
to tell the model which regions it shouldn't visit. In essence our process
shrinks the prior to a region where the model thinks that all values of the
log-posterior distribution are finite.
"""

import warnings
import numpy as np
from sklearn.svm import SVC
from gpry.tools import check_random_state


class SVM(SVC):
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
        self.X_train = None
        self.y_train = None
        self.y_finite = None
        self.at_least_one_finite = False
        self.all_finite = False
        self.diff_threshold = None
        self._max_y = None
        # In the SVM, since we have not wrapper the calls to the RNG,
        # (as we have for the GPR), we need to repackage the new numpy Generator
        # as a RandomState, which is achieved by gpry.tools.check_random_state
        random_state = check_random_state(random_state, convert_to_random_state=True)
        super().__init__(
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

    @property
    def d(self):
        """Dimension of the feature space."""
        if self.X_train is None:
            raise ValueError(
                "You need to add some data before determining its dimension."
            )
        return self.X_train.shape[1]

    @property
    def n(self):
        """Number of training points."""
        if self.y_train is None:
            return 0
        return len(self.y_train)

    def fit(self, X, y, diff_threshold):
        r"""
        Fits the SVM with two categorial classes:

        * :math:`\tilde{y}=True` Finite points
        * :math:`\tilde{y}=False` Infinite points

        where :math:`\tilde{y}` is produced after checking the input ``y``'s against
        an internal threshold value, which may also be adjusted at this step.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data.

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values.

        Returns
        -------
        y_finite : array-like bool
            Classification of current points.
        """
        self.X_train = np.copy(X)
        self.y_train = np.copy(y)
        # Corner case: only -inf points being trained on: nothing to do.
        if np.all(self.y_train == -np.inf):
            self.at_least_one_finite = False
            self.y_finite = np.full(len(X), False)
            return self.y_finite
        self.at_least_one_finite = True
        # Update threshold value
        self.diff_threshold = diff_threshold
        self._max_y = max(self.y_train)
        # Turn into boolean categorial values
        self.y_finite = self._is_finite_raw(
            self.y_train, self.diff_threshold, max_y=self._max_y
        )
        # If no value below the threshold, nothing to do. Save test for faster checks.
        if np.all(self.y_finite):
            self.all_finite = True
            return self.y_finite
        self.all_finite = False
        super().fit(self.X_train, self.y_finite)
        return self.y_finite

    @staticmethod
    def _is_finite_raw(y, diff_threshold, max_y=None):
        """
        Returns the indices of the finite points, depending on some delta-like threshold,
        in the same space (transformed or not) as the y's.
        """
        if max_y is None:
            max_y = np.max(y)
        return np.greater_equal(y, max_y - diff_threshold)

    def is_finite(self, y=None):
        """
        Returns True for finite values above the current threshold, and False otherwise.

        Notes
        -----
        This is not a predictor method, but a simple threshold check, i.e. it does not
        predict whether the value at some particular location is expected to be finite.
        For that purpose, use the ``predict`` method.
        """
        if y is None:
            y = self.y_train
        else:
            warnings.warn(
                "Calling '.is_finite_()' with an argument: its result is only consistent "
                "when calling with the training set or a subset of it, but not when "
                "calling with points not yet in the training set, since they may change "
                "the threshold after addition."
            )
        if self.y_train is None:
            raise ValueError(
                "The SVM has not been trained yet, so no check can be performed, "
                "since classifying thresholds are defined as a difference with the "
                "maximum."
            )
        if not self.at_least_one_finite:
            raise ValueError(
                "The SVM has not received any finite training points yet, so no check can"
                " be performed, since classifying thresholds are defined as a difference "
                "with a finite maximum."
            )
        return np.atleast_1d(y) > self.abs_threshold

    def predict(self, X, validate=True):
        """
        Wrapper for the predict method of the SVM which does the preprocessing.
        Returns a boolean array which is true at locations where the SVM
        predicts a finite posterior distribution and False where it predicts
        infinite values.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where SVM is evaluated.

        Returns
        -------
        A boolean array which is True at locations predicted finite posterior
        and False at locations with predicted infinite posterior.

        Raises
        ------
        ValueError: "ndarray is not C-contiguous"
           May be raised if ``validate`` is False. Call ``numpy.ascontiguousarray()`` on
           the input before the call.
        """
        if self.y_train is None:
            raise ValueError("The SVM has not been trained yet.")
        if validate:
            X = np.atleast_2d(X)
        if self.all_finite:
            return np.full(len(X), True)
        if not self.at_least_one_finite:
            warnings.warn(
                "Only -inf points added to the classifier so far. "
                "Returning False unconditionally."
            )
            return np.full(len(X), False)
        if validate:
            return super().predict(X)
        else:  # valid for our use only (dense, 2 classes), when input is guaranteed valid
            y = self._dense_predict(X)
            return self.classes_.take(np.asarray(y, dtype=np.intp))
