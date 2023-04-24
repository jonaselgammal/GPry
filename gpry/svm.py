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

import numpy as np
from sklearn.svm import SVC
from gpry.tools import nstd_of_1d_nstd


class SVM(SVC):
    r"""Wrapper for the sklearn `RBF kernel SVM <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_. Implements the same
    "append_to_data" function as the GP Regressor and is designed to be passed
    to a GP regressor. This is done to classify the data into a "finite" group
    (values with a finite log-likelihood) and an "infinite" group. That way the
    GP can correctly recover the log-likelihood or log-posterior even if has
    regions where it returns :math:`-\infty` or very low log-likelihood values.
    The threshold for what is considered infinite is either set by the
    ``threshold_sigma`` parameter or by the ``threshold`` parameter. This is to
    account for the fact that the log-likelihood can take very low values far
    away from the mode which can confuse the GP.
    Also saves the training data internally and has a function to return all
    non-infinite values. Therefore it is supposed to be used inside of a GP
    Regressor.
    Furthermore this accounts for the fact that all values in the SVM might be
    finite and therefore the SVM can be ignored.


    Parameters
    ----------
    threshold_sigma : float or None, default=10
        Distance to the mode which shall be considered finite in :math:`\sigma`
        using a :math:`\chi^2` distribution. Either this or ``threshold`` have
        to be specified while ``threshold_sigma`` it is overwritten by the
        ``threshold`` parameter
    threshold : float or None, default=None
        threshold value for the posterior to be considered infinite. Any value
        below this will be in the infinite category. Overwrites the
        ``threshold_sigma`` parameter if specified. If you want to consider all
        samples where the posterior returns a finite value set this to
        ``-np.inf``.
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

    preprocessing_X : X-preprocessor or pipeline, optional (default=None)
        The transformation in X-direction (parameter space of the posterior)
        that shall be used to preprocess the data before fitting to the SVM.
    preprocessing_y : y-preprocessor or pipeline, optional (default=None)
        The transformation in y-direction (posterior value) that shall be used
        before fitting to the SVM.
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

    def __init__(self, threshold_sigma=20, threshold=None, C=1e7, kernel='rbf',
                 degree=3, gamma='scale', preprocessing_X=None, preprocessing_y=None,
                 coef0=0.0, shrinking=True, probability=False, tol=0.001,
                 cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', break_ties=False, random_state=None):

        self.threshold_sigma = threshold_sigma
        self.init_threshold = threshold
        # Current threshold (and preprocessed one ending in "_")
        self._threshold = None
        self._threshold_ = None

        self.preprocessing_X = preprocessing_X
        self.preprocessing_y = preprocessing_y
        self.all_finite = False
        self.newly_appended = 0

        # In the SVM, since we have not wrapper the calls to the RNG,
        # (as we have for the GPR), we need to repackage the new numpy Generator
        # as a RandomState, which is achieved by gpry.tools.check_random_state
        from gpry.tools import check_random_state
        random_state = check_random_state(random_state, convert_to_random_state=True)

        super().__init__(C=C, kernel=kernel, degree=degree, gamma=gamma,
                         coef0=coef0, shrinking=shrinking, probability=probability,
                         tol=tol, cache_size=cache_size, class_weight=class_weight,
                         verbose=verbose, max_iter=max_iter,
                         decision_function_shape=decision_function_shape,
                         break_ties=break_ties, random_state=random_state)

    @property
    def d(self):
        """Dimension of the feature space."""
        try:
            return self.X_train.shape[1]
        except AttributeError:
            raise ValueError(
                "You need to add some data before determining its dimension.")

    @property
    def n(self):
        """Number of training points."""
        return len(getattr(self, "y_train", []))

    @property
    def last_appended(self):
        """Returns a copy of the last appended training points, as (X, y in [0, 1])."""
        return (np.copy(self.X_train[-self.newly_appended:]),
                np.copy(self.y_train[-self.newly_appended:]))

    def append_to_data(self, X, y, fit_preprocessors=True):
        """
        This method works similarly to the GP regressor's
        :meth:`append_to_data <gpr.GaussianProcessRegressor.append_to_data>` method.
        This means that it adds samples and internally calls the fit method of the SVM.
        It furthermore provides the option to fit the preprocessor(s) (if they need
        fitting).

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data to append to the model.

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values to append to the data

        fit_preprocessors : bool, optional (default: True)
            Whether the preprocessors are to be refit. This only applies if
            the transformation of at least one of the preprocessors depends on
            the samples.

        Returns
        -------
        self
        """
        # Copy stuff
        X = np.copy(X)
        y = np.copy(y)
        self.newly_appended = len(y)

        # Check if X_train and y_train exist to see if a model
        # has previously been fit to the data
        if (hasattr(self, "X_train_") and hasattr(self, "y_train_")):
            X_train = np.append(self.X_train, X, axis=0)
            y_train = np.append(self.y_train, y)
        else:
            X_train = X
            y_train = y
        # Fit SVM
        self.fit(X_train, y_train, fit_preprocessors=fit_preprocessors)

        return self.finite[-self.newly_appended:]

    def fit(self, X, y, fit_preprocessors=True):
        r"""
        Wrapper for the fit value of the sklearn SVM which fits the SVM with
        two categorial classes:

        * :math:`\\tilde{y}=1` Finite points
        * :math:`\\tilde{y}=0` Infinite points

        where :math:`\\tilde{y}` is produced by the fit method after
        preprocessing the sampling locations and posterior values.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data to append to the model.

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values to append to the data

        fit_preprocessors : bool, optional (default: True)
            Whether the preprocessors are to be refit. This only applies if
            the transformation of at least one of the preprocessors depends on
            the samples.

        Returns
        -------
        self
        """
        # Copy X_train and y_train to be able to reproduce stuff
        self.X_train = np.copy(X)
        self.y_train = np.copy(y)

        # Preprocess if neccessary
        if self.preprocessing_X is not None:
            if fit_preprocessors:
                self.preprocessing_X.fit(X, y)
            self.X_train_ = self.preprocessing_X.transform(X)
        else:
            self.X_train_ = X
        if self.preprocessing_y is not None:
            if fit_preprocessors:
                self.preprocessing_y.fit(X, y)
            self.y_train_ = self.preprocessing_y.transform(y)
        else:
            self.y_train_ = y

        # Update threshold value
        self.update_threshold(self.d)

        # Turn into categorial values (1 for finite and 0 for infinite)
        self.finite = self.is_finite(self.y_train_)

        # Check if all values belong to one class, in that case do not fit the
        # SVM but rather save this.
        if np.all(self.finite):
            self.all_finite = True
        elif np.all(~self.finite):
            raise ValueError("All values that have been passed are infinite. "
                             "This cannot be tolerated as it breaks the GP.")
        else:
            super().fit(self.X_train_, self.finite)
        return self.finite

    def is_finite(self, y, y_is_preprocessed=True):
        """
        Returns True for finite values above the current threshold, and False otherwise.
        """
        threshold = self.threshold_preprocessed if y_is_preprocessed else self.threshold
        return np.logical_and(np.isfinite(y), y - np.max(y) > threshold)

    def predict(self, X):
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

        """
        # Check if all training values were finite, then just return one for
        # every value
        if self.all_finite:
            return np.ones(X.shape[0], dtype=bool)
        # preprocess to the right dimensions if neccessary
        if self.preprocessing_X is not None:
            X = self.preprocessing_X.transform(X)
        return super().predict(X)

    @property
    def threshold(self):
        """
        Returns the threshold value which is used to determine whether a value
        is considered to be -inf.
        """
        return self._threshold

    @property
    def threshold_preprocessed(self):
        """
        Returns the threshold value which is used to determine whether a value
        is considered to be -inf, that threshold having been preprocessed.
        """
        return self._threshold_

    def update_threshold(self, dimension=None):
        """
        Sets the threshold value threshold for un-transformed y's.
        """
        dimension = dimension or self.d
        # if self._threshold is None:
        if self.init_threshold is None:
            if self.threshold_sigma is None:
                raise ValueError(
                    "You either need to specify threshold or threshold_sigma.")
            self._threshold = \
                self.compute_threshold_given_sigma(self.threshold_sigma, dimension)
        else:
            self._threshold = self.init_threshold
        # Update threshold for preprocessed data
        if self.preprocessing_y is not None:
            self._threshold_ = self.preprocessing_y.transform(self._threshold)
        else:
            self._threshold_ = self._threshold

    @staticmethod
    def compute_threshold_given_sigma(n_sigma, n_dimensions):
        r"""
        Computes threshold value given a number of :math:`\sigma` away from the maximum,
        assuming a :math:`\chi^2` distribution.
        """
        return -0.5 * nstd_of_1d_nstd(n_sigma, n_dimensions)**2
