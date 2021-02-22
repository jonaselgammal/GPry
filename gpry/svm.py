"""
This module uses a Support vector machine (SVM) with an RBF kernel to classify
regions which are "safe" to explore in contrast to regions which are "unsafe"
to explore since they are infinite. This is done in an attempt to hinder the
exploration of parts of the parameter space which have a -inf log-posterior
value. These values need to be filtered out since feeding them to the GP
Regressor will break it. Nevertheless this is important information that we
shouldn't throw away. We will also need the SVM later when doing the MCMC run
to tell the model which regions it shouldn't visit. In essence our process
shrinks the prior to a region where the model thinks that all values of the
log-posterior distribution are finite.
"""

import numpy as np
from sklearn.svm import SVC

class SVM(SVC):
    """Wrapper for the sklearn RBF kernel SVM. Implements the same
    "append_to_data" function as the GP Regressor and is designed to be passed
    to a GP regressor. This is done to classify the data into a "finite" group
    (values with a finite log-likelihood) and an "infinite" group. That way the
    GP can correctly recover the log-likelihood or log-posterior even if has
    regions where it returns -inf.
    Also saves the training data internally and has a function to return all
    non-infinite values. Therefore it is supposed to be used inside of a GP
    Regressor.
    Furthermore this accounts for the fact that all values in the SVM might be
    finite and therefore the SVM can be ignored.
    """

    def __init__(self, C=np.inf, kernel='rbf', degree=3, gamma='scale',
        preprocessing_X=None, preprocessing_y=None,
        coef0=0.0, shrinking=True, probability=False, tol=0.001,
        cache_size=200, class_weight=None, verbose=False, max_iter=-1,
        decision_function_shape='ovr', break_ties=False, random_state=None):

        self.preprocessing_X = preprocessing_X
        self.preprocessing_y = preprocessing_y
        self.all_finite = False

        super().__init__(C=C, kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, shrinking=shrinking, probability=probability, tol=tol,
            cache_size=cache_size, class_weight=class_weight, verbose=verbose,
            max_iter=max_iter, decision_function_shape=decision_function_shape,
            break_ties=break_ties, random_state=random_state)

    def append_to_data(self, X, y, fit_preprocessors=True):
        """
        Append to data method works similarly to that of the GP Regressor.
        Returns all samples which are finite
        """

        # Copy stuff
        X = np.copy(X)
        y = np.copy(y)
        newly_appended = len(y)

        # Check if X_train and y_train exist to see if a model
        # has previously been fit to the data
        if (hasattr(self, "X_train_") and hasattr(self, "y_train_")):
            X_train = np.append(self.X_train, X, axis=0)
            y_train = np.append(self.y_train, y)
        # Fit SVM
        self.fit(X_train, y_train, fit_preprocessors=fit_preprocessors)

        return self.finite[-newly_appended:]

    def fit(self, X, y, fit_preprocessors=True):
        """
        Fits the SVM with two classes:

        * ``y=0``: Finite points
        * ``y=1``: Infinite points
        """
        # Copy X_train and y_train to be able to reproduce stuff
        self.X_train = np.copy(X)
        self.y_train = np.copy(y)

        # preprocess if neccessary
        if self.preprocessing_X is not None:
            if fit_preprocessors:
                self.preprocessing_X.fit(X, y)
            self.X_train_ = self.preprocessing_X.transform(X)
        if self.preprocessing_y is not None:
            if fit_preprocessors:
                self.preprocessing_y.fit(X, y)
            self.y_train_ = self.preprocessing_y.transform(y)

        else:
            self.X_train_ = X
            self.y_train_ = y

        # turn into categorial values (1 for finite and 0 for infinite)
        self.finite = np.isfinite(y)

        # Check if all values belong to one class, in that case do not fit the
        # SVM but rather save this.
        if np.all(self.finite):
            self.all_finite = True
            return self.finite
        elif np.all(~self.finite):
            raise ValueError("All values that have been passed are infinite. "\
                "This cannot be tolerated as it breaks the GP.")
        else:
            super().fit(self.X_train_, self.finite)
            return self.finite

    def predict(self, X):
        """
        Wrapper for the predict method. Does all the pre-transformation.
        """
        # Check if all training values were finite, then just return one for
        # every value
        if self.all_finite:
            return np.ones(X.shape[0], dtype=bool)
        # preprocess to the right dimensions if neccessary
        if self.preprocessing_X is not None:
            X = self.preprocessing_X.transform(X)
        return super().predict(X)
