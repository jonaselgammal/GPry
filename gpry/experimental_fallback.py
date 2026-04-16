"""
Experimental fallback regressors for when the GP struggles.

This module is not part of the conservative GPry baseline. It is kept behind
explicit opt-in options for experiments where the standard GP fit fails.

Provides a RandomForestFallback that implements the same predict() interface
as GaussianProcessRegressor, using sklearn's RandomForestRegressor internally.
Tree variance across the ensemble provides uncertainty estimates.

The fallback is trained alongside the GP and can be used when:
- GP fitting fails (numerical issues, Cholesky decomposition errors)
- GP predictions are unreliable (very high uncertainty everywhere)
- As part of an ensemble blend with the GP
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestFallback:
    """Random Forest fallback regressor with GP-compatible predict() interface.

    Uses the variance across trees as an uncertainty estimate (analogous to
    GP posterior variance). Gradients are approximated via finite differences.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.

    max_depth : int or None, default=None
        Maximum tree depth. None means nodes are expanded until all leaves
        are pure or contain less than min_samples_split samples.

    min_samples_leaf : int, default=5
        Minimum samples per leaf. Higher values = smoother predictions.

    random_state : int or None, default=None
        Random seed for reproducibility.

    fd_eps : float, default=1e-5
        Step size for finite-difference gradient approximation.
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=5,
                 random_state=None, fd_eps=1e-5):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        self.fd_eps = fd_eps
        self._fitted = False
        self.X_train_ = None
        self.y_train_ = None

    @property
    def fitted(self):
        return self._fitted

    def fit(self, X, y):
        """Fit the random forest on training data.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,)

        Returns
        -------
        self
        """
        self.X_train_ = np.copy(X)
        self.y_train_ = np.copy(y)
        self.rf.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X, return_std=False, return_mean_grad=False,
                return_std_grad=False, validate=True):
        """Predict with GP-compatible return interface.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        return_std : bool
            Return standard deviation (tree variance).
        return_mean_grad : bool
            Return gradient of mean (finite differences). Single point only.
        return_std_grad : bool
            Return gradient of std (finite differences). Single point only.

        Returns
        -------
        list of arrays, in order: [y_mean, y_std?, y_mean_grad?, y_std_grad?]
        """
        if return_std_grad and not (return_std and return_mean_grad):
            raise ValueError(
                "Not returning std_gradient without returning "
                "the std and the mean grad."
            )
        X = np.atleast_2d(X)
        # Get per-tree predictions for uncertainty
        if return_std or return_std_grad:
            tree_preds = np.array([t.predict(X) for t in self.rf.estimators_])
            y_mean = tree_preds.mean(axis=0)
            y_std = tree_preds.std(axis=0)
        else:
            y_mean = self.rf.predict(X)
            y_std = None

        result = [y_mean]
        if return_std:
            result.append(y_std)

        if return_mean_grad:
            mean_grad = self._finite_diff_grad(X[0], "mean")
            result.append(mean_grad)

        if return_std_grad:
            std_grad = self._finite_diff_grad(X[0], "std")
            result.append(std_grad)

        return result

    def predict_std(self, X, validate=True):
        """Predict only the standard deviation."""
        X = np.atleast_2d(X)
        tree_preds = np.array([t.predict(X) for t in self.rf.estimators_])
        return tree_preds.std(axis=0)

    def _finite_diff_grad(self, x, target="mean"):
        """Compute gradient via central finite differences.

        Parameters
        ----------
        x : array, shape (n_features,)
            Single query point.
        target : str, "mean" or "std"

        Returns
        -------
        grad : array, shape (1, n_features)
        """
        x = np.asarray(x, dtype=float)
        ndim = len(x)
        grad = np.zeros(ndim)
        eps = self.fd_eps

        for i in range(ndim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            X_pair = np.vstack([x_plus, x_minus])

            if target == "std":
                vals = self.predict_std(X_pair)
            else:
                vals = self.rf.predict(X_pair)

            grad[i] = (vals[0] - vals[1]) / (2.0 * eps)

        return grad.reshape(1, ndim)
