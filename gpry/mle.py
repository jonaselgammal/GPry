# general purpose stuff
import warnings
from copy import deepcopy
from operator import itemgetter

# numpy and scipy
import numpy as np
from scipy.linalg import cholesky, solve_triangular
import scipy.optimize

# gpry kernels and SVM
from gpry.kernels import RBF, Matern, ConstantKernel as C
from gpry.svm import SVM
from gpry.preprocessing import Normalize_bounds
from gpry.tools import check_random_state
from gpry.tools import nstd_of_1d_nstd

# sklearn GP and kernel utilities
from sklearn.gaussian_process import GaussianProcessRegressor \
    as sk_GaussianProcessRegressor
from sklearn.base import clone, BaseEstimator as BE

# sklearn utilities
from sklearn.utils.validation import check_array

from scipy.optimize import differential_evolution


def objective_function(x, logpost, mle):
    val = logpost.logpost(x)
    if len(mle.X_values) == 0:
        mle.X_values = np.atleast_2d(x)
        mle.y_values = np.array([val])
    else:
        mle.X_values = np.append(mle.X_values, [x], axis=0)
        mle.y_values = np.append(mle.y_values, val)
    return -1*val

class MLE:
    r"""
    Maximum Likelihood estimation to reduce the number of acquired -inf in the initial training set.

    STILL IN TESTING PHASE
    """

    def __init__(self, model, bounds=None):
        self.model = model

        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = self.model.prior.bounds(confidence_for_unbounded=0.99995)

        self.log_prior_volume = np.sum(np.log(self.bounds[:,1] - self.bounds[:,0]))

        self.n_d = self.bounds.shape[0]
        self.equivalent_countour_depth = nstd_of_1d_nstd(5, self.n_d)
        print(self.equivalent_countour_depth)
        self.X_values = np.array([])
        self.y_values = np.array([])
        print(self.bounds)
    
    def callback(self, x, convergence):
        """
        print(x)
        if len(self.X_values) == 0:
            self.X_values = np.atleast_2d(x)
            self.y_values = np.array([])
        else:
            self.X_values = np.append(self.X_values, [x], axis=0)
        # self.y_values = np.append(self.y_values)
        """
        # print(np.max(self.y_values)+self.log_prior_volume)
        # print(convergence)

    def find_maximum_posterior(self):

        def callback_callable(x, convergence=None):
            return self.callback(x, convergence)
        
        res = differential_evolution(objective_function, self.bounds, workers=1, 
            args=[self.model, self], tol=0.3, callback=callback_callable)

        self.best_point = res.x
        self.best_val = -res.fun

        print(res.x)
        print("### max-real max ###")
        print(-res.fun+self.log_prior_volume)
        print("####################")
        print(res.nfev)
        print(res.success)

    def get_training_from_optim(self, n_points=5):
        candidates = self.y_values >= self.best_val-self.equivalent_countour_depth
        
        if np.sum(candidates) < n_points-1:
            print("Number of requested points larger than points in 5-sigma interval.")
            print(f"Requested {n_points} points. Returning {np.sum(candidates)+1} instead.")

        val_indices = np.random.choice(np.sum(candidates), size=n_points-1)
        
        X_values_choice = self.X_values[candidates, :][val_indices, :]
        y_values_choice = self.y_values[candidates][val_indices]

        X_values_choice = np.append(X_values_choice, [self.best_point], axis=0)
        y_values_choice = np.append(y_values_choice, [self.best_val])

        return X_values_choice, y_values_choice

        # print(self.y_values)
