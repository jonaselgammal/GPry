from math import sqrt

import numpy as np
import warnings
from sklearn.gaussian_process.kernels import Kernel as sk_Kernel
from sklearn.gaussian_process.kernels import ConstantKernel \
    as sk_ConstantKernel
from sklearn.gaussian_process.kernels import DotProduct as sk_DotProduct
from sklearn.gaussian_process.kernels import Exponentiation \
    as sk_Exponentiation
from sklearn.gaussian_process.kernels import ExpSineSquared \
    as sk_ExpSineSquared
from sklearn.gaussian_process.kernels import Matern as sk_Matern
from sklearn.gaussian_process.kernels import Product as sk_Product
from sklearn.gaussian_process.kernels import RationalQuadratic \
    as sk_RationalQuadratic
from sklearn.gaussian_process.kernels import RBF as sk_RBF
from sklearn.gaussian_process.kernels import Sum as sk_Sum
from sklearn.gaussian_process.kernels import WhiteKernel as sk_WhiteKernel

from collections import namedtuple

# Copyright (c) 2016-2020 The scikit-optimize developers.
# This module contains (heavily modified) code of the scikit-optimize package.

class Hyperparameter(namedtuple('Hyperparameter',
                                ('name', 'value_type', 'bounds',
                                 'max_length',
                                 'n_elements', 'fixed', 'dynamic'))):
    """A kernel hyperparameter's specification in form of a namedtuple.

    .. note::

        We overwrite the whole class here since the namedtuple approach does not
        allow for easy extension. For more information on this see
        `this link <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Hyperparameter.html>`_

    Attributes
    ----------

    name : str
        The name of the hyperparameter. Note that a kernel using a
        hyperparameter with name "x" must have the attributes self.x and
        self.x_bounds
    value_type : str
        The type of the hyperparameter. Currently, only "numeric"
        hyperparameters are supported.
    bounds : pair of floats >= 0 or "fixed"
        The lower and upper bound on the parameter. If n_elements>1, a pair
        of 1d array with n_elements each may be given alternatively. If
        the string "fixed" is passed as bounds, the hyperparameter's value
        cannot be changed.
    n_elements : int, default=1
        The number of elements of the hyperparameter value. Defaults to 1,
        which corresponds to a scalar hyperparameter. n_elements > 1
        corresponds to a hyperparameter which is vector-valued,
        such as, e.g., anisotropic length-scales.
    fixed : bool, default=None
        Whether the value of this hyperparameter is fixed, i.e., cannot be
        changed during hyperparameter tuning. If None is passed, the "fixed" is
        derived based on the given bounds.
    dynamic : bool, default=None
        Whether the value of this hyperparameter is dynamic, i.e. whether the
        bounds of the hyperparameter should automatically be adjusted to two
        orders of magnitude above and below the current best fit value. If None
        is passed, the "dynamic" is derived based on the given bounds.
    max_length : float or array-like, shape = (n_dimensions,)
        The prior bounds of the posterior distribution (of the parameter-space,
        not the hyperparameter space) is required for hyperparameters which are
        length scales (correlation lengths) if their bounds are set to
        "dynamic". This is done to restrict their range to the same order of
        magnitude as the prior size (actually 2x the prior).
    """

    # A raw namedtuple is very memory efficient as it packs the attributes
    # in a struct to get rid of the __dict__ of attributes in particular it
    # does not copy the string for the keys on each instance.
    # By deriving a namedtuple class just to introduce the __init__ method we
    # would also reintroduce the __dict__ on the instance. By telling the
    # Python interpreter that this subclass uses static __slots__ instead of
    # dynamic attributes. Furthermore we don't need any additional slot in the
    # subclass so we set __slots__ to the empty tuple.
    __slots__ = ()

    def __new__(cls, name, value_type, bounds, max_length, n_elements=1, fixed=None,
                dynamic=None):
        if not isinstance(bounds, str) or (bounds != "fixed" and
                                           bounds != "dynamic"):
            bounds = np.atleast_2d(bounds)
            if n_elements > 1:  # vector-valued parameter
                if bounds.shape[0] == 1:
                    bounds = np.repeat(bounds, n_elements, 0)
                elif bounds.shape[0] != n_elements:
                    raise ValueError("Bounds on %s should have either 1 or "
                                     "%d dimensions. Given are %d"
                                     % (name, n_elements, bounds.shape[0]))

        if fixed is None:
            fixed = isinstance(bounds, str) and bounds == "fixed"
        if dynamic is None:
            dynamic = isinstance(bounds, str) and bounds == "dynamic"
        return super(Hyperparameter, cls).__new__(
            cls, name, value_type, bounds, max_length, n_elements, fixed,
            dynamic)

    # This is mainly a testing utility to check that two hyperparameters
    # are equal.
    def __eq__(self, other):
        return (self.name == other.name and
                self.value_type == other.value_type and
                np.all(self.bounds == other.bounds) and
                self.n_elements == other.n_elements and
                self.fixed == other.fixed and
                self.dynamic == other.dynamic and
                self.max_length == other.max_length)


class Kernel(sk_Kernel):
    """
    Base class for gpry kernels.
    Supports computation of the gradient of the kernel with respect to X

     .. note::
        This kernel class is taken entirely from the Scikit-optimize package.
    """

    def __add__(self, b):
        if not isinstance(b, Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, Kernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def __pow__(self, b):
        return Exponentiation(self, b)

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        r = [getattr(self, attr) for attr in dir(self)
             if attr.startswith("hyperparameter_")]
        return r

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        bounds = []
        params = self.get_params(deep=True)
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                if hyperparameter.dynamic:
                    thetas = params[hyperparameter.name]
                    if np.iterable(thetas):
                        for t, theta in enumerate(thetas):
                            if hyperparameter.max_length[t] is None:
                                bounds.append([theta * 1e-2, theta * 1.])
                            else:
                                bounds.append(
                                    [hyperparameter.max_length[t] * 1e-2,
                                     hyperparameter.max_length[t] * 1e2])
                    else:
                        if hyperparameter.max_length[0] is None:
                            bounds.append([thetas * 1e-2, thetas * 1.])
                        else:
                            bounds.append([hyperparameter.max_length[0] * 1e-2,
                                           hyperparameter.max_length[0] * 1e2])
                else:
                    bounds.append(hyperparameter.bounds)
        if len(bounds) > 0:
            return np.log(np.vstack(bounds))
        else:
            return np.array([])

    def gradient_x(self, x, X_train):
        """
        Computes gradient of K(x, X_train) with respect to x

        Parameters
        ----------
        x: array-like, shape=(n_features,)
            A single test point.

        X_train: array-like, shape=(n_samples, n_features)
            Training data used to fit the gaussian process.

        Returns
        -------
        gradient_x: array-like, shape=(n_samples, n_features)
            Gradient of K(x, X_train) with respect to x.
        """
        raise NotImplementedError


class RBF(Kernel, sk_RBF):

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 prior_bounds=None):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.prior_bounds = prior_bounds
        if length_scale_bounds == "dynamic":
            if prior_bounds is None:
                raise TypeError(
                    "Prior bounds are required for the RBF kernel "
                    "if its hyperparameter bounds are set to 'dynamic'. "
                    "You can either provide these bounds or set the "
                    "hyperparameter bounds to either numeric values or "
                    "'fixed'")
            elif not np.iterable(prior_bounds):
                raise TypeError("prior_bounds needs to be an iterable.")
            prior_bounds = np.asarray(prior_bounds)
            if not self.anisotropic:
                if prior_bounds.shape[0] > 1:
                    warnings.warn(
                        "The hyperparameter bounds of the isotropic RBF "
                        "kernel were set to 'dynamic' even though the "
                        "posterior distribution has more than one dimension. "
                        "The maximum length scale will be adapted to the "
                        "dimension with the largest prior. This may lead to "
                        "unintended behaviour.")
                self.max_length = (prior_bounds[:, 1] - prior_bounds[:, 0])
            else:
                self.max_length = (prior_bounds[:, 1] - prior_bounds[:, 0])
        else:
            self.max_length = None

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  self.max_length,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds,
            self.max_length)

    def gradient_x(self, x, X_train):
        # diff = (x - X) / length_scale
        # size = (n_train_samples, n_dimensions)
        x = np.asarray(x)
        X_train = np.asarray(X_train)

        length_scale = np.asarray(self.length_scale)
        diff = x - X_train
        diff /= length_scale

        # e = -exp(0.5 * \sum_{i=1}^d (diff ** 2))
        # size = (n_train_samples, 1)
        exp_diff_squared = np.sum(diff**2, axis=1)
        exp_diff_squared *= -0.5
        exp_diff_squared = np.exp(exp_diff_squared, exp_diff_squared)
        exp_diff_squared = np.expand_dims(exp_diff_squared, axis=1)
        exp_diff_squared *= -1

        # gradient = (e * diff) / length_scale
        gradient = exp_diff_squared * diff
        gradient /= length_scale
        return gradient


class Matern(Kernel, sk_Matern):

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 nu=1.5, prior_bounds=None):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.nu = nu
        self.prior_bounds = prior_bounds
        if length_scale_bounds == "dynamic":
            if prior_bounds is None:
                raise TypeError(
                    "Prior bounds are required for the Matern kernel "
                    "if its hyperparameter bounds are set to 'dynamic'. "
                    "You can either provide these bounds or set the "
                    "hyperparameter bounds to either numeric values or "
                    "'fixed'")
            elif not np.iterable(prior_bounds):
                raise TypeError("prior_bounds needs to be an iterable.")
            prior_bounds = np.asarray(prior_bounds)
            if not self.anisotropic:
                if prior_bounds.shape[0] > 1:
                    warnings.warn(
                        "The hyperparameter bounds of the isotropic Matern "
                        "kernel were set to 'dynamic' even though the "
                        "posterior distribution has more than one dimension. "
                        "The maximum length scale will be adapted to the "
                        "dimension with the largest prior. This may lead to "
                        "unintended behaviour.")
                self.max_length = (prior_bounds[:, 1] - prior_bounds[:, 0])
            else:
                self.max_length = (prior_bounds[:, 1] - prior_bounds[:, 0])
        else:
            self.max_length = None

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  self.max_length,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds,
            self.max_length)

    def gradient_x(self, x, X_train):
        x = np.asarray(x)
        X_train = np.asarray(X_train)
        length_scale = np.asarray(self.length_scale)

        # diff = (x - X_train) / length_scale
        # size = (n_train_samples, n_dimensions)
        diff = x - X_train
        diff /= length_scale

        # dist_sq = \sum_{i=1}^d (diff ^ 2)
        # dist = sqrt(dist_sq)
        # size = (n_train_samples,)
        dist_sq = np.sum(diff**2, axis=1)
        dist = np.sqrt(dist_sq)

        if self.nu == 0.5:
            # e = -np.exp(-dist) / dist
            # size = (n_train_samples, 1)
            scaled_exp_dist = -dist
            scaled_exp_dist = np.exp(scaled_exp_dist, scaled_exp_dist)
            scaled_exp_dist *= -1

            # grad = (e * diff) / length_scale
            # For all i in [0, D) if x_i equals y_i.
            # 1. e -> -1
            # 2. (x_i - y_i) / \sum_{j=1}^D (x_i - y_i)**2 approaches 1.
            # Hence the gradient when for all i in [0, D),
            # x_i equals y_i is -1 / length_scale[i].
            gradient = -np.ones((X_train.shape[0], x.shape[0]))
            mask = dist != 0.0
            scaled_exp_dist[mask] /= dist[mask]
            scaled_exp_dist = np.expand_dims(scaled_exp_dist, axis=1)
            gradient[mask] = scaled_exp_dist[mask] * diff[mask]
            gradient /= length_scale
            return gradient

        elif self.nu == 1.5:
            # grad(fg) = f'g + fg'
            # where f = 1 + sqrt(3) * euclidean((X - Y) / length_scale)
            # where g = exp(-sqrt(3) * euclidean((X - Y) / length_scale))
            sqrt_3_dist = sqrt(3) * dist
            f = np.expand_dims(1 + sqrt_3_dist, axis=1)

            # When all of x_i equals y_i, f equals 1.0, (1 - f) equals
            # zero, hence from below
            # f * g_grad + g * f_grad (where g_grad = -g * f_grad)
            # -f * g * f_grad + g * f_grad
            # g * f_grad * (1 - f) equals zero.
            # sqrt_3_by_dist can be set to any value since diff equals
            # zero for this corner case.
            sqrt_3_by_dist = np.zeros_like(dist)
            nzd = dist != 0.0
            sqrt_3_by_dist[nzd] = sqrt(3) / dist[nzd]
            dist_expand = np.expand_dims(sqrt_3_by_dist, axis=1)

            f_grad = diff / length_scale
            f_grad *= dist_expand

            sqrt_3_dist *= -1
            exp_sqrt_3_dist = np.exp(sqrt_3_dist, sqrt_3_dist)
            g = np.expand_dims(exp_sqrt_3_dist, axis=1)
            g_grad = -g * f_grad

            # f * g_grad + g * f_grad (where g_grad = -g * f_grad)
            f *= -1
            f += 1
            return g * f_grad * f

        elif self.nu == 2.5:
            # grad(fg) = f'g + fg'
            # where f = (1 + sqrt(5) * euclidean((X - Y) / length_scale) +
            #            5 / 3 * sqeuclidean((X - Y) / length_scale))
            # where g = exp(-sqrt(5) * euclidean((X - Y) / length_scale))
            sqrt_5_dist = sqrt(5) * dist
            f2 = (5.0 / 3.0) * dist_sq
            f2 += sqrt_5_dist
            f2 += 1
            f = np.expand_dims(f2, axis=1)

            # For i in [0, D) if x_i equals y_i
            # f = 1 and g = 1
            # Grad = f'g + fg' = f' + g'
            # f' = f_1' + f_2'
            # Also g' = -g * f1'
            # Grad = f'g - g * f1' * f
            # Grad = g * (f' - f1' * f)
            # Grad = f' - f1'
            # Grad = f2' which equals zero when x = y
            # Since for this corner case, diff equals zero,
            # dist can be set to anything.
            nzd_mask = dist != 0.0
            nzd = dist[nzd_mask]
            dist[nzd_mask] = np.reciprocal(nzd, nzd)

            dist *= sqrt(5)
            dist = np.expand_dims(dist, axis=1)
            diff /= length_scale
            f1_grad = dist * diff
            f2_grad = (10.0 / 3.0) * diff
            f_grad = f1_grad + f2_grad

            sqrt_5_dist *= -1
            g = np.exp(sqrt_5_dist, sqrt_5_dist)
            g = np.expand_dims(g, axis=1)
            g_grad = -g * f1_grad
            return f * g_grad + g * f_grad


class RationalQuadratic(Kernel, sk_RationalQuadratic):

    def __init__(self, length_scale=1.0, alpha=1.0,
                 length_scale_bounds=(1e-5, 1e5),
                 alpha_bounds=(1e-5, 1e5), prior_bounds=None):
        self.length_scale = length_scale
        self.alpha = alpha
        self.length_scale_bounds = length_scale_bounds
        self.alpha_bounds = alpha_bounds
        self.prior_bounds = prior_bounds
        if length_scale_bounds == "dynamic":
            if prior_bounds is None:
                raise TypeError(
                    "Prior bounds are required for the RQ kernel "
                    "if its hyperparameter bounds are set to 'dynamic'. "
                    "You can either provide these bounds or set the "
                    "hyperparameter bounds to either numeric values or "
                    "'fixed'")
            elif not np.iterable(prior_bounds):
                raise TypeError("prior_bounds needs to be an iterable.")
            prior_bounds = np.asarray(prior_bounds)
            if not self.anisotropic:
                if prior_bounds.shape[0] > 1:
                    warnings.warn(
                        "The hyperparameter bounds of the isotropic RQ "
                        "kernel were set to 'dynamic' even though the "
                        "posterior distribution has more than one dimension. "
                        "The maximum length scale will be adapted to the "
                        "dimension with the largest prior. This may lead to "
                        "unintended behaviour.")
                self.max_length = 2 * \
                    max(prior_bounds[:, 1] - prior_bounds[:, 0])
            else:
                self.max_length = 2 * (prior_bounds[:, 1] - prior_bounds[:, 0])
        else:
            self.max_length = None

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  self.max_length,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds,
            self.max_length)

    @property
    def hyperparameter_alpha(self):
        return Hyperparameter("alpha", "numeric", self.alpha_bounds)

    def gradient_x(self, x, X_train):
        x = np.asarray(x)
        X_train = np.asarray(X_train)
        alpha = self.alpha
        length_scale = self.length_scale

        # diff = (x - X_train) / length_scale
        # size = (n_train_samples, n_dimensions)
        diff = x - X_train
        diff /= length_scale

        # dist = -(1 + (\sum_{i=1}^d (diff^2) / (2 * alpha)))** (-alpha - 1)
        # size = (n_train_samples,)
        scaled_dist = np.sum(diff**2, axis=1)
        scaled_dist /= (2 * self.alpha)
        scaled_dist += 1
        scaled_dist **= (-alpha - 1)
        scaled_dist *= -1

        scaled_dist = np.expand_dims(scaled_dist, axis=1)
        diff_by_ls = diff / length_scale
        return scaled_dist * diff_by_ls


class ExpSineSquared(Kernel, sk_ExpSineSquared):

    def __init__(self, length_scale=1.0, periodicity=1.0,
                 length_scale_bounds=(1e-5, 1e5),
                 periodicity_bounds=(1e-5, 1e5), prior_bounds=None):
        self.length_scale = length_scale
        self.periodicity = periodicity
        self.length_scale_bounds = length_scale_bounds
        self.periodicity_bounds = periodicity_bounds
        self.prior_bounds = prior_bounds
        if length_scale_bounds == "dynamic":
            if prior_bounds is None:
                raise TypeError(
                    "Prior bounds are required for the RQ kernel "
                    "if its hyperparameter bounds are set to 'dynamic'. "
                    "You can either provide these bounds or set the "
                    "hyperparameter bounds to either numeric values or "
                    "'fixed'")
            elif not np.iterable(prior_bounds):
                raise TypeError("prior_bounds needs to be an iterable.")
            prior_bounds = np.asarray(prior_bounds)
            if not self.anisotropic:
                if prior_bounds.shape[0] > 1:
                    warnings.warn(
                        "The hyperparameter bounds of the isotropic RQ "
                        "kernel were set to 'dynamic' even though the "
                        "posterior distribution has more than one dimension. "
                        "The maximum length scale will be adapted to the "
                        "dimension with the largest prior. This may lead to "
                        "unintended behaviour.")
                self.max_length = 2 * \
                    max(prior_bounds[:, 1] - prior_bounds[:, 0])
            else:
                self.max_length = 2 * (prior_bounds[:, 1] - prior_bounds[:, 0])
        else:
            self.max_length = None

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale),
                                  max_length=self.max_length)
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds,
            max_length=self.max_length)

    @property
    def hyperparameter_periodicity(self):
        return Hyperparameter(
            "periodicity", "numeric", self.periodicity_bounds)

    def gradient_x(self, x, X_train):
        x = np.asarray(x)
        X_train = np.asarray(X_train)
        length_scale = self.length_scale
        periodicity = self.periodicity

        diff = x - X_train
        sq_dist = np.sum(diff**2, axis=1)
        dist = np.sqrt(sq_dist)

        pi_by_period = dist * (np.pi / periodicity)
        sine = np.sin(pi_by_period) / length_scale
        sine_squared = -2 * sine**2
        exp_sine_squared = np.exp(sine_squared)

        grad_wrt_exp = -2 * np.sin(2 * pi_by_period) / length_scale**2

        # When x_i -> y_i for all i in [0, D), the gradient becomes
        # zero. See https://github.com/MechCoder/Notebooks/blob/master/ExpSineSquared%20Kernel%20gradient%20computation.ipynb
        # for a detailed math explanation
        # grad_wrt_theta can be anything since diff is zero
        # for this corner case, hence we set to zero.
        grad_wrt_theta = np.zeros_like(dist)
        nzd = dist != 0.0
        grad_wrt_theta[nzd] = np.pi / (periodicity * dist[nzd])
        return np.expand_dims(
            grad_wrt_theta * exp_sine_squared * grad_wrt_exp, axis=1) * diff


class ConstantKernel(Kernel, sk_ConstantKernel):

    @property
    def hyperparameter_constant_value(self):
        return Hyperparameter(
            "constant_value", "numeric", self.constant_value_bounds, None)

    def gradient_x(self, x, X_train):
        return np.zeros_like(X_train)


class WhiteKernel(Kernel, sk_WhiteKernel):

    @property
    def hyperparameter_noise_level(self):
        return Hyperparameter(
            "noise_level", "numeric", self.noise_level_bounds, None)

    def gradient_x(self, x, X_train):
        return np.zeros_like(X_train)


class KernelOperator:
    """
    Updated to accomodate the new kernel hyperparameter definition.
    """

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = [Hyperparameter("k1__" + hyperparameter.name,
                            hyperparameter.value_type,
                            hyperparameter.bounds, hyperparameter.max_length,
                            hyperparameter.n_elements)
             for hyperparameter in self.k1.hyperparameters]

        for hyperparameter in self.k2.hyperparameters:
            r.append(Hyperparameter("k2__" + hyperparameter.name,
                                    hyperparameter.value_type,
                                    hyperparameter.bounds,
                                    hyperparameter.max_length,
                                    hyperparameter.n_elements))
        return r


class Exponentiation(Kernel, sk_Exponentiation):

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(Hyperparameter("kernel__" + hyperparameter.name,
                                    hyperparameter.value_type,
                                    hyperparameter.bounds,
                                    hyperparameter.max_length,
                                    hyperparameter.n_elements))
        return r

    def gradient_x(self, x, X_train):
        x = np.asarray(x)
        X_train = np.asarray(X_train)
        expo = self.exponent
        kernel = self.kernel

        K = np.expand_dims(
            kernel(np.expand_dims(x, axis=0), X_train)[0], axis=1)
        return expo * K ** (expo - 1) * kernel.gradient_x(x, X_train)


class Sum(KernelOperator, Kernel, sk_Sum):

    @property
    def hyperparameters(self):
        return super().hyperparameters

    def gradient_x(self, x, X_train):
        return self.k1.gradient_x(x, X_train) + self.k2.gradient_x(x, X_train)


class Product(KernelOperator, Kernel, sk_Product):

    @property
    def hyperparameters(self):
        return super().hyperparameters

    def gradient_x(self, x, X_train):
        x = np.asarray(x)
        x = np.expand_dims(x, axis=0)
        X_train = np.asarray(X_train)
        f_ggrad = (
            np.expand_dims(self.k1(x, X_train)[0], axis=1) *
            self.k2.gradient_x(x, X_train)
        )
        fgrad_g = (
            np.expand_dims(self.k2(x, X_train)[0], axis=1) *
            self.k1.gradient_x(x, X_train)
        )
        return f_ggrad + fgrad_g


class DotProduct(Kernel, sk_DotProduct):

    @property
    def hyperparameter_sigma_0(self):
        return Hyperparameter("sigma_0", "numeric", self.sigma_0_bounds)

    def gradient_x(self, x, X_train):
        return np.asarray(X_train)
