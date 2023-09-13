"""
Base Class
==========
All acquisition functions are derived from this class. If you want to define
your own acquisition functions it needs to inherit from this class. A tutorial
on how to define such a class is given in :class:`.AcquisitionFunction`

.. autosummary::

     AcquisitionFunction

Inbuilt Acquisition Functions
=============================

All inbuilt Acquisition Functions can evaluate the gradient in addition
to the value of the acquisition function at point X.

The recommended acquisition that we derived in order to efficiently sample the parameter
space is called ``LogExp``:

.. autosummary::
     LogExp

Furthermore there are more inbuilt acquisition functions (building blocks) which should
offer a great deal of flexibility. If you want to define your own acquisition function
it needs to inherit from the parent class :class:`AcquisitionFunction`.

.. autosummary::

     ConstantAcqFunc
     Mu
     Std
     ExponentialMu
     ExponentialStd
     ExpectedImprovement

Additional things
=================

The things listed here are tools and similar things which in normal operation
should not be needed.

.. autosummary::

     is_acquisition_function
     Hyperparameter
     AcquisitionFunctionOperator
     Sum
     Product
     Exponentiation

"""

import sys
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Iterable
from inspect import signature, getmembers
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.base import clone


# UNUSED
def _safe_log_expm1(x):
    """
    Numerically safer ``log(exp(x) - 1)``.
    """
    mask = x < 1
    ret = np.empty_like(x)
    ret[mask] = np.log(np.expm1(x[mask]))
    ret[~mask] = x[~mask] + np.log1p(-np.exp(-x[~mask]))
    return ret


def builtin_names():
    """
    Lists all names of all built-in acquisition functions criteria.
    """
    list_names = [name for name, obj in getmembers(sys.modules[__name__])
                  if (issubclass(obj.__class__, AcquisitionFunction.__class__) and
                      obj is not AcquisitionFunction)]
    return list_names


class AcquisitionFunction(metaclass=ABCMeta):
    """Base class for all Acquisition Functions (AF's). All acquisition
    functions are derived from this class.

    Currently several acquisition functions are supported which should be
    versatile enough for most tasks.
    If however one wants to specify a custom acquisition function
    it should be a class which inherits from this abstract class.
    This class needs to be of the format::

        from Acquisition_functions import Acquisition_function
        Class custom_acq_func(Acquisition_Function):
            def __init__(self, param_1, ..., fixed=..., dimension=...):
                # * 'hyperparam_i': The hyperparameters of the custom
                #   acquisition function.
                # * 'fixed': whether the hyperparameters of the acquisition
                #   function are to be kept fixed.
                # * 'dimension': the dimensionality of the target function,
                #   which can be used to automatically adapt hyperparameters
                #   function are to be kept fixed.
                # * 'hasgradient': Whether the acquisition function can return
                #   a gradient. Furthermore the bool hasgradient needs to be
                #   specified to 'True' if the acquisition function can return
                #   gradient(s) or 'False' otherwise.
                self.param_1 = param_1
                ...
                self.fixed=fixed
                self.hasgradient = True/False

            @property
            def hyperparameter_param_1(self):
                # Returns the type of hyperparameter and whether it is
                # fixed or not. This method needs to exist for every hyper-
                # parameter.
                return Hyperparameter(
                    "param_1", "numeric", fixed=self.fixed)

                return Hyperparameter(
                    "param_1", "numeric", fixed=self.fixed)

            def __call__(self, X, gp, eval_gradient=False):
                # * 'X': The value(s) at which the acquisition function is
                #    evaluated
                # * 'GP': The surrogate GP model which shall be used.
                # * 'eval_gradient': Whether the gradient shall be given or
                #   not. Only required if 'self.hasgradient' is true.
                ....
                # Returned are the value(s) of the acquisition function at
                # point(s) X and optionally their gradient(s)

    Once the Acquisition function is defined in this way it can be used with
    the same operators as the inbuilt acquisition functions.

     .. note::

        If one of the
        operands of a composite acquisiton function does not return a gradient
        the same applies for all operands. Furthermore some optimizers require
        gradients, which cannot be used in this case.

    """

    def get_params(self, deep=True):
        """Get hyperparameters of this Acquisition function.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict()

        # introspect the constructor arguments to find the model parameters
        # to represent
        cls = self.__class__
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        init_sign = signature(init)
        args, varargs = [], []
        for parameter in init_sign.parameters.values():
            if parameter.kind != parameter.VAR_KEYWORD and \
               parameter.name != 'self':
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)

        if len(varargs) != 0:
            raise RuntimeError("GPry acquisition functions should always "
                               "specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls, ))
        for arg in args:
            try:
                value = getattr(self, arg)
            except AttributeError:
                warnings.warn('From version 0.24, get_params will raise an '
                              'AttributeError if a parameter cannot be '
                              'retrieved as an instance attribute. Previously '
                              'it would return None.',
                              FutureWarning)
                value = None
            params[arg] = value
        return params

    def set_params(self, **params):
        """Set the parameters of this acquisition function.
        The method works on simple AF's as well as on nested AF's.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Any number of parameters which shall be set. Should be of the form
            ``{"parameter_1_name" : parameter_1_value,
            "parameter_2_name" : parameter_2_value, ...}``

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for AF %s. '
                                     'Check the list of available parameters '
                                     'with `acquisition_function.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for AF %s. '
                                     'Check the list of available parameters '
                                     'with `acquisition_function.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def clone_with_theta(self, theta):
        """Returns a clone of self with given hyperparameters theta.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The hyperparameters
        """
        cloned = clone(self)
        cloned.theta = theta
        return cloned

    def check_X(self, X):
        """Internal method to check the dimensionality of any input X
        provided to an AF when called. Checks the correct type and turns X into
        a 2d array if a 1d array is provided.

        .. warning::

            This method only checks for the correct type of an input,
            inappropriate values might still cause problems.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_dims) or (ndims,)
            The input array for any X value passed to an acquisition function

        Returns
        -------
        X_new: ndarray of shape (n_samples, n_dims)
            The reshaped array of input data X
        """
        if not isinstance(X, np.ndarray):
            raise ValueError(
                "Expected a numpy array for X, instead got %s" % X)

        if X.ndim == 1:
            return X.reshape(1, -1)
        else:
            return X

    @property
    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the acquisition function."""
        return self.theta.shape[0]

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        r = [getattr(self, attr) for attr in dir(self)
             if attr.startswith("hyperparameter_")]
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        acquisition function's hyperparameters as this representation of the
        search space is more amenable for hyperparameter search, as
        hyperparameters likelength-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the acquisition
            function.
        """
        theta = []
        params = self.get_params()
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                theta.append(params[hyperparameter.name])
        if len(theta) > 0:
            return np.log(np.hstack(theta))
        else:
            return np.array([])

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the acquisition
            function.
        """
        params = self.get_params()
        i = 0
        for hyperparameter in self.hyperparameters:
            if hyperparameter.fixed:
                continue
            if hyperparameter.n_elements > 1:
                # vector-valued parameter
                params[hyperparameter.name] = np.exp(
                    theta[i:i + hyperparameter.n_elements])
                i += hyperparameter.n_elements
            else:
                params[hyperparameter.name] = np.exp(theta[i])
                i += 1

        if i != len(theta):
            raise ValueError("theta has not the correct number of entries."
                             " Should be %d; given are %d"
                             % (i, len(theta)))
        self.set_params(**params)

    @property
    def hasgradient(self):
        """Specifies whether a certain acquisition function can return
        gradients or not.
        """
        return self._hasgradient

    @hasgradient.setter
    def hasgradient(self, hasgradient):
        if isinstance(hasgradient, bool):
            self._hasgradient = hasgradient
        else:
            raise TypeError("hasgradient needs to be"
                            "bool, not %s" % hasgradient)

    @abstractmethod
    def __call__(self, X, gp, eval_gradient=False):
        """Evaluate the acquisition function."""

    def __add__(self, b):
        if not isinstance(b, AcquisitionFunction):
            return Sum(self, ConstantAcqFunc(b))
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, AcquisitionFunction):
            return Sum(ConstantAcqFunc(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, AcquisitionFunction):
            return Product(self, ConstantAcqFunc(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, AcquisitionFunction):
            return Product(ConstantAcqFunc(b), self)
        return Product(b, self)

    def __pow__(self, b):
        return Exponentiation(self, b)

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        params_a = self.get_params()
        params_b = b.get_params()
        for key in set(list(params_a.keys()) + list(params_b.keys())):
            if np.any(params_a.get(key, None) != params_b.get(key, None)):
                return False
        return True

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map("{0:.3g}".format, self.theta)))


class ConstantAcqFunc(AcquisitionFunction):
    r"""Constant Acquisition function.

    Can be used as part of a product-Composition where it scales the magnitude
    of the other factor or as part of a sum.

    .. math::
        A_f(X) = constant\_value \;\forall\; X

    Parameters
    ----------
    constant_value : float, default=1.0
        The constant value :
        A_f(X) = constant_value

    fixed: bool, default=False,
        whether the constant value shall be fixed or not.
    """

    def __init__(self, constant_value=1.0, fixed=False, dimension=None):
        self.constant_value = constant_value
        self.fixed = fixed
        self.hasgradient = True

    @property
    def hyperparameter_constant_value(self):
        return Hyperparameter(
            "constant_value", "numeric", fixed=self.fixed)

    def __call__(self, X, gp, eval_gradient=False):
        """Return the Value of the AF at x (``A_f(X, gp)``) and optionally its
        gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            X-Value at which the Acquisition function shall be evaluated

        gp : SKLearn GaussianProcessRegressor
            The GPRegressor (surrogate model) from which to evaluate
            GP(X) and optionally X_train and Y_train.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to X is calculated.

        Returns
        -------
        A_f : array of shape (n_samples_X)
            The value of the acquisition function at point(s) X

        A_f_gradient : array of shape (n_samples_X, n_dim)
            The gradient of the Acquisition function with respect to X.
            Only returned when eval_gradient is True.
        """
        X = self.check_X(X)

        if not np.iterable(X):
            X = np.array([X])
        A_f = np.ones(X.shape[0]) * self.constant_value
        if eval_gradient:
            return A_f, np.zeros(X.shape)
        else:
            return A_f

    def __repr__(self):
        return "{0:.3g}**2".format(np.sqrt(self.constant_value))


# UNUSED
class Mu(AcquisitionFunction):
    r""":math:`\mu(X)` of the surrogate model.

    .. math::
        A_f(X) = a\cdot\mu(X)

    Parameters
    ----------
    a : float, default=1.0
        The value with which :math:`\mu` is multiplied .

    fixed: bool, default=False,
        whether the constant value shall be fixed or not.
    """

    def __init__(self, a=1.0, fixed=False, dimension=None):
        self.a = a
        self.fixed = fixed
        self.hasgradient = True

    @property
    def hyperparameter_a(self):
        return Hyperparameter(
            "a", "numeric", fixed=self.fixed)

    def __call__(self, X, gp, eval_gradient=False):
        """Return the Value of the AF at x (``A_f(X, gp)``) and optionally
        its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            X-Value at which the Acquisition function shall be evaluated

        gp : SKLearn GaussianProcessRegressor
            The GPRegressor (surrogate model) from which to evaluate
            GP(X) and optionally X_train and Y_train.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to X is calculated.

        Returns
        -------
        A_f : array of shape (n_samples_X)
            The value of the acquisition function at point(s) X

        A_f_gradient : array of shape (n_samples_X, n_dim)
            The gradient of the Acquisition function with respect to X.
            Only returned when eval_gradient is True.
        """
        X = self.check_X(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if eval_gradient:
                mu, _, mu_grad = gp.predict(
                    X, return_std=True, return_mean_grad=True,
                    return_std_grad=False)

            else:
                mu, std = gp.predict(X, return_std=True)

        if eval_gradient:
            return mu, mu_grad
        else:
            return mu

    def __repr__(self):
        return "{0:.3g}**2".format(np.sqrt(self.a))


# UNUSED
class ExponentialMu(AcquisitionFunction):
    r""":math:`\exp[\mu(X)]` of the surrogate model.

    .. math::
        A_f(X) = \exp(a\cdot\mu(X))

    Parameters
    ----------
    a : float, default=1.0
        The value with which :math:`\mu` is multiplied before exponentiating.

    fixed: bool, default=False,
        whether the constant value shall be fixed or not.
    """

    def __init__(self, a=1.0, fixed=False, dimension=None):
        self.a = a
        self.fixed = fixed
        self.hasgradient = True

    @property
    def hyperparameter_a(self):
        return Hyperparameter(
            "a", "numeric", fixed=self.fixed)

    def __call__(self, X, gp, eval_gradient=False):
        """Return the Value of the AF at x (``A_f(X, gp)``) and optionally its
        gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            X-Value at which the Acquisition function shall be evaluated

        gp : SKLearn GaussianProcessRegressor
            The GPRegressor (surrogate model) from which to evaluate
            GP(X) and optionally X_train and Y_train.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to X is calculated.

        Returns
        -------
        A_f : array of shape (n_samples_X)
            The value of the acquisition function at point(s) X

        A_f_gradient : array of shape (n_samples_X, n_dim)
            The gradient of the Acquisition function with respect to X.
            Only returned when eval_gradient is True.
        """
        X = self.check_X(X)

        if not np.iterable(X):
            X = np.array([X])
        mu, mu_grad = gp.predict(X, return_std=False,
                                 return_cov=False,
                                 return_mean_grad=True,
                                 return_std_grad=False)
        A_f = np.exp(self.a * mu)
        if eval_gradient:
            A_f_grad = self.a * mu_grad * np.exp(self.a * mu)
            return A_f, A_f_grad
        else:
            return A_f

    def __repr__(self):
        return "{0:.3g}**2".format(np.sqrt(self.a))


# UNUSED
class Std(AcquisitionFunction):
    r""":math:`\sigma(X)` of the surrogate model.

    .. math::
        A_f(X) = a\cdot\sigma(X)

    Parameters
    ----------
    a : float, default=1.0
        The value with which :math:`\sigma` is multiplied before
        exponentiating.

    fixed: bool, default=False,
        whether the constant value shall be fixed or not.
    """

    def __init__(self, a=1.0, fixed=False, dimension=None):
        self.a = a
        self.fixed = fixed
        self.hasgradient = True

    @property
    def hyperparameter_a(self):
        return Hyperparameter(
            "a", "numeric", fixed=self.fixed)

    def __call__(self, X, gp, eval_gradient=False):
        """Return the Value of the AF at x (``A_f(X, gp)``) and optionally
        its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            X-Value at which the Acquisition function shall be evaluated

        gp : SKLearn GaussianProcessRegressor
            The GPRegressor (surrogate model) from which to evaluate
            GP(X) and optionally X_train and Y_train.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to X is calculated.

        Returns
        -------
        A_f : array of shape (n_samples_X)
            The value of the acquisition function at point(s) X

        A_f_gradient : array of shape (n_samples_X, n_dim)
            The gradient of the Acquisition function with respect to X.
            Only returned when eval_gradient is True.
        """
        X = self.check_X(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if eval_gradient:
                _, std, _, std_grad = gp.predict(
                    X, return_std=True, return_mean_grad=True,
                    return_std_grad=True)

            else:
                mu, std = gp.predict(X, return_std=True)

        if eval_gradient:
            return std, std_grad
        else:
            return std

    def __repr__(self):
        return "{0:.3f}".format(self.a)


# UNUSED
class ExponentialStd(AcquisitionFunction):
    r""":math:`\exp[\sigma(X)]` of the surrogate model.

    .. math::
        A_f(X) = \exp(a\cdot\sigma(X))

    Parameters
    ----------
    a : float, default=1.0
        The value with which :math:`\sigma` is multiplied before
        exponentiating.

    fixed: bool, default=False,
        whether the constant value shall be fixed or not.
    """

    def __init__(self, a=1.0, fixed=False, dimension=None):
        self.a = a
        self.fixed = fixed
        self.hasgradient = True

    @property
    def hyperparameter_a(self):
        return Hyperparameter(
            "a", "numeric", fixed=self.fixed)

    def __call__(self, X, gp, eval_gradient=False):
        """Return the Value of the AF at x (``A_f(X, gp)``) and optionally its
        gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            X-Value at which the Acquisition function shall be evaluated

        gp : SKLearn GaussianProcessRegressor
            The GPRegressor (surrogate model) from which to evaluate
            GP(X) and optionally X_train and Y_train.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to X is calculated.

        Returns
        -------
        A_f : array of shape (n_samples_X)
            The value of the acquisition function at point(s) X

        A_f_gradient : array of shape (n_samples_X, n_dim)
            The gradient of the Acquisition function with respect to X.
            Only returned when eval_gradient is True.
        """
        X = self.check_X(X)

        if not np.iterable(X):
            X = np.array([X])
        _, std, _, std_grad = gp.predict(X,
                                         return_std=True,
                                         return_cov=False,
                                         return_mean_grad=True,
                                         return_std_grad=True)
        A_f = np.exp(self.a * std)
        if eval_gradient:
            A_f_grad = self.a * std_grad * np.exp(self.a * std)
            return A_f, A_f_grad
        else:
            return A_f

    def __repr__(self):
        return "{0:.3f}".format(self.a)


# UNUSED
class ExpectedImprovement(AcquisitionFunction):
    r"""Computes the (negative) Expected improvement function.

    The conditional probability `P(y=f(x) | x)` form a gaussian with a certain
    mean and standard deviation approximated by the model.

    The EI condition is derived by computing :math:`E[u(f(x))]`
    where :math:`u(f(x)) = 0`, if :math:`f(x) > y_{\mathrm{opt}}`
    and :math:`u(f(x)) = y_{\mathrm{opt}} - f(x)`,
    if :math:`f(x) < y_{\mathrm{opt}}`.

    This solves one of the issues of the PI condition by giving a reward
    proportional to the amount of improvement got.

    Parameters
    ----------
    xi : float, default=0.01
        Controls how much improvement one wants over the previous best
        values. Useful only when ``method`` is set to "EI"

    fixed: bool, default=False,
        whether the constant value shall be fixed or not.
    """

    def __init__(self, xi=0.01, fixed=False, dimension=None):
        self.xi = xi
        self.fixed = fixed
        self.hasgradient = True

    @property
    def hyperparameter_xi(self):
        return Hyperparameter(
            "xi", "numeric", fixed=self.fixed)

    def __call__(self, X, gp, eval_gradient=False):
        """Return the Value of the AF at x (``A_f(X, gp)``) and optionally its
        gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            X-Value at which the Acquisition function shall be evaluated

        gp : SKLearn GaussianProcessRegressor
            The GPRegressor (surrogate model) from which to evaluate
            GP(X) and optionally X_train and Y_train.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to X is calculated.

        Returns
        -------
        A_f : array of shape (n_samples_X)
            The value of the acquisition function at point(s) X

        A_f_gradient : array of shape (n_samples_X, n_dim)
            The gradient of the Acquisition function with respect to X.
            Only returned when eval_gradient is True.
        """
        X = self.check_X(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if eval_gradient:
                mu, std, mu_grad, std_grad = gp.predict(
                    X, return_std=True, return_mean_grad=True,
                    return_std_grad=True)

            else:
                mu, std = gp.predict(X, return_std=True)

            y_opt = gp.y_max

        values = np.zeros_like(mu)
        mask = std > 0
        improve = mu[mask] - y_opt - self.xi
        scaled = improve / std[mask]
        cdf = norm.cdf(scaled)
        pdf = norm.pdf(scaled)
        exploit = improve * cdf
        explore = std[mask] * pdf
        values[mask] = exploit + explore

        if eval_gradient:
            if not np.all(mask):
                return values, np.zeros_like(std_grad)

            # Substitute (y_opt - xi - mu) / sigma = t and apply chain rule.
            # improve_grad is the gradient of t wrt x.
            improve_grad = -mu_grad * std - std_grad * improve
            improve_grad /= std ** 2
            cdf_grad = improve_grad * pdf
            pdf_grad = -improve * cdf_grad
            exploit_grad = -mu_grad * cdf - pdf_grad
            explore_grad = std_grad * pdf + pdf_grad

            grad = exploit_grad + explore_grad
            return values, grad
        return values

    def __repr__(self):
        return "{0:.3f}".format(self.xi)


class BaseLogExp(AcquisitionFunction, metaclass=ABCMeta):
    r"""Acquisition function which is designed to efficiently sample
    log-probability distributions. This is achieved by transforming
    :math:`\tilde{\mu}\cdot\tilde{\sigma}` (of the true, non-logarithmic
    probability distribution) to logarithmic space.

    Parameters
    ----------
    zeta : float, default=1
        Controls the exploration-exploitation tradeoff parameter. The value
        of :math:`\zeta` should not exceed 1 under normal circumstances as a
        value <1 accounts for the fact that the GP's estimate for
        :math:`\mu` is not correct at the beginning. A good suggestion
        for setting zeta which is inspired by simulated annealing is

        .. math::

            \zeta = \exp(-N_0/N)

        where :math:`N_0\geq 0` is a "decay constant" and :math:`N`
        the number of training points
        in the GP.

    sigma_n : float, default=None
        The (constant) noise level of the data. If set to ``None`` the
        square-root of alpha of the training data (or the square root of the
        mean of alpha if alpha is an array) will be used.

    fixed: bool, default=False,
        whether zeta and sigma_n shall be fixed or not.

    dimension: double, default=None
        the dimension of the parameter space used for auto-scaling the zeta

    zeta_scaling: double, default=1.1
        the scaling power of the zeta with dimension, if auto-scaled
    """

    def __init__(self, zeta=None, sigma_n=None, fixed=False, dimension=None,
                 zeta_scaling=0.85, linear=True):
        if zeta is None:
            if dimension is None:
                raise ValueError("We need the dimensionality of the problem to "
                                 "guess an appropriate zeta value.")
            self.zeta = self.auto_zeta(dimension, scaling=zeta_scaling)
        else:
            self.zeta = zeta
        self.sigma_n = sigma_n
        self.fixed = fixed
        self.hasgradient = True

    @abstractmethod
    def f(mu, std, zeta):
        return

    @property
    def hyperparameter_zeta(self):
        return Hyperparameter(
            "zeta", "numeric", fixed=self.fixed)

    @property
    def hyperparameter_sigma_n(self):
        return Hyperparameter(
            "sigma_n", "numeric", fixed=self.fixed)

    def auto_zeta(self, dimension, scaling=1.1):
        return dimension**(-scaling)

    def __call__(self, X, gp, eval_gradient=False):
        """Return the Value of the AF at x (``A_f(X, gp)``) and optionally
        its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            X-Value at which the Acquisition function shall be evaluated

        gp : SKLearn GaussianProcessRegressor
            The GPRegressor (surrogate model) from which to evaluate
            GP(X) and optionally X_train and Y_train.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to X is calculated.

        Returns
        -------
        A_f : array of shape (n_samples_X)
            The value of the acquisition function at point(s) X

        A_f_gradient : array of shape (n_samples_X, n_dim)
            The gradient of the Acquisition function with respect to X.
            Only returned when eval_gradient is True.
        """
        X = self.check_X(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if eval_gradient:
                mu, std, mu_grad, std_grad = gp.predict(
                    X, return_std=True, return_mean_grad=True,
                    return_std_grad=True)

            else:
                mu, std = gp.predict(X, return_std=True)

        if self.sigma_n is None:
            sigma_n = gp.noise_level
            if isinstance(sigma_n, Iterable):
                noise_var = np.mean(sigma_n)
            else:
                noise_var = sigma_n
        else:
            noise_var = self.sigma_n
        zeta = self.zeta
        var = std**2 - noise_var**2.
        mask = (var > 0) & np.isfinite(mu)
        values = np.zeros_like(std)
        baseline = gp.y_max
        # Alternative option, but found not to work extremely well
        # baseline = gp.preprocessing_y.inverse_transform([0])[0]
        if np.any(mask):
            values[mask] = self.f(mu[mask], std[mask], baseline, noise_var, zeta)
        if np.any(~mask):
            values[~mask] = - np.inf
        if eval_gradient:
            if np.array(std_grad).ndim > 1:
                grad = np.zeros_like(std_grad)
                if np.any(mask):
                    grad[mask] = np.array(std_grad)[mask] / \
                        (std[mask] - sigma_n) + 2 * zeta * np.array(mu_grad)[mask]
                if np.any(~mask):
                    grad[~mask] = np.ones_like(std_grad[~mask]) * np.inf
            else:
                std = std[0]
                if std > sigma_n:
                    grad = std_grad / (std - sigma_n) + 2 * zeta * mu_grad
                else:
                    grad = np.ones_like(std_grad) * np.inf
            return values, grad
        else:
            return values

    def __repr__(self):
        return str(self.__class__) + "with zeta={0:.3f}".format(self.zeta)


class LogExp(BaseLogExp):
    r"""Acquisition function which is designed to efficiently sample
    log-probability distributions.
    This is achieved by transforming
    :math:`\tilde{\mu}\cdot\tilde{\sigma}` (of the true, non-logarithmic
    probability distribution) to logarithmic space which yields

    .. math::

        A_{\mathrm{LE}}(X) = \exp(2\zeta\cdot\mu(X))\cdot (\sigma(X)-\sigma_n)

    For numerical convenience we take the log of this expression which yields:

    .. math::

        \log(A_{\mathrm{LE}})(X) = 2\zeta\cdot\mu(X) + \log(\sigma(X)-\sigma_n)

    .. note::
        :math:`\mu(x)` and :math:`\sigma(X)` are the mean and sigma of the
        GP regressor which follows the **log**-probability distribution.

    Parameters
    ----------
    zeta : float, default=1
        Controls the exploration-exploitation tradeoff parameter. The value
        of :math:`\zeta` should not exceed 1 under normal circumstances as a
        value <1 accounts for the fact that the GP's estimate for
        :math:`\mu` is not correct at the beginning. A good suggestion
        for setting zeta which is inspired by simulated annealing is

        .. math::

            \zeta = \exp(-N_0/N)

        where :math:`N_0\geq 0` is a "decay constant" and :math:`N`
        the number of training points
        in the GP.

    sigma_n : float, default=None
        The (constant) noise level of the data. If set to ``None`` the
        square-root of alpha of the training data (or the square root of the
        mean of alpha if alpha is an array) will be used.

    fixed: bool, default=False,
        whether zeta and sigma_n shall be fixed or not.

    dimension: double, default=None
        the dimension of the parameter space used for auto-scaling the zeta

    zeta_scaling: double, default=1.1
        the scaling power of the zeta with dimension, if auto-scaled
    """

    @staticmethod
    def f(mu, std, baseline, noise_level, zeta):
        """Linearized exponentiated log-error bar."""
        return 2 * zeta * (mu - baseline) + np.log(np.sqrt(np.clip(std**2.-noise_level**2., 0., None)))


# UNUSED
# TODO: gradient assumed by parent class is not correct for this acquisition function
class NonlinearLogExp(BaseLogExp):
    r"""
    .. warning::
        The gradients for this acquisition function are not yet implemented correctly.
        Use with caution!

    An alternative approach which keeps both scales exponentiated:

    .. math::

        A_{\mathrm{LE}}(X) = \exp(2\zeta\cdot\mu(X))\cdot \exp(\sigma(X)-\sigma_n)

    Again we take the log of this.

    Parameters
    ----------
    zeta : float, default=1
        Controls the exploration-exploitation tradeoff parameter. The value
        of :math:`\zeta` should not exceed 1 under normal circumstances as a
        value <1 accounts for the fact that the GP's estimate for
        :math:`\mu` is not correct at the beginning. A good suggestion
        for setting zeta which is inspired by simulated annealing is

        .. math::

            \zeta = \exp(-N_0/N)

        where :math:`N_0\geq 0` is a "decay constant" and :math:`N`
        the number of training points
        in the GP.

    sigma_n : float, default=None
        The (constant) noise level of the data. If set to ``None`` the
        square-root of alpha of the training data (or the square root of the
        mean of alpha if alpha is an array) will be used.

    fixed: bool, default=False,
        whether zeta and sigma_n shall be fixed or not.

    dimension: double, default=None
        the dimension of the parameter space used for auto-scaling the zeta

    zeta_scaling: double, default=1.1
        the scaling power of the zeta with dimension, if auto-scaled
    """

    @staticmethod
    def f(mu, std, baseline, noise_level, zeta):
        """Exponentiated log-error bar"""
        return 2 * zeta * (mu - baseline) + _safe_log_expm1(np.sqrt(np.clip(std**2.-noise_level**2., 0., None)))


# Function for determining whether an object is an acquisition function
def is_acquisition_function(acq_func):
    """Determines if a given object is an acquisition function
    or not.

    Parameters
    ----------
    acq_func: Any
        The object which shall be examined

    Returns
    -------
    is_acquisition_function: bool
        whether the specified object is an acquisition
        function.
    """
    return isinstance(acq_func, AcquisitionFunction)


class Hyperparameter(namedtuple('Hyperparameter',
                                ('name', 'value_type',
                                 'n_elements', 'fixed'))):
    """An acquisition function hyperparameter's specification in form of a
    namedtuple. This formalism is copied from the ``kernel`` module of
    Scikit-Learn.

    .. note::
        The current code does not support optimization of any hyperparameters
        of the acquisition functions. This might be added in the future
        (which is why the ``fixed`` parameter exists).

    Attributes
    ----------
    name : str
        The name of the hyperparameter. Unlike in the kernels for the
        GPRegressor the hyperparameters of the Acquisition functions do
        not have bounds.
    value_type : str
        The type of the hyperparameter. Currently, only ``'numeric'``
        hyperparameters are supported.
    n_elements : int, default=1
        The number of elements of the hyperparameter value. Defaults to 1,
        which corresponds to a scalar hyperparameter. n_elements > 1
        corresponds to a hyperparameter which is vector-valued,
        such as, e.g., anisotropic length-scales.
    fixed : bool, default=False
        Whether the value of this hyperparameter is fixed, i.e., cannot be
        changed during hyperparameter tuning.
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

    def __new__(cls, name, value_type, n_elements=1, fixed=False):

        return super(Hyperparameter, cls).__new__(
            cls, name, value_type, n_elements, fixed)

    # This is mainly a testing utility to check that two hyperparameters
    # are equal.
    def __eq__(self, other):
        return (self.name == other.name and
                self.value_type == other.value_type and
                self.n_elements == other.n_elements and
                self.fixed == other.fixed)


class AcquisitionFunctionOperator(AcquisitionFunction):
    """Base class for all AF operators."""

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2
        self.hasgradient = k1.hasgradient and k2.hasgradient

    def get_params(self, deep=True):
        """Get parameters of this acquisition function.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict(k1=self.k1, k2=self.k2)
        if deep:
            deep_items = self.k1.get_params().items()
            params.update(('k1__' + k, val) for k, val in deep_items)
            deep_items = self.k2.get_params().items()
            params.update(('k2__' + k, val) for k, val in deep_items)

        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = [Hyperparameter("k1__" + hyperparameter.name,
                            hyperparameter.value_type,
                            hyperparameter.n_elements)
             for hyperparameter in self.k1.hyperparameters]

        for hyperparameter in self.k2.hyperparameters:
            r.append(Hyperparameter("k2__" + hyperparameter.name,
                                    hyperparameter.value_type,
                                    hyperparameter.n_elements))
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.
        Note that theta are typically the log-transformed values of the
        AF's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the acquisition
            function
        """
        return np.append(self.k1.theta, self.k2.theta)

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the acquisition
            function
        """
        k1_dims = self.k1.n_dims
        self.k1.theta = theta[:k1_dims]
        self.k2.theta = theta[k1_dims:]

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        return (self.k1 == b.k1 and self.k2 == b.k2) \
            or (self.k1 == b.k2 and self.k2 == b.k1)


class Sum(AcquisitionFunctionOperator):
    """Overwrites the ``+`` operator for two or more AF's.
    Additionally gradients are computed and calculated together
    according to the rules of differentiation.

    A sum of an AF can be either with another (composite) AF or
    a real number.
    """

    def __call__(self, X, gp, eval_gradient=False):
        if eval_gradient:
            k1, k1_grad = self.k1(X, gp, eval_gradient)
            k2, k2_grad = self.k2(X, gp, eval_gradient)
            return k1 + k2, k1_grad + k2_grad
        else:
            return self.k1(X, gp) + self.k2(X, gp)

    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)


class Product(AcquisitionFunctionOperator):
    """Overwrites the ``*`` operator for two or more AF's.
    Additionally gradients are computed and calculated together
    according to the rules of differentiation.

    A product of an AF can be either with another (composite) AF or
    a real number.
    """

    def __call__(self, X, gp, eval_gradient=False):
        if eval_gradient:
            k1, k1_grad = self.k1(X, gp, eval_gradient)
            k2, k2_grad = self.k2(X, gp, eval_gradient)
            return k1 * k2, k1_grad * k2 + k2_grad * k1
        else:
            return self.k1(X, gp) * self.k2(X, gp)

    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)


class Exponentiation(AcquisitionFunction):
    """Defines the expontentiation of an AF with a real number.
    Additionally gradients are computed and calculated together
    according to the rules of differentiation.

    .. warning::
        An AF can only be exponentiated with a number and not
        with another AF::

            new_af = old_af ** number

    """

    def __init__(self, acquisition_function, exponent):
        self.acquisition_function = acquisition_function
        self.exponent = exponent
        self.hasgradient = acquisition_function.hasgradient

    def get_params(self, deep=True):
        """Get parameters of this Acquisition function.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict(acquisition_function=self.acquisition_function,
                      exponent=self.exponent)
        if deep:
            deep_items = self.acquisition_function.get_params().items()
            params.update(('acquisition_function__' + k, val) for k,
                          val in deep_items)
        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.acquisition_function.hyperparameters:
            r.append(Hyperparameter("acquisition_function__" +
                                    hyperparameter.name,
                                    hyperparameter.value_type,
                                    hyperparameter.n_elements))
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.
        Note that theta are typically the log-transformed values of the
        acquisition function's hyperparameters as this representation of the
        search space is more amenable for hyperparameter search, as
        hyperparameters like length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the acquisition
            function
        """
        return self.acquisition_function.theta

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the acquisition
            function
        """
        self.acquisition_function.theta = theta

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        return (self.acquisition_function == b.acquisition_function and
                self.exponent == b.exponent)

    def __call__(self, X, gp, eval_gradient=False):
        """Return the Value of the AF at x (``A_f(X, gp)``) and optionally its
        gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            X-Value at which the Acquisition function shall be evaluated

        gp : SKLearn GaussianProcessRegressor
            The GPRegressor (surrogate model) from which to evaluate
            GP(X) and optionally X_train and Y_train.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to X is calculated.

        Returns
        -------
        A_f : array of shape (n_samples_X)
            The value of the acquisition function at point(s) X

        A_f_gradient : array of shape (n_samples_X, n_dim)
            The gradient of the Acquisition function with respect to X.
            Only returned when eval_gradient is True.
        """
        X = self.check_X(X)
        if eval_gradient:
            K, K_grad = self.acquisition_function(X, gp, eval_gradient)
            return K ** self.exponent, K_grad * self.exponent * \
                K ** (self.exponent - 1)
        else:
            K = self.acquisition_function(X, gp)
            return K ** self.exponent

    def __repr__(self):
        return "{0} ** {1}".format(self.acquisition_function, self.exponent)
