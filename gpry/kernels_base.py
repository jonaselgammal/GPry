"""Shared abstract kernel scaffolding for the GPry sklearn and JAX backends.

Public names (concrete numpy kernel classes such as ``RBF``, ``Matern``,
``ConstantKernel``, ``WhiteKernel``, ``RationalQuadratic``,
``ExpSineSquared``, ``DotProduct``) live in ``kernels_sklearn.py``. JAX-native
kernel math lives in ``kernels_jax.py``.

This module owns the shared building blocks used by both:
- ``Hyperparameter`` — extended namedtuple used by every kernel.
- ``Kernel`` — base class (wraps the sklearn ``Kernel`` base).
- ``KernelOperator`` + ``Sum``, ``Product``, ``Exponentiation`` — pure-Python
  composition wrappers (no numpy math beyond delegation).

These are kept here because they are inherently backend-agnostic:
``Hyperparameter`` is a config-bearing namedtuple; ``Sum``/``Product`` /
``Exponentiation`` compose other kernels and only express gradients through
delegated calls.
"""

import warnings
from collections import namedtuple

import numpy as np
from sklearn.gaussian_process.kernels import (  # type: ignore
    Kernel as sk_Kernel,
    Exponentiation as sk_Exponentiation,
    Product as sk_Product,
    Sum as sk_Sum,
)


class Hyperparameter(
    namedtuple(
        "Hyperparameter",
        (
            "name",
            "value_type",
            "bounds",
            "max_length",
            "n_elements",
            "fixed",
            "dynamic",
        ),
    )
):
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

    __slots__ = ()

    def __new__(
        cls,
        name,
        value_type,
        bounds,
        max_length,
        n_elements=1,
        fixed=None,
        dynamic=None,
    ):
        if not isinstance(bounds, str) or (bounds != "fixed" and bounds != "dynamic"):
            bounds = np.atleast_2d(bounds)
            if n_elements > 1:  # vector-valued parameter
                if bounds.shape[0] == 1:
                    bounds = np.repeat(bounds, n_elements, 0)
                elif bounds.shape[0] != n_elements:
                    raise ValueError(
                        "Bounds on %s should have either 1 or "
                        "%d dimensions. Given are %d"
                        % (name, n_elements, bounds.shape[0])
                    )

        if fixed is None:
            fixed = isinstance(bounds, str) and bounds == "fixed"
        if dynamic is None:
            dynamic = isinstance(bounds, str) and bounds == "dynamic"
        return super(Hyperparameter, cls).__new__(
            cls, name, value_type, bounds, max_length, n_elements, fixed, dynamic
        )

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.value_type == other.value_type
            and np.all(self.bounds == other.bounds)
            and self.n_elements == other.n_elements
            and self.fixed == other.fixed
            and self.dynamic == other.dynamic
            and self.max_length == other.max_length
        )


class Kernel(sk_Kernel):
    """Base class for gpry kernels.

    Wraps sklearn's ``Kernel`` base to add:
    - ``gradient_x`` (gradient of ``K(x, X_train)`` w.r.t. the test point ``x``),
      used by the acquisition layer.
    - GPry-specific ``bounds`` handling that supports the ``"dynamic"`` policy
      where hyperparameter bounds adapt to the prior extent.
    - Custom ``+``, ``*``, ``**`` composition that produces the gpry-flavoured
      ``Sum`` / ``Product`` / ``Exponentiation`` (so the composition
      machinery keeps gpry's hyperparameter renaming and gradient interface).

    JAX backend contract:
    - Concrete length-scale kernels expose a class- or instance-level
      ``evaluate_jax_fn()`` returning a JIT-compiled ``(X1, X2, length_scale) -> K``
      function from ``kernels_jax.py``. Kernels with no JAX implementation
      raise ``NotImplementedError``.
    """

    def __add__(self, b):
        if not isinstance(b, Kernel):
            from gpry.kernels_sklearn import ConstantKernel
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, Kernel):
            from gpry.kernels_sklearn import ConstantKernel
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            from gpry.kernels_sklearn import ConstantKernel
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, Kernel):
            from gpry.kernels_sklearn import ConstantKernel
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def __pow__(self, b):
        return Exponentiation(self, b)

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        r = [
            getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("hyperparameter_")
        ]
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
                                bounds.append([theta * 1e-3, theta * 100.0])
                            else:
                                bounds.append(
                                    [
                                        hyperparameter.max_length[t] * 1e-3,
                                        hyperparameter.max_length[t] * 100.0,
                                    ]
                                )
                    else:
                        if hyperparameter.max_length[0] is None:
                            bounds.append([thetas * 1e-3, thetas * 100.0])
                        else:
                            bounds.append(
                                [
                                    hyperparameter.max_length[0] * 1e-3,
                                    hyperparameter.max_length[0] * 100.0,
                                ]
                            )
                else:
                    bounds.append(hyperparameter.bounds)
        if len(bounds) > 0:
            return np.log(np.vstack(bounds))
        else:
            return np.array([])

    def gradient_x(self, x, X_train):
        """
        Computes gradient of K(x, X_train) with respect to x.

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

    def evaluate_jax_fn(self):
        """Return a JIT-compiled JAX kernel function for this kernel.

        Concrete subclasses with a JAX implementation override this and return
        a callable ``(X1, X2, length_scale) -> K`` from ``kernels_jax.py``.
        Subclasses without a JAX implementation raise ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"No JAX kernel implementation available for {type(self).__name__}."
        )


class KernelOperator:
    """Updated to accomodate the new kernel hyperparameter definition."""

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = [
            Hyperparameter(
                "k1__" + hyperparameter.name,
                hyperparameter.value_type,
                hyperparameter.bounds,
                hyperparameter.max_length,
                hyperparameter.n_elements,
            )
            for hyperparameter in self.k1.hyperparameters
        ]

        for hyperparameter in self.k2.hyperparameters:
            r.append(
                Hyperparameter(
                    "k2__" + hyperparameter.name,
                    hyperparameter.value_type,
                    hyperparameter.bounds,
                    hyperparameter.max_length,
                    hyperparameter.n_elements,
                )
            )
        return r


class Exponentiation(Kernel, sk_Exponentiation):
    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(
                Hyperparameter(
                    "kernel__" + hyperparameter.name,
                    hyperparameter.value_type,
                    hyperparameter.bounds,
                    hyperparameter.max_length,
                    hyperparameter.n_elements,
                )
            )
        return r

    def gradient_x(self, x, X_train):
        x = np.asarray(x)
        X_train = np.asarray(X_train)
        expo = self.exponent
        kernel = self.kernel

        K = np.expand_dims(kernel(np.expand_dims(x, axis=0), X_train)[0], axis=1)
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
        f_ggrad = np.expand_dims(self.k1(x, X_train)[0], axis=1) * self.k2.gradient_x(
            x, X_train
        )
        fgrad_g = np.expand_dims(self.k2(x, X_train)[0], axis=1) * self.k1.gradient_x(
            x, X_train
        )
        return f_ggrad + fgrad_g
