"""Acquisition functions used by GPry.

Only the LogExp family is in production use: ``LogExp`` is the default
acquisition function wired through ``gp_acquisition.py`` and supported by the
native JAX acquisition objective in ``surrogate.make_native_acquisition_objective``.
``NonlinearLogExp`` is an alternative variant kept because the native objective
still has a branch for it; it shares the base class with ``LogExp``.

Historical context: this module previously contained ``Mu``, ``Std``,
``ExpectedImprovement`` etc. plus composite operator scaffolding modelled on
sklearn's ``Kernel`` API. None of it was wired into the runner, the engine,
or the tests, and it was dropped in Phase 5 of the JAX-split refactor.

Public surface
==============

.. autosummary::

    AcquisitionFunction
    BaseLogExp
    LogExp
    NonlinearLogExp
    is_acquisition_function
    builtin_names
"""

import sys
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from inspect import getmembers
from typing import Protocol, Tuple, Union

import numpy as np


def _safe_log_expm1(x):
    """Numerically safer ``log(exp(x) - 1)`` for the ``NonlinearLogExp`` branch."""
    mask = x < 1
    ret = np.empty_like(x)
    ret[mask] = np.log(np.expm1(x[mask]))
    ret[~mask] = x[~mask] + np.log1p(-np.exp(-x[~mask]))
    return ret


def builtin_names():
    """List the names of every built-in acquisition function class.

    Only used by error messages in ``gp_acquisition.py`` to tell the user what
    acquisition-function names are resolvable by string.
    """
    return [
        name
        for name, obj in getmembers(sys.modules[__name__])
        if isinstance(obj, type)
        and issubclass(obj, BaseLogExp)
        and obj is not BaseLogExp
    ]


class AcquisitionFunction(Protocol):
    """Minimal acquisition-function interface for GPry.

    Implementations receive the ``SurrogateModel`` (not the raw GP regressor)
    and return either the acquisition values (when ``eval_gradient=False``) or
    ``(values, gradients)`` (when ``eval_gradient=True``).

    The first positional argument after ``self`` is ``X``, the second is
    ``surrogate``. The ``validate`` kwarg is a hot-path optimization that
    skips the surrogate-side input-shape checks; ``gp_acquisition.py`` passes
    ``validate=False`` from inside its inner loop.
    """

    hasgradient: bool

    def __call__(
        self,
        X: np.ndarray,
        surrogate,
        *,
        eval_gradient: bool = False,
        validate: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        ...


def is_acquisition_function(obj):
    """Whether ``obj`` is a GPry acquisition function instance.

    Used by ``gp_acquisition.py`` to decide whether to take the user-supplied
    object as-is or to construct one from a string/dict spec.
    """
    return isinstance(obj, BaseLogExp)


def _check_X(X):
    """Coerce ``X`` to a 2-D numpy array. Raises if not an ndarray."""
    if not isinstance(X, np.ndarray):
        raise ValueError(f"Expected a numpy array for X, instead got {X!r}")
    if X.ndim == 1:
        return X.reshape(1, -1)
    return X


class BaseLogExp(metaclass=ABCMeta):
    r"""Shared implementation for the LogExp family of acquisition functions.

    Computes the log of an exponentiated-mean times log-error-bar criterion
    designed to efficiently sample log-probability distributions:

    .. math::

        \log A(X) = 2\zeta \cdot \mu(X) + g(\sigma(X), \sigma_n)

    where ``g`` is supplied by the concrete subclass (``LogExp`` /
    ``NonlinearLogExp``).

    Parameters
    ----------
    zeta : float, optional
        Exploration-exploitation tradeoff parameter. If ``None``, computed
        from ``dimension`` via :meth:`auto_zeta`.
    sigma_n : float, optional
        Constant noise level. If ``None``, the surrogate's noise level is
        used at call time.
    fixed : bool, default=False
        Reserved for backwards compatibility with the old hyperparameter API.
    dimension : int, optional
        Dimensionality of the parameter space, used to auto-scale ``zeta``.
    zeta_scaling : float, default=0.85
        Scaling power for ``zeta`` when auto-computed from ``dimension``.
    zeta_schedule : None, ``"ramp"``, or callable, default=None
        Optional dynamic zeta schedule.

        - ``None``: static zeta.
        - ``"ramp"``: linear ramp from 0 to ``zeta`` over ``n_explore``
          acquisitions after the initial training set.
        - callable ``f(n_acquired, n_train, d) -> zeta_eff``: custom schedule.
    n_explore : int or str, default=``"3d"``
        Width of the ramp window (only when ``zeta_schedule="ramp"``).
        ``"Nd"`` is interpreted as ``N * dimension``.
    """

    def __init__(
        self,
        zeta=None,
        sigma_n=None,
        fixed=False,
        dimension=None,
        zeta_scaling=0.85,
        zeta_schedule=None,
        n_explore="3d",
    ):
        self._dimension = dimension
        if zeta is None:
            if dimension is None:
                raise ValueError(
                    "We need the dimensionality of the problem to "
                    "guess an appropriate zeta value."
                )
            self.zeta = self.auto_zeta(dimension, scaling=zeta_scaling)
        else:
            self.zeta = zeta
        self.sigma_n = sigma_n
        self.fixed = fixed
        self.hasgradient = True
        self.zeta_schedule = zeta_schedule
        self._n_explore_raw = n_explore
        self._n_explore = self._parse_n_explore(n_explore, dimension)

    @staticmethod
    @abstractmethod
    def f(mu, std, baseline, noise_level, zeta):
        """Compute the (vectorized) AF value at given mean/std/scalars."""

    @staticmethod
    def auto_zeta(dimension, scaling=0.85):
        return dimension ** (-scaling)

    @staticmethod
    def _parse_n_explore(n_explore, dimension):
        """Parse ``n_explore`` (int or ``"Nd"`` string) into an integer."""
        if isinstance(n_explore, str):
            if n_explore.endswith("d"):
                if dimension is None:
                    return None  # resolved later when ``d`` is known
                return int(n_explore[:-1]) * int(dimension)
            raise ValueError(
                "n_explore string must be of the form 'Nd', e.g. '3d'. "
                f"Got: {n_explore!r}"
            )
        return int(n_explore)

    def effective_zeta(self, surrogate):
        """Return the effective ``zeta`` for the current surrogate state.

        For ``zeta_schedule=None`` this is just ``self.zeta``. With a ramp or
        callable schedule, ``surrogate.n_regress`` and ``surrogate.d`` set the
        progress through the schedule.

        Called from ``surrogate.make_native_acquisition_objective`` to bake
        the scalar into the JAX closure (the closure cache keys on this
        value). Also called from ``gp_acquisition.BatchOptimizer.multi_add``
        and from this AF's own ``__call__`` so all evaluation paths share one
        scheduling rule.
        """
        if self.zeta_schedule is None:
            return self.zeta
        # ``surrogate.fitted`` is False until the first ``append()`` runs.
        # During the first call the schedule cannot be evaluated meaningfully
        # (n_regress / d are zero); fall back to the static zeta.
        if not getattr(surrogate, "fitted", False):
            return self.zeta
        n_train = surrogate.n_regress
        d = surrogate.d
        n_initial = 2 * d  # typical GPry default
        n_acquired = max(0, n_train - n_initial)
        if self.zeta_schedule == "ramp":
            n_explore = self._n_explore
            if n_explore is None:
                n_explore = self._parse_n_explore(self._n_explore_raw, d)
            if n_explore <= 0:
                return self.zeta
            frac = min(n_acquired / n_explore, 1.0)
            return frac * self.zeta
        if callable(self.zeta_schedule):
            return self.zeta_schedule(n_acquired, n_train, d)
        raise ValueError(
            f"Unknown zeta_schedule: {self.zeta_schedule!r}. "
            "Use None, 'ramp', or a callable."
        )

    def _resolve_noise(self, surrogate):
        """Return the scalar noise level used in the AF body."""
        if self.sigma_n is None:
            sigma_n = surrogate.noise_level
            if isinstance(sigma_n, Iterable):
                return float(np.mean(sigma_n))
            return sigma_n
        return self.sigma_n

    def __call__(self, X, surrogate, *, eval_gradient=False, validate=True):
        """Return the AF value at ``X`` given ``surrogate`` (and its gradient
        if ``eval_gradient=True``).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_features,)
            Points at which to evaluate the AF.
        surrogate : SurrogateModel
            The GPry surrogate model.
        eval_gradient : bool, default=False
            Whether to also compute the gradient w.r.t. ``X``.
        validate : bool, default=True
            Hot-path optimization: forwards to ``surrogate.predict``.

        Returns
        -------
        values : ndarray of shape (n_samples,)
            The AF values at ``X``.
        grad : ndarray of shape (n_samples, n_features), optional
            Gradient w.r.t. ``X``. Only returned when ``eval_gradient=True``.
        """
        X = _check_X(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if eval_gradient:
                mu, std, mu_grad, std_grad = surrogate.predict(
                    X,
                    return_std=True,
                    return_mean_grad=True,
                    return_std_grad=True,
                    validate=validate,
                )
            else:
                mu, std = surrogate.predict(X, return_std=True, validate=validate)

        noise_var = self._resolve_noise(surrogate)
        zeta = self.effective_zeta(surrogate)
        baseline = surrogate.y_max

        var = std**2 - noise_var**2.0
        mask = (var > 0) & np.isfinite(mu)
        values = np.zeros_like(std)
        if np.any(mask):
            values[mask] = self.f(mu[mask], std[mask], baseline, noise_var, zeta)
        if np.any(~mask):
            values[~mask] = -np.inf

        if not eval_gradient:
            return values

        sigma_n_scalar = noise_var
        if np.array(std_grad).ndim > 1:
            grad = np.zeros_like(std_grad)
            if np.any(mask):
                grad[mask] = (
                    np.array(std_grad)[mask] / (std[mask] - sigma_n_scalar)
                    + 2 * zeta * np.array(mu_grad)[mask]
                )
            if np.any(~mask):
                grad[~mask] = np.ones_like(std_grad[~mask]) * np.inf
        else:
            std0 = std[0]
            if std0 > sigma_n_scalar:
                grad = std_grad / (std0 - sigma_n_scalar) + 2 * zeta * mu_grad
            else:
                grad = np.ones_like(std_grad) * np.inf
        return values, grad

    def __repr__(self):
        return f"{self.__class__.__name__}(zeta={self.zeta:.3f})"


class LogExp(BaseLogExp):
    r"""Linearized exponentiated log-error-bar AF.

    .. math::

        A_{\mathrm{LE}}(X) = \exp(2\zeta\cdot\mu(X)) \cdot (\sigma(X) - \sigma_n)

    and we take the log of this for numerical stability:

    .. math::

        \log A_{\mathrm{LE}}(X) = 2\zeta\cdot\mu(X) + \log(\sigma(X) - \sigma_n)

    See :class:`BaseLogExp` for parameter docs. The variance is floored at
    ``max(noise_level * 0.01, 1e-6)**2`` inside ``f`` to prevent the ranked
    pool from depleting when std collapses to the noise floor near training
    points.
    """

    @staticmethod
    def f(mu, std, baseline, noise_level, zeta):
        """Linearized exponentiated log-error bar."""
        # Floor epistemic variance to prevent -inf acquisition near training
        # points, which causes NORA's ranked pool to deplete completely.
        min_epistemic_std = max(noise_level * 0.01, 1e-6)
        return 2 * zeta * (mu - baseline) + np.log(
            np.sqrt(
                np.clip(std**2.0 - noise_level**2.0, min_epistemic_std**2, None)
            )
        )


class NonlinearLogExp(BaseLogExp):
    r"""Alternative AF that keeps both scales exponentiated.

    .. math::

        A_{\mathrm{LE}}(X) = \exp(2\zeta\cdot\mu(X)) \cdot \exp(\sigma(X) - \sigma_n)

    and we take the log of this.

    .. warning::
        The analytic gradient inherited from :class:`BaseLogExp` is the
        ``LogExp`` gradient; it is **not** correct for this variant. Use the
        finite-difference / native-JAX path if you need gradients.
    """

    @staticmethod
    def f(mu, std, baseline, noise_level, zeta):
        """Exponentiated log-error bar."""
        return 2 * zeta * (mu - baseline) + _safe_log_expm1(
            np.sqrt(np.clip(std**2.0 - noise_level**2.0, 0.0, None))
        )
