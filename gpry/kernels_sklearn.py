"""Concrete numpy/sklearn kernel implementations for GPry.

This module contains the public concrete kernel classes (``RBF``, ``Matern``,
``ConstantKernel``, ``WhiteKernel``, ``RationalQuadratic``, ``ExpSineSquared``,
``DotProduct``). They wrap sklearn's kernel objects to inherit the
parameter-validation, ``theta``/``bounds`` plumbing and ``__call__`` numerics
(with ``eval_gradient=True`` support).

Each kernel that has a JAX implementation overrides ``evaluate_jax_fn()`` to
return the corresponding JIT-compiled function from ``kernels_jax.py``. This
lets the JAX backend dispatch to kernel-specific math via a method call,
removing the ``type(kernel)`` switch that used to live in ``jax_accel.py``.

Module imports the abstract pieces (``Kernel``, ``Hyperparameter``) from
``kernels_base.py``; the JAX functions from ``kernels_jax.py`` are imported
lazily inside ``evaluate_jax_fn()`` so users without JAX can still use the
numpy classes.
"""

import warnings
from math import sqrt

import numpy as np
from sklearn.gaussian_process.kernels import (  # type: ignore
    ConstantKernel as sk_ConstantKernel,
    DotProduct as sk_DotProduct,
    ExpSineSquared as sk_ExpSineSquared,
    Matern as sk_Matern,
    RationalQuadratic as sk_RationalQuadratic,
    RBF as sk_RBF,
    WhiteKernel as sk_WhiteKernel,
)

from gpry.kernels_base import Hyperparameter, Kernel


class RBF(Kernel, sk_RBF):
    def __init__(
        self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), prior_bounds=None
    ):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.prior_bounds = prior_bounds
        if isinstance(length_scale_bounds, str) and length_scale_bounds == "dynamic":
            if prior_bounds is None:
                raise TypeError(
                    "Prior bounds are required for the RBF kernel "
                    "if its hyperparameter bounds are set to 'dynamic'. "
                    "You can either provide these bounds or set the "
                    "hyperparameter bounds to either numeric values or "
                    "'fixed'"
                )
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
                        "unintended behaviour."
                    )
                self.max_length = prior_bounds[:, 1] - prior_bounds[:, 0]
            else:
                self.max_length = prior_bounds[:, 1] - prior_bounds[:, 0]
        else:
            self.max_length = None

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                self.max_length,
                len(self.length_scale),
            )
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds, self.max_length
        )

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

    def evaluate_jax_fn(self):
        from gpry.kernels_jax import _rbf_kernel_matrix
        return _rbf_kernel_matrix


class Matern(Kernel, sk_Matern):
    def __init__(
        self,
        length_scale=1.0,
        length_scale_bounds=(1e-5, 1e5),
        nu=1.5,
        prior_bounds=None,
    ):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.nu = nu
        self.prior_bounds = prior_bounds
        if isinstance(length_scale_bounds, str) and length_scale_bounds == "dynamic":
            if prior_bounds is None:
                raise TypeError(
                    "Prior bounds are required for the Matern kernel "
                    "if its hyperparameter bounds are set to 'dynamic'. "
                    "You can either provide these bounds or set the "
                    "hyperparameter bounds to either numeric values or "
                    "'fixed'"
                )
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
                        "unintended behaviour."
                    )
                self.max_length = prior_bounds[:, 1] - prior_bounds[:, 0]
            else:
                self.max_length = prior_bounds[:, 1] - prior_bounds[:, 0]
        else:
            self.max_length = None

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                self.max_length,
                len(self.length_scale),
            )
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds, self.max_length
        )

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

            gradient = -np.ones((X_train.shape[0], x.shape[0]))
            mask = dist != 0.0
            scaled_exp_dist[mask] /= dist[mask]
            scaled_exp_dist = np.expand_dims(scaled_exp_dist, axis=1)
            gradient[mask] = scaled_exp_dist[mask] * diff[mask]
            gradient /= length_scale
            return gradient

        elif self.nu == 1.5:
            sqrt_3_dist = sqrt(3) * dist
            f = np.expand_dims(1 + sqrt_3_dist, axis=1)

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

            f *= -1
            f += 1
            return g * f_grad * f

        elif self.nu == 2.5:
            sqrt_5_dist = sqrt(5) * dist
            f2 = (5.0 / 3.0) * dist_sq
            f2 += sqrt_5_dist
            f2 += 1
            f = np.expand_dims(f2, axis=1)

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

    def evaluate_jax_fn(self):
        from gpry.kernels_jax import get_kernel_fn
        return get_kernel_fn("matern", self.nu)


class RationalQuadratic(Kernel, sk_RationalQuadratic):
    def __init__(
        self,
        length_scale=1.0,
        alpha=1.0,
        length_scale_bounds=(1e-5, 1e5),
        alpha_bounds=(1e-5, 1e5),
        prior_bounds=None,
    ):
        self.length_scale = length_scale
        self.alpha = alpha
        self.length_scale_bounds = length_scale_bounds
        self.alpha_bounds = alpha_bounds
        self.prior_bounds = prior_bounds
        if isinstance(length_scale_bounds, str) and length_scale_bounds == "dynamic":
            if prior_bounds is None:
                raise TypeError(
                    "Prior bounds are required for the RQ kernel "
                    "if its hyperparameter bounds are set to 'dynamic'. "
                    "You can either provide these bounds or set the "
                    "hyperparameter bounds to either numeric values or "
                    "'fixed'"
                )
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
                        "unintended behaviour."
                    )
                self.max_length = 2 * max(prior_bounds[:, 1] - prior_bounds[:, 0])
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
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                self.max_length,
                len(self.length_scale),
            )
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds, self.max_length
        )

    @property
    def hyperparameter_alpha(self):
        return Hyperparameter("alpha", "numeric", self.alpha_bounds)

    def gradient_x(self, x, X_train):
        x = np.asarray(x)
        X_train = np.asarray(X_train)
        alpha = self.alpha
        length_scale = self.length_scale

        diff = x - X_train
        diff /= length_scale

        scaled_dist = np.sum(diff**2, axis=1)
        scaled_dist /= 2 * self.alpha
        scaled_dist += 1
        scaled_dist **= -alpha - 1
        scaled_dist *= -1

        scaled_dist = np.expand_dims(scaled_dist, axis=1)
        diff_by_ls = diff / length_scale
        return scaled_dist * diff_by_ls


class ExpSineSquared(Kernel, sk_ExpSineSquared):
    def __init__(
        self,
        length_scale=1.0,
        periodicity=1.0,
        length_scale_bounds=(1e-5, 1e5),
        periodicity_bounds=(1e-5, 1e5),
        prior_bounds=None,
    ):
        self.length_scale = length_scale
        self.periodicity = periodicity
        self.length_scale_bounds = length_scale_bounds
        self.periodicity_bounds = periodicity_bounds
        self.prior_bounds = prior_bounds
        if isinstance(length_scale_bounds, str) and length_scale_bounds == "dynamic":
            if prior_bounds is None:
                raise TypeError(
                    "Prior bounds are required for the RQ kernel "
                    "if its hyperparameter bounds are set to 'dynamic'. "
                    "You can either provide these bounds or set the "
                    "hyperparameter bounds to either numeric values or "
                    "'fixed'"
                )
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
                        "unintended behaviour."
                    )
                self.max_length = 2 * max(prior_bounds[:, 1] - prior_bounds[:, 0])
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
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
                max_length=self.max_length,
            )
        return Hyperparameter(
            "length_scale",
            "numeric",
            self.length_scale_bounds,
            max_length=self.max_length,
        )

    @property
    def hyperparameter_periodicity(self):
        return Hyperparameter("periodicity", "numeric", self.periodicity_bounds)

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

        grad_wrt_theta = np.zeros_like(dist)
        nzd = dist != 0.0
        grad_wrt_theta[nzd] = np.pi / (periodicity * dist[nzd])
        return (
            np.expand_dims(grad_wrt_theta * exp_sine_squared * grad_wrt_exp, axis=1)
            * diff
        )


class ConstantKernel(Kernel, sk_ConstantKernel):
    @property
    def hyperparameter_constant_value(self):
        return Hyperparameter(
            "constant_value", "numeric", self.constant_value_bounds, None
        )

    def gradient_x(self, x, X_train):
        return np.zeros_like(X_train)


class WhiteKernel(Kernel, sk_WhiteKernel):
    @property
    def hyperparameter_noise_level(self):
        return Hyperparameter("noise_level", "numeric", self.noise_level_bounds, None)

    def gradient_x(self, x, X_train):
        return np.zeros_like(X_train)

    def __repr__(self):
        return "{0}(noise_level={1:.3g}**2)".format(
            self.__class__.__name__, np.sqrt(self.noise_level)
        )


class DotProduct(Kernel, sk_DotProduct):
    @property
    def hyperparameter_sigma_0(self):
        return Hyperparameter("sigma_0", "numeric", self.sigma_0_bounds)

    def gradient_x(self, x, X_train):
        return np.asarray(X_train)
