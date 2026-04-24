"""Concrete numpy/sklearn Gaussian-process backend."""

from gpry.gpr_base import BaseGaussianProcessRegressor


class SklearnGaussianProcessRegressor(BaseGaussianProcessRegressor):
    """Default sklearn/scipy-backed GP implementation."""

    pass
