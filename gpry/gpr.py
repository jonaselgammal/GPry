"""Public GP entrypoints and backend re-exports."""

from gpry.gpr_base import BaseGaussianProcessRegressor, EPS_SQ_NOISE, GPR_CHOLESKY_LOWER
from gpry.gpr_sklearn import SklearnGaussianProcessRegressor
from gpry.gpr_jax import JaxGaussianProcessRegressor


GaussianProcessRegressor = BaseGaussianProcessRegressor


__all__ = [
    "BaseGaussianProcessRegressor",
    "GaussianProcessRegressor",
    "SklearnGaussianProcessRegressor",
    "JaxGaussianProcessRegressor",
    "EPS_SQ_NOISE",
    "GPR_CHOLESKY_LOWER",
]
