import os
import sys

import numpy as np
from sklearn.base import clone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry.gpr import GaussianProcessRegressor, JaxGaussianProcessRegressor
from gpry.preprocessing import DummyPreprocessor


def fit_gpr(gpr, X, y, **kwargs):
    if gpr.kernel_ is None:
        gpr.kernel_ = clone(gpr.kernel)
    return gpr.fit(X, y, validate=False, **kwargs)


def make_gpr(dim=2, random_state=0):
    bounds = np.tile([[-5.0, 5.0]], (dim, 1))
    length_scale_prior = np.tile([[1e-2, 1e2]], (dim, 1))
    return GaussianProcessRegressor(
        kernel="RBF",
        output_scale_prior=[1e-2, 1e3],
        length_scale_prior=length_scale_prior,
        noise_level=1e-2,
        noise_fixed=True,
        n_restarts_optimizer=2,
        random_state=random_state,
    )


def test_jax_backend_exposes_native_contract():
    rng = np.random.RandomState(0)
    X = rng.randn(25, 2)
    y = np.sin(X[:, 0]) + 0.1 * rng.randn(25)

    gpr = make_gpr()
    fit_gpr(gpr, X, y)

    assert isinstance(gpr, JaxGaussianProcessRegressor)
    assert gpr.native_backend_ready
    assert gpr.array_contract.preferred_input == "jax"


def test_disabling_native_acceleration_preserves_numpy_predictions():
    rng = np.random.RandomState(1)
    X = rng.randn(30, 2)
    y = np.cos(X[:, 0] - X[:, 1]) + 0.05 * rng.randn(30)
    X_test = rng.randn(15, 2)

    gpr = make_gpr(random_state=1)
    fit_gpr(gpr, X, y)

    mean_jax, std_jax = gpr.predict(X_test, return_std=True, validate=False)
    gpr.disable_native_acceleration()
    mean_np, std_np = gpr.predict(X_test, return_std=True, validate=False)

    np.testing.assert_allclose(mean_jax, mean_np, atol=1e-8)
    np.testing.assert_allclose(std_jax, std_np, atol=1e-6)


def test_jax_backend_builds_transformed_loglikelihood():
    X = np.array([[0.0, 0.0], [0.5, -0.5], [1.0, 1.0], [-1.0, 0.2]])
    y = np.array([-0.1, -0.2, -0.3, -0.4])
    gpr = make_gpr()
    fit_gpr(gpr, X, y)

    builder = gpr.make_ns_loglikelihood_builder(
        preprocessing_y=DummyPreprocessor,
        clip_factor=None,
        y_clip_min=float(y.min()),
        y_clip_max=float(y.max()),
    )
    loglikelihood_fn = builder(["x_1", "x_2"])
    params = {"x_1": 0.25, "x_2": -0.25}

    value_jax = float(loglikelihood_fn(params))
    value_ref = float(
        gpr.predict(
            np.array([[params["x_1"], params["x_2"]]]),
            validate=False,
        )[0][0]
    )

    np.testing.assert_allclose(value_jax, value_ref, atol=1e-8)
