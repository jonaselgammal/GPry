import os
import sys

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry.preprocessing import (
    NormalizeBounds,
    InputWarping,
    PipelineX,
    NormalizeY,
    SoftClipY,
    PipelineY,
)
from gpry.run import Runner


def test_preprocessing_jax_parity():
    bounds = np.array([[-5.0, 5.0], [-3.0, 7.0]])
    X = np.array(
        [
            [-4.0, -2.0],
            [0.5, 1.5],
            [3.0, 6.0],
        ]
    )
    y = np.array([-20.0, -5.0, -7.5, -1.0, -3.0])
    X_train = np.array(
        [
            [-4.0, -2.0],
            [-3.0, -1.0],
            [-2.0, 0.0],
            [-1.0, 1.0],
            [0.0, 2.0],
            [1.0, 3.0],
            [2.0, 4.0],
            [3.0, 5.0],
            [4.0, 6.0],
            [4.5, 6.5],
        ]
    )
    y_train = np.array([-25.0, -16.0, -9.0, -4.0, -1.5, -0.5, -0.4, -1.0, -2.0, -3.5])

    px = PipelineX(
        [NormalizeBounds(bounds), InputWarping(concentration=0.4, min_points=5)]
    )
    px.fit(X_train, y_train)
    py = PipelineY([SoftClipY(delta=3.0, tau=2.0), NormalizeY()])
    py.fit(X_train, y)

    X_np = px.transform(X)
    X_jax = np.asarray(px.transform_jax(jnp.asarray(X)))
    np.testing.assert_allclose(X_jax, X_np, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(px.inverse_transform_jax(jnp.asarray(X_np))),
        px.inverse_transform(X_np),
        atol=1e-10,
    )

    y_np = py.transform(y)
    y_jax = np.asarray(py.transform_jax(jnp.asarray(y)))
    np.testing.assert_allclose(y_jax, y_np, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(py.inverse_transform_jax(jnp.asarray(y_np))),
        py.inverse_transform(y_np),
        atol=1e-10,
    )


@pytest.mark.filterwarnings("ignore:Some of the initial training points are very close")
def test_blackjax_nora_keeps_internal_samples_transformed(tmp_path):
    pytest.importorskip("blackjax")
    os.environ["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")

    mean = np.array([3.0, 2.0])
    cov = np.array([[0.5, 0.4], [0.4, 1.5]])
    rv = multivariate_normal(mean, cov)

    def logl(x0, x1):
        return rv.logpdf(np.array([x0, x1]).T)

    runner = Runner(
        logl,
        [[-10, 10], [-10, 10]],
        surrogate={
            "regressor": {
                "kernel": "RBF",
                "use_jax": True,
                "noise_level": 0.01,
                "output_scale_prior": [0.01, 100.0],
                "length_scale_prior": np.array([[0.01, 100.0], [0.01, 100.0]]),
            }
        },
        gp_acquisition={"NORA": {"sampler": "blackjax", "mc_every": 1}},
        convergence_criterion=False,
        options={
            "n_initial": 6,
            "max_initial": 10,
            "max_finite": 20,
            "max_total": 20,
            "n_points_per_acq": 2,
        },
        checkpoint=str(tmp_path / "checkpoint"),
        load_checkpoint="overwrite",
        verbose=0,
        seed=42,
    )
    runner.run()

    internal = runner.acquisition._X_mc_internal
    external, _, _, w = runner.acquisition.last_mc_sample(copy=False, warn_reweight=False)
    assert internal is not None
    assert external is not None
    assert np.all(internal >= -1e-12)
    assert np.all(internal <= 1.0 + 1e-12)
    assert np.all(external >= -10.0 - 1e-12)
    assert np.all(external <= 10.0 + 1e-12)
    assert np.max(np.abs(external - internal)) > 1.0

    probe = np.array([[1.5, 1.0], [3.0, 2.0], [4.5, 2.5]])
    probe_internal = runner.surrogate.preprocessing_X.transform(probe)
    pred = runner.surrogate.predict(probe, return_std=False, validate=False)
    pred_internal = runner.surrogate.predict_transformed(
        probe_internal, return_std=False, validate=False
    )
    np.testing.assert_allclose(pred_internal, pred, atol=1e-10)

    w = np.ones(len(external)) if w is None else np.asarray(w, dtype=float)
    w = w / np.sum(w)
    mean_est = np.sum(external * w[:, None], axis=0)
    assert np.max(np.abs(mean_est - mean)) < 1.0
