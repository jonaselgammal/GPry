"""High-dimensional pipeline + JAX/numpy parity regression test.

This file replaces two former ``__main__`` script shells
(``test_jax_high_dim.py`` and ``test_jax_full_analysis.py``) that defined no
test functions and no assertions, so pytest collected and silently skipped
them -- a green CI there meant nothing.

It is the regression net for the ``speed-levers`` work: every later change
(warm-start sampling, mixed-precision predict kernel) is measured against it.
It asserts two things on a known high-dimensional anisotropic Gaussian:

1. The (JAX-backed) GPry pipeline recovers the true posterior mean and
   per-parameter variance within tolerance -- i.e. the active-learning loop,
   surrogate, and final nested-sampling pass are correct end to end.
2. On the *final fitted surrogate*, the JAX predict path matches the
   numpy/sklearn predict path (mean and std) to near machine precision -- the
   deterministic backend-parity guarantee that the speed levers must preserve.

The pipeline run is bounded (``max_finite``) and marked ``slow`` so it can be
deselected with ``-m 'not slow'`` for quick iteration.
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pytest
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry.run import Runner
from gpry.tools import mean_covmat_from_samples


# Calibrated so the bounded run converges and recovers the Gaussian well in
# ~1-2 min on CPU (observed at this size: converged at n~142, max mean error
# ~0.02, max per-variance relative error ~0.03). Tolerances below are
# deliberately loose multiples of the observed error -- this is a regression
# net (catch gross breakage, stay flake-free), not a precision benchmark.
# The nested sampler is not fully seedable (UltraNest), so leave wide margin.
NDIM = 10
MAX_FINITE = 120
SEED = 3

MEAN_ABS_TOL = 0.30          # observed ~0.03
VAR_REL_TOL = 0.35           # observed ~0.03
# Deterministic backend parity on the final fitted GP (float64 both paths).
PREDICT_MEAN_ATOL = 1e-5
PREDICT_STD_ATOL = 1e-4


def _make_anisotropic_gaussian(ndim, seed):
    """Anisotropic multivariate Gaussian likelihood with known mean/cov.

    Diagonal-dominant covariance (per-dimension variances in [0.3, 1.5]) plus
    a small off-diagonal perturbation, so each parameter has a distinct scale.
    Returns ``(logp, true_mean, true_cov)`` where ``logp`` takes ``ndim``
    named scalar arguments (GPry inspects the signature for parameter names).
    """
    rng = np.random.RandomState(seed)
    true_mean = rng.uniform(-2.0, 2.0, size=ndim)
    diag = rng.uniform(0.3, 1.5, size=ndim)
    A = rng.randn(ndim, ndim) * 0.05
    true_cov = np.diag(diag) + A @ A.T  # guaranteed positive definite
    rv = multivariate_normal(true_mean, true_cov)

    names = [f"x_{i}" for i in range(ndim)]
    src = "def logp({0}):\n    return rv.logpdf(np.array([{0}]))\n".format(", ".join(names))
    ns = {"np": np, "rv": rv}
    exec(src, ns)
    return ns["logp"], true_mean, true_cov


@pytest.mark.slow
def test_high_dim_pipeline_recovers_truth_and_jax_numpy_parity():
    logp, true_mean, true_cov = _make_anisotropic_gaussian(NDIM, SEED)
    bounds = [[-10.0, 10.0]] * NDIM
    tmpdir = tempfile.mkdtemp(prefix="gpry_highdim_")
    try:
        runner = Runner(
            logp,
            bounds,
            checkpoint=os.path.join(tmpdir, "run"),
            load_checkpoint="overwrite",
            verbose=0,
            seed=SEED,
            options={"max_finite": MAX_FINITE},
        )
        runner.run()

        gpr = runner.surrogate.gpr
        # This must genuinely be the JAX backend, otherwise the parity check
        # below is vacuous (it would compare numpy against numpy).
        assert gpr.use_jax, "expected the JAX backend to be selected"
        assert gpr.native_backend_ready, "JAX backend should be ready after the run"

        # ----- (1) Posterior recovery vs ground truth -----
        runner.generate_mc_sample()
        mc = runner.last_mc_samples()
        assert mc is not None, "final MC sample should be available after run()"
        mean_mc, cov_mc = mean_covmat_from_samples(mc["X"], mc["w"])

        mean_abs_err = np.abs(mean_mc - true_mean)
        assert mean_abs_err.max() < MEAN_ABS_TOL, (
            f"posterior mean off by {mean_abs_err.max():.3f} "
            f"(>{MEAN_ABS_TOL}); recovered={np.round(mean_mc, 3)} "
            f"true={np.round(true_mean, 3)}"
        )
        var_rel_err = np.abs(np.diag(cov_mc) - np.diag(true_cov)) / np.diag(true_cov)
        assert var_rel_err.max() < VAR_REL_TOL, (
            f"posterior variance relative error {var_rel_err.max():.3f} "
            f"(>{VAR_REL_TOL}); recovered_diag={np.round(np.diag(cov_mc), 3)} "
            f"true_diag={np.round(np.diag(true_cov), 3)}"
        )

        # ----- (2) JAX vs numpy predict parity on the final fitted GP -----
        # Same fitted hyperparameters / Cholesky factor; only the linalg
        # backend differs, so the two must agree to ~float64 round-off.
        rng = np.random.RandomState(12345)
        X_test = runner.surrogate.preprocessing_X.transform(
            rng.uniform(-6.0, 6.0, size=(300, NDIM))
        )
        mean_jax, std_jax = gpr.predict_native(X_test, return_std=True)

        gpr.disable_native_acceleration()
        res_np = gpr.predict(X_test, return_std=True, validate=False)
        mean_np, std_np = res_np[0], res_np[1]

        np.testing.assert_allclose(
            mean_jax, mean_np, atol=PREDICT_MEAN_ATOL,
            err_msg="JAX vs numpy predicted mean mismatch on the final surrogate",
        )
        np.testing.assert_allclose(
            std_jax, std_np, atol=PREDICT_STD_ATOL,
            err_msg="JAX vs numpy predicted std mismatch on the final surrogate",
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
