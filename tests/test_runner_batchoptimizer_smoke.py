"""
End-to-end BatchOptimizer runner smoke test.

Companion to ``test_runner_nora_smoke.py`` (NORA + BlackJAX) and
``test_runner_nora_polychord_smoke.py``. Same fixture, same budget, same
posterior tolerance — but using **BatchOptimizer** (the current default
acquisition engine) instead of NORA.

After the JAX-split refactor lands, NORA is intended to become the default
acquisition engine. BatchOptimizer remains a supported path and the user has
explicitly asked that it stay maintained. This smoke test pins the gate: it
must converge end-to-end on the canonical 2-D Gaussian fixture and produce a
posterior accurate to ``0.25 σ`` in each marginal.

Note vs. ``test_jax_e2e.py``: those tests cover *JAX-vs-numpy prediction
agreement* but do not assert ``has_converged``. This test fills that gap.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


_HAS_BLACKJAX = True
try:
    import blackjax  # noqa: F401
except ImportError:
    _HAS_BLACKJAX = False


pytestmark = pytest.mark.skipif(
    not _HAS_BLACKJAX,
    reason="blackjax not installed; the BatchOptimizer + JAX path's default "
           "final-MC sampler is BlackJAX, which we want to exercise here",
)


def _anisotropic_gaussian_2d():
    """Same 2-D anisotropic Gaussian fixture as the NORA smokes."""
    sigma = np.array([1.0, 0.5])
    mu = np.array([0.0, 0.0])
    inv_var = 1.0 / sigma ** 2

    def logLkl(x_0, x_1):
        x = np.array([x_0, x_1])
        diff = x - mu
        return -0.5 * float(np.sum(diff ** 2 * inv_var))

    bounds = [[-5.0, 5.0], [-2.5, 2.5]]
    return logLkl, bounds, mu, sigma


_MAX_EVALS = 150


def test_batchoptimizer_2d_anisotropic_gaussian_converges():
    """BatchOptimizer + JAX-GPR converges end-to-end on a 2-D Gaussian.

    Asserts:
    1. ``runner.has_converged`` is True after ``runner.run()`` returns
       (i.e. convergence triggered before the eval budget was exhausted).
    2. ``runner.surrogate.n_total <= 150`` — the budget cap was not the
       reason the loop exited.
    3. The final MC sample's mean is within ``0.25 σ`` of the truth in each
       marginal.
    """
    from gpry.run import Runner
    from gpry.tools import mean_covmat_from_samples

    logLkl, bounds, true_mean, true_sigma = _anisotropic_gaussian_2d()

    tmpdir = tempfile.mkdtemp(prefix="gpry_batchopt_smoke_")
    try:
        runner = Runner(
            logLkl, bounds,
            surrogate={"regressor": {"kernel": "RBF", "use_jax": True}},
            # BatchOptimizer is the current default; pass it explicitly so the
            # test reads as a deliberate choice rather than relying on the
            # implicit default that Phase 7 will flip to NORA.
            gp_acquisition={"BatchOptimizer": {}},
            options={"max_total": _MAX_EVALS, "max_finite": _MAX_EVALS},
            checkpoint=os.path.join(tmpdir, "run"),
            load_checkpoint="overwrite",
            verbose=0,
            seed=42,
        )
        runner.run()

        assert runner.has_converged, (
            f"Runner did not converge in {runner.surrogate.n_total} evals "
            f"(budget {_MAX_EVALS}). BatchOptimizer is supposed to converge "
            f"on a 2-D Gaussian well inside this budget."
        )
        assert runner.surrogate.n_total <= _MAX_EVALS, (
            f"Runner used {runner.surrogate.n_total} evals, budget was "
            f"{_MAX_EVALS}."
        )

        mc = runner.last_mc_samples()
        assert mc is not None and isinstance(mc, dict) and "X" in mc, (
            "Runner did not produce an MC sample after convergence."
        )
        mc_mean, _ = mean_covmat_from_samples(mc["X"], w=mc.get("w"))

        offset_in_sigmas = np.abs(mc_mean - true_mean) / true_sigma
        assert np.all(offset_in_sigmas < 0.25), (
            f"MC mean {mc_mean} is more than 0.25 σ away from the truth "
            f"{true_mean}; offset/σ = {offset_in_sigmas}."
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
