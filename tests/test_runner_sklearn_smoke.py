"""
Sklearn-only Runner smoke test for ``AGENTS/jax_split_refactor_plan.md`` Phase 0.4.

Exercises the Runner with **no JAX path active**:

- ``surrogate.regressor.use_jax = False`` selects the sklearn-numpy GPR backend.
- ``gp_acquisition`` uses NORA with UltraNest (no JAX dependency).
- ``mc`` uses UltraNest as the final sampler (no JAX dependency).

Reuses the same 2D anisotropic Gaussian, fixed seed, and budget as the BlackJAX
smoke test in ``test_runner_nora_smoke.py`` so that the two tests stay
comparable across the refactor.

This test is the gate for "the package still works when JAX is unavailable",
which is non-negotiable per the plan's "Goal 7" / "Merge Gate" notes in
``AGENTS/gpry_jax_merge_roadmap.md``.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


_HAS_ULTRANEST = True
try:
    import ultranest  # noqa: F401
except ImportError:
    _HAS_ULTRANEST = False


pytestmark = pytest.mark.skipif(
    not _HAS_ULTRANEST,
    reason="ultranest not installed; sklearn-only NORA path needs it",
)


def _anisotropic_gaussian_2d():
    """2D Gaussian with diagonal covariance and different per-dim scales.
    Identical fixture to ``test_runner_nora_smoke.py``."""
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


def test_nora_ultranest_2d_anisotropic_gaussian_no_jax():
    """Runner converges with ``use_jax=False`` and an UltraNest NS backend.

    Uses the same fixture, seed, budget, and ``0.25 σ`` MC-mean tolerance as
    the BlackJAX smoke test. The point here is not to verify identical outputs
    across backends (the parity tests in ``test_gpr_backend_parity.py`` cover
    that) but to verify the sklearn-only path runs to completion.
    """
    from gpry.run import Runner
    from gpry.tools import mean_covmat_from_samples

    logLkl, bounds, true_mean, true_sigma = _anisotropic_gaussian_2d()

    tmpdir = tempfile.mkdtemp(prefix="gpry_sklearn_smoke_")
    try:
        runner = Runner(
            logLkl, bounds,
            surrogate={"regressor": {"kernel": "RBF", "use_jax": False}},
            gp_acquisition={"NORA": {"sampler": "ultranest"}},
            mc={"ultranest": {}},
            options={"max_total": _MAX_EVALS, "max_finite": _MAX_EVALS},
            checkpoint=os.path.join(tmpdir, "run"),
            load_checkpoint="overwrite",
            verbose=0,
            seed=42,
        )
        runner.run()

        assert runner.has_converged, (
            f"Runner did not converge in {runner.surrogate.n_total} evals "
            f"(budget {_MAX_EVALS})."
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
