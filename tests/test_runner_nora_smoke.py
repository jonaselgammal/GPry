"""
End-to-end NORA + BlackJAX runner smoke test.

This is the **final refactor gate** for ``AGENTS/jax_split_refactor_plan.md``.
It is intentionally not run as part of the per-phase inner loop because a full
runner pass is more expensive and more brittle than the GPR parity tests; it is
the integration check that the structural refactor has not broken the path the
algorithm actually exercises in production.

Configuration matches the plan (Phase 0.3):

- 2D anisotropic Gaussian, fixed seed.
- ``max_total`` ≤ 150 evaluations. (Initial guess of 90 was tight against
  the production-default convergence gate ``RobustConvergence + GaussianKL +
  TrainAlignment``; ``RobustConvergence`` alone needs ~50–70 evals on a 2D
  Gaussian before the rolling-window accuracy gate can sustain three batches.
  150 keeps production defaults honest with a safety margin.)
- NORA acquisition with the BlackJAX nested-sampling backend.
- Final NS posterior-sample mean within ``0.25 σ`` of the truth in each
  marginal (loose enough to absorb BlackJAX RNG variance; can be tightened
  later if the test proves rock-solid).
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
    reason="blackjax not installed; NORA + BlackJAX path is not exercisable",
)


def _anisotropic_gaussian_2d():
    """2D Gaussian with diagonal covariance and different per-dim scales."""
    sigma = np.array([1.0, 0.5])  # anisotropy: y is twice as tight as x
    mu = np.array([0.0, 0.0])
    inv_var = 1.0 / sigma ** 2

    def logLkl(x_0, x_1):
        x = np.array([x_0, x_1])
        diff = x - mu
        return -0.5 * float(np.sum(diff ** 2 * inv_var))

    bounds = [[-5.0, 5.0], [-2.5, 2.5]]
    return logLkl, bounds, mu, sigma


_MAX_EVALS = 150


def test_nora_blackjax_2d_anisotropic_gaussian():
    """NORA + BlackJAX converges on a 2D anisotropic Gaussian within 150 evals.

    The MC mean from the surrogate's last NS sample must lie within ``0.25 σ``
    of the truth in each marginal.
    """
    from gpry.run import Runner
    from gpry.tools import mean_covmat_from_samples

    logLkl, bounds, true_mean, true_sigma = _anisotropic_gaussian_2d()

    tmpdir = tempfile.mkdtemp(prefix="gpry_nora_smoke_")
    try:
        runner = Runner(
            logLkl, bounds,
            surrogate={"regressor": {"kernel": "RBF", "use_jax": True}},
            gp_acquisition={"NORA": {"sampler": "blackjax"}},
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
