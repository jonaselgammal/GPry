"""
NORA + PolyChord runner smoke test.

Companion to ``test_runner_nora_smoke.py`` (BlackJAX) and
``test_runner_sklearn_smoke.py`` (sklearn-only). Same fixture, same seed, same
budget, same posterior-tolerance. The point is to gate the second JAX-friendly
NS backend (PolyChord) so the BlackJAX path is not the only NORA backend
under test.

Skipped when ``pypolychord`` is not installed.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


_HAS_POLYCHORD = True
try:
    import pypolychord  # noqa: F401
except ImportError:
    _HAS_POLYCHORD = False


pytestmark = pytest.mark.skipif(
    not _HAS_POLYCHORD,
    reason="pypolychord not installed; NORA + PolyChord path is not exercisable",
)


def _anisotropic_gaussian_2d():
    """2D Gaussian with diagonal covariance and different per-dim scales.
    Identical fixture to the BlackJAX smoke test."""
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


def test_nora_polychord_2d_anisotropic_gaussian():
    """NORA + PolyChord converges on a 2D anisotropic Gaussian within 150 evals.

    The MC mean from the surrogate's last NS sample must lie within ``0.25 σ``
    of the truth in each marginal.
    """
    from gpry.run import Runner
    from gpry.tools import mean_covmat_from_samples

    logLkl, bounds, true_mean, true_sigma = _anisotropic_gaussian_2d()

    tmpdir = tempfile.mkdtemp(prefix="gpry_nora_polychord_smoke_")
    try:
        runner = Runner(
            logLkl, bounds,
            surrogate={"regressor": {"kernel": "RBF", "use_jax": True}},
            gp_acquisition={"NORA": {"sampler": "polychord"}},
            mc={"polychord": {}},
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
