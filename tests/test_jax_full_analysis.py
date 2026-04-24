"""
Full end-to-end likelihood analysis: JAX vs numpy comparison.

Runs GPry on a toy 2D Gaussian likelihood, recovers posterior mean and
covariance, and compares wall-clock runtime and accuracy between JAX and
numpy backends.
"""

import sys
import os
import time
import tempfile
import shutil
import numpy as np
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry.run import Runner
from gpry.tools import mean_covmat_from_samples


# ---------------------------------------------------------------------------
# Toy likelihood: 2D Gaussian with known mean and covariance
# ---------------------------------------------------------------------------

TRUE_MEAN = np.array([3.0, 2.0])
TRUE_COV = np.array([[0.5, 0.4],
                      [0.4, 1.5]])
TRUE_RV = multivariate_normal(TRUE_MEAN, TRUE_COV)


def logLkl(x_1, x_2):
    return TRUE_RV.logpdf(np.array([x_1, x_2]).T)


BOUNDS = [[-10, 10], [-10, 10]]


def run_gpry(use_jax, label, seed=12345):
    """Run GPry on the toy likelihood with JAX enabled or disabled.

    Returns dict with timing, convergence, and posterior statistics.
    """
    tmpdir = tempfile.mkdtemp(prefix=f"gpry_{label.replace(' ', '_')}_")
    checkpoint = os.path.join(tmpdir, "run")

    # Control JAX via the GPR constructor parameter
    surrogate_cfg = {
        "regressor": {
            "kernel": "RBF",
            "use_jax": use_jax,
        },
    }

    try:
        t_start = time.time()

        runner = Runner(
            logLkl,
            BOUNDS,
            surrogate=surrogate_cfg,
            checkpoint=checkpoint,
            load_checkpoint="overwrite",
            verbose=0,
            seed=seed,
        )
        runner.run()

        t_total = time.time() - t_start

        # Extract results
        gpr = runner.surrogate.gpr
        result = {
            "label": label,
            "use_jax": use_jax,
            "jax_active": getattr(gpr, "native_backend_ready", False),
            "converged": runner.has_converged,
            "n_total": runner.surrogate.n_total,
            "n_regress": runner.surrogate.n_regress,
            "n_eval": gpr.n_eval,
            "n_eval_loglike": gpr.n_eval_loglike,
            "t_total": t_total,
        }

        # Extract posterior from MC samples if available
        mc = runner.last_mc_samples()
        if mc is not None:
            mean_mc, cov_mc = mean_covmat_from_samples(mc["X"], mc["w"])
            result["mean_mc"] = mean_mc
            result["cov_mc"] = cov_mc
        else:
            # Fall back to weighted estimate from training evaluations
            from gpry.tools import mean_covmat_from_evals
            mean_est, cov_est = mean_covmat_from_evals(
                runner.surrogate.X_regress,
                runner.surrogate.y_regress,
            )
            result["mean_mc"] = mean_est
            result["cov_mc"] = cov_est

        return result

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def print_results(r):
    """Pretty-print results from a run."""
    print(f"\n{'='*60}")
    print(f"  {r['label']}")
    print(f"{'='*60}")
    print(f"  JAX active:          {r['jax_active']}")
    print(f"  Converged:           {r['converged']}")
    print(f"  Total evaluations:   {r['n_total']}")
    print(f"  Training points:     {r['n_regress']}")
    print(f"  GP predict calls:    {r['n_eval']}")
    print(f"  GP LML evaluations:  {r['n_eval_loglike']}")
    print(f"  Wall-clock time:     {r['t_total']:.2f} s")
    print()
    print(f"  True mean:       {TRUE_MEAN}")
    print(f"  Recovered mean:  {r['mean_mc']}")
    mean_err = np.abs(r['mean_mc'] - TRUE_MEAN)
    print(f"  Mean error:      {mean_err} (max: {np.max(mean_err):.4f})")
    print()
    print(f"  True covariance:")
    print(f"    {TRUE_COV[0]}")
    print(f"    {TRUE_COV[1]}")
    print(f"  Recovered covariance:")
    print(f"    {r['cov_mc'][0]}")
    print(f"    {r['cov_mc'][1]}")
    cov_err = np.abs(r['cov_mc'] - TRUE_COV)
    print(f"  Cov error (max): {np.max(cov_err):.4f}")


def compare_results(r_jax, r_np):
    """Compare JAX vs numpy results."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON: JAX vs NumPy")
    print(f"{'='*60}")

    speedup = r_np['t_total'] / r_jax['t_total']
    print(f"\n  Wall-clock time:")
    print(f"    JAX:     {r_jax['t_total']:.1f} s")
    print(f"    NumPy:   {r_np['t_total']:.1f} s")
    print(f"    Speedup: {speedup:.2f}x")

    # Posterior recovery accuracy
    mean_err_jax = np.max(np.abs(r_jax['mean_mc'] - TRUE_MEAN))
    mean_err_np = np.max(np.abs(r_np['mean_mc'] - TRUE_MEAN))
    cov_err_jax = np.max(np.abs(r_jax['cov_mc'] - TRUE_COV))
    cov_err_np = np.max(np.abs(r_np['cov_mc'] - TRUE_COV))
    print(f"\n  Posterior mean recovery (max abs error):")
    print(f"    JAX:   {mean_err_jax:.4f}")
    print(f"    NumPy: {mean_err_np:.4f}")
    print(f"\n  Posterior covariance recovery (max abs error):")
    print(f"    JAX:   {cov_err_jax:.4f}")
    print(f"    NumPy: {cov_err_np:.4f}")

    print(f"\n  GP predict calls:")
    print(f"    JAX:   {r_jax['n_eval']}")
    print(f"    NumPy: {r_np['n_eval']}")
    print(f"\n  LML evaluations (hyperparam opt):")
    print(f"    JAX:   {r_jax['n_eval_loglike']}")
    print(f"    NumPy: {r_np['n_eval_loglike']}")
    print(f"\n  Convergence:")
    print(f"    JAX:   {'YES' if r_jax['converged'] else 'NO'} "
          f"({r_jax['n_total']} likelihood evals)")
    print(f"    NumPy: {'YES' if r_np['converged'] else 'NO'} "
          f"({r_np['n_total']} likelihood evals)")


if __name__ == "__main__":
    print("=" * 60)
    print("  Full end-to-end likelihood analysis: JAX vs NumPy")
    print("=" * 60)
    print(f"Toy model: 2D Gaussian")
    print(f"  True mean: {list(TRUE_MEAN)}")
    print(f"  True cov:  {TRUE_COV.tolist()}")
    print(f"  Bounds:    {BOUNDS}")

    # Run with JAX
    print("\n>>> Running with JAX...")
    r_jax = run_gpry(use_jax=True, label="JAX-accelerated GPry")
    print_results(r_jax)

    # Run with NumPy (same seed for fair comparison)
    print("\n>>> Running with NumPy...")
    r_np = run_gpry(use_jax=False, label="NumPy-only GPry")
    print_results(r_np)

    # Compare
    compare_results(r_jax, r_np)
    print()
