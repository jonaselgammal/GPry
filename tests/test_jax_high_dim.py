"""
High-dimensional end-to-end tests: 8D and 12D JAX vs NumPy comparison.

Tests GPry convergence and runtime on multivariate Gaussian likelihoods
in 8 and 12 dimensions with both JAX and NumPy backends.
"""

import sys
import os
import time
import tempfile
import shutil
import signal
import traceback
import numpy as np
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpry.run import Runner
from gpry.tools import mean_covmat_from_samples


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Run exceeded time limit")


TIMEOUT_SECONDS = 600  # 10 minutes per individual run


# ---------------------------------------------------------------------------
# Build multivariate Gaussian likelihood for arbitrary dimension
# ---------------------------------------------------------------------------

def make_likelihood(ndim, seed=42):
    """Create a multivariate Gaussian likelihood and its true parameters.

    Returns (logLkl_func, true_mean, true_cov, bounds).
    The logLkl function accepts ndim separate scalar arguments.
    """
    rng = np.random.RandomState(seed)

    # True mean near center of [-10, 10]
    true_mean = rng.uniform(-2.0, 2.0, size=ndim)

    # Diagonal-dominant covariance: diagonal in [0.3, 1.5], small off-diag
    diag = rng.uniform(0.3, 1.5, size=ndim)
    A = rng.randn(ndim, ndim) * 0.05  # small off-diagonal perturbation
    true_cov = np.diag(diag) + A @ A.T  # guaranteed positive definite

    rv = multivariate_normal(true_mean, true_cov)
    bounds = [[-10.0, 10.0]] * ndim

    # Build function with the right number of named arguments
    # so that GPry can inspect the signature
    arg_names = [f"x_{i}" for i in range(ndim)]
    func_args = ", ".join(arg_names)
    func_body = f"""
def logLkl({func_args}):
    x = np.array([{func_args}])
    return rv.logpdf(x)
"""
    local_ns = {"np": np, "rv": rv}
    exec(func_body, local_ns)
    logLkl = local_ns["logLkl"]

    return logLkl, true_mean, true_cov, bounds


# ---------------------------------------------------------------------------
# Run GPry
# ---------------------------------------------------------------------------

def run_gpry(logLkl, bounds, true_mean, use_jax, label, seed=42, options=None):
    """Run GPry on the given likelihood.

    Returns dict with timing, convergence, and posterior statistics,
    or a dict with error info if the run fails or times out.
    """
    tmpdir = tempfile.mkdtemp(prefix=f"gpry_{label.replace(' ', '_')}_")
    checkpoint = os.path.join(tmpdir, "run")

    surrogate_cfg = {
        "regressor": {
            "kernel": "RBF",
            "use_jax": use_jax,
        },
    }

    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    try:
        t_start = time.time()

        runner = Runner(
            logLkl,
            bounds,
            surrogate=surrogate_cfg,
            checkpoint=checkpoint,
            load_checkpoint="overwrite",
            verbose=0,
            seed=seed,
            options=options,
        )
        runner.run()

        t_total = time.time() - t_start

        # Cancel alarm
        signal.alarm(0)

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
            "error": None,
        }

        # Extract posterior from MC samples if available
        mc = runner.last_mc_samples()
        if mc is not None:
            mean_mc, cov_mc = mean_covmat_from_samples(mc["X"], mc["w"])
            result["mean_mc"] = mean_mc
            result["cov_mc"] = cov_mc
        else:
            from gpry.tools import mean_covmat_from_evals
            mean_est, cov_est = mean_covmat_from_evals(
                runner.surrogate.X_regress,
                runner.surrogate.y_regress,
            )
            result["mean_mc"] = mean_est
            result["cov_mc"] = cov_est

        return result

    except TimeoutError:
        signal.alarm(0)
        t_total = time.time() - t_start
        return {
            "label": label,
            "use_jax": use_jax,
            "jax_active": None,
            "converged": False,
            "n_total": None,
            "n_regress": None,
            "n_eval": None,
            "n_eval_loglike": None,
            "t_total": t_total,
            "error": f"TIMEOUT after {TIMEOUT_SECONDS}s",
            "mean_mc": None,
            "cov_mc": None,
        }
    except Exception as e:
        signal.alarm(0)
        t_total = time.time() - t_start
        return {
            "label": label,
            "use_jax": use_jax,
            "jax_active": None,
            "converged": False,
            "n_total": None,
            "n_regress": None,
            "n_eval": None,
            "n_eval_loglike": None,
            "t_total": t_total,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "mean_mc": None,
            "cov_mc": None,
        }
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_results(r, true_mean, true_cov):
    """Pretty-print results from a run."""
    ndim = len(true_mean)
    print(f"\n{'='*65}")
    print(f"  {r['label']}")
    print(f"{'='*65}")

    if r["error"]:
        print(f"  ERROR: {r['error']}")
        print(f"  Wall-clock time:     {r['t_total']:.2f} s")
        return

    print(f"  JAX active:          {r['jax_active']}")
    print(f"  Converged:           {r['converged']}")
    print(f"  Total evaluations:   {r['n_total']}")
    print(f"  Training points:     {r['n_regress']}")
    print(f"  GP predict calls:    {r['n_eval']}")
    print(f"  GP LML evaluations:  {r['n_eval_loglike']}")
    print(f"  Wall-clock time:     {r['t_total']:.2f} s")

    if r["mean_mc"] is not None:
        mean_err = np.abs(r['mean_mc'] - true_mean)
        print(f"\n  True mean:       {np.array2string(true_mean, precision=3)}")
        print(f"  Recovered mean:  {np.array2string(r['mean_mc'], precision=3)}")
        print(f"  Mean abs error:  max={np.max(mean_err):.4f}, "
              f"avg={np.mean(mean_err):.4f}")

        if r["cov_mc"] is not None:
            # Compare diagonals (variances)
            true_diag = np.diag(true_cov)
            rec_diag = np.diag(r["cov_mc"])
            diag_err = np.abs(rec_diag - true_diag)
            print(f"\n  True variances:      {np.array2string(true_diag, precision=3)}")
            print(f"  Recovered variances: {np.array2string(rec_diag, precision=3)}")
            print(f"  Variance abs error:  max={np.max(diag_err):.4f}, "
                  f"avg={np.mean(diag_err):.4f}")


def compare_results(r_jax, r_np, true_mean, true_cov, ndim):
    """Compare JAX vs numpy results for a given dimension."""
    print(f"\n{'='*65}")
    print(f"  COMPARISON {ndim}D: JAX vs NumPy")
    print(f"{'='*65}")

    # Handle errors
    if r_jax["error"] and r_np["error"]:
        print(f"\n  Both runs failed.")
        print(f"    JAX:   {r_jax['error'][:80]}")
        print(f"    NumPy: {r_np['error'][:80]}")
        return

    if r_jax["error"]:
        print(f"\n  JAX run failed: {r_jax['error'][:80]}")
        print(f"  NumPy time: {r_np['t_total']:.1f} s")
        return

    if r_np["error"]:
        print(f"\n  NumPy run failed: {r_np['error'][:80]}")
        print(f"  JAX time: {r_jax['t_total']:.1f} s")
        return

    # Timing
    speedup = r_np['t_total'] / r_jax['t_total'] if r_jax['t_total'] > 0 else float('inf')
    print(f"\n  Wall-clock time:")
    print(f"    JAX:     {r_jax['t_total']:.1f} s")
    print(f"    NumPy:   {r_np['t_total']:.1f} s")
    print(f"    Speedup: {speedup:.2f}x {'(JAX faster)' if speedup > 1 else '(NumPy faster)'}")

    # Posterior mean error
    if r_jax["mean_mc"] is not None and r_np["mean_mc"] is not None:
        mean_err_jax = np.max(np.abs(r_jax['mean_mc'] - true_mean))
        mean_err_np = np.max(np.abs(r_np['mean_mc'] - true_mean))
        print(f"\n  Posterior mean recovery (max abs error):")
        print(f"    JAX:   {mean_err_jax:.4f}")
        print(f"    NumPy: {mean_err_np:.4f}")

    # Evaluations
    print(f"\n  Convergence:")
    print(f"    JAX:   {'YES' if r_jax['converged'] else 'NO'} "
          f"({r_jax['n_total']} likelihood evals)")
    print(f"    NumPy: {'YES' if r_np['converged'] else 'NO'} "
          f"({r_np['n_total']} likelihood evals)")

    print(f"\n  GP predict calls:")
    print(f"    JAX:   {r_jax['n_eval']}")
    print(f"    NumPy: {r_np['n_eval']}")
    print(f"\n  LML evaluations:")
    print(f"    JAX:   {r_jax['n_eval_loglike']}")
    print(f"    NumPy: {r_np['n_eval_loglike']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEED = 42
    OPTIONS = {
        "max_finite": 150,
        "max_total": 300,
        "max_initial": 60,
    }

    all_results = {}

    for ndim in [8, 12]:
        print("\n" + "#" * 65)
        print(f"#  {ndim}D MULTIVARIATE GAUSSIAN TEST")
        print("#" * 65)

        logLkl, true_mean, true_cov, bounds = make_likelihood(ndim, seed=SEED)

        print(f"  Bounds:        [-10, 10]^{ndim}")
        print(f"  True mean:     {np.array2string(true_mean, precision=3)}")
        print(f"  True var diag: {np.array2string(np.diag(true_cov), precision=3)}")
        print(f"  Options:       {OPTIONS}")

        # --- JAX ---
        print(f"\n>>> Running {ndim}D with JAX...")
        r_jax = run_gpry(
            logLkl, bounds, true_mean,
            use_jax=True,
            label=f"{ndim}D JAX",
            seed=SEED,
            options=OPTIONS,
        )
        print_results(r_jax, true_mean, true_cov)

        # --- NumPy ---
        print(f"\n>>> Running {ndim}D with NumPy...")
        r_np = run_gpry(
            logLkl, bounds, true_mean,
            use_jax=False,
            label=f"{ndim}D NumPy",
            seed=SEED,
            options=OPTIONS,
        )
        print_results(r_np, true_mean, true_cov)

        # --- Compare ---
        compare_results(r_jax, r_np, true_mean, true_cov, ndim)

        all_results[ndim] = {"jax": r_jax, "numpy": r_np}

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    print("\n" + "#" * 65)
    print("#  FINAL SUMMARY")
    print("#" * 65)
    print(f"\n  {'Dim':<6} {'Backend':<8} {'Time (s)':<12} {'Converged':<12} "
          f"{'Evals':<8} {'Mean Err':<12} {'JAX Active'}")
    print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*12} {'-'*10}")

    for ndim in [8, 12]:
        true_mean_nd = all_results[ndim]["jax"]["mean_mc"]  # just for shape ref
        logLkl_nd, true_mean_nd, true_cov_nd, _ = make_likelihood(ndim, seed=SEED)

        for backend_key, backend_label in [("jax", "JAX"), ("numpy", "NumPy")]:
            r = all_results[ndim][backend_key]
            if r["error"]:
                print(f"  {ndim:<6} {backend_label:<8} {r['t_total']:<12.1f} "
                      f"{'ERROR':<12} {'-':<8} {'-':<12} {'-'}")
            else:
                mean_err = np.max(np.abs(r['mean_mc'] - true_mean_nd)) if r['mean_mc'] is not None else float('nan')
                print(f"  {ndim:<6} {backend_label:<8} {r['t_total']:<12.1f} "
                      f"{'YES' if r['converged'] else 'NO':<12} "
                      f"{r['n_total']:<8} {mean_err:<12.4f} {r['jax_active']}")

    print()
