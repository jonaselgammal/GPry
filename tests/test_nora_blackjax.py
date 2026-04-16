"""
Test NORA + BlackJAX nested sampler on 3 benchmark likelihoods.

1. 2D correlated Gaussian (sanity check)
2. 2D Rosenbrock likelihood
3. 2D Himmelblau likelihood
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


# ============================================================
# Likelihood definitions
# ============================================================

def make_correlated_gaussian_2d():
    """2D Gaussian with correlation rho=0.8."""
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    rv = multivariate_normal([0, 0], cov)

    def logLkl(x_0, x_1):
        return rv.logpdf([x_0, x_1])

    bounds = [[-5, 5], [-5, 5]]
    true_mean = np.array([0.0, 0.0])
    return logLkl, bounds, true_mean, "Corr. Gaussian 2D"


def make_rosenbrock_2d():
    """Rosenbrock function as a likelihood: logL = -0.5 * [(a-x)^2 + b*(y-x^2)^2]."""
    a, b = 1.0, 100.0

    def logLkl(x_0, x_1):
        return -0.5 * ((a - x_0) ** 2 + b * (x_1 - x_0 ** 2) ** 2)

    bounds = [[-4, 4], [-4, 4]]
    # The mode is at (1, 1); the posterior mean is not analytically simple
    # but should be close to (1, 1) for tight enough bounds.
    true_mean = None  # unknown analytically
    return logLkl, bounds, true_mean, "Rosenbrock 2D"


def make_himmelblau_2d():
    """Himmelblau function as a likelihood: logL = -0.5 * [(x^2+y-11)^2 + (x+y^2-7)^2].
    4 local minima of the chi2, so 4 modes in the likelihood."""
    def logLkl(x_0, x_1):
        return -0.5 * ((x_0 ** 2 + x_1 - 11) ** 2 + (x_0 + x_1 ** 2 - 7) ** 2)

    bounds = [[-5, 5], [-5, 5]]
    true_mean = None  # multimodal, no single true mean
    return logLkl, bounds, true_mean, "Himmelblau 2D"


# ============================================================
# Run benchmark
# ============================================================

def run_one(make_fn, use_jax=True, verbose=1):
    logLkl, bounds, true_mean, name = make_fn()

    tmpdir = tempfile.mkdtemp(prefix=f"gpry_nora_{name.replace(' ', '_')}_")
    checkpoint = os.path.join(tmpdir, "run")

    try:
        t0 = time.time()
        runner = Runner(
            logLkl, bounds,
            surrogate={"regressor": {"kernel": "RBF", "use_jax": use_jax}},
            gp_acquisition={"NORA": {"sampler": "blackjax"}},
            options={"max_finite": 200, "max_total": 400},
            checkpoint=checkpoint,
            load_checkpoint="overwrite",
            verbose=verbose,
            seed=42,
        )
        runner.run()
        t_total = time.time() - t0

        converged = runner.has_converged
        n_total = runner.surrogate.n_total

        # Get posterior samples
        mc = runner.last_mc_samples()
        if mc is not None and isinstance(mc, dict) and "X" in mc:
            mc_mean, mc_cov = mean_covmat_from_samples(mc["X"], w=mc.get("w"))
        elif mc is not None and hasattr(mc, 'shape'):
            mc_mean = np.mean(mc, axis=0)
        else:
            mc_mean = None

        if true_mean is not None and mc_mean is not None:
            error = np.max(np.abs(mc_mean - true_mean))
        elif mc_mean is not None:
            error = None
        else:
            error = None

        return {
            "name": name,
            "converged": converged,
            "n_total": n_total,
            "time": t_total,
            "mc_mean": mc_mean,
            "error": error,
            "err_str": None,
        }
    except Exception as e:
        return {
            "name": name,
            "converged": False,
            "n_total": None,
            "time": time.time() - t0 if 't0' in dir() else 0,
            "mc_mean": None,
            "error": None,
            "err_str": str(e)[:60],
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    benchmarks = [
        make_correlated_gaussian_2d,
        make_rosenbrock_2d,
        make_himmelblau_2d,
    ]

    print("=" * 85, flush=True)
    print("  NORA + BlackJAX Nested Sampler — Standard GP with JAX", flush=True)
    print("=" * 85, flush=True)
    header = f"  {'Benchmark':<22} {'Conv':<6} {'Evals':<7} {'Time (s)':<10} {'Error/Info':<30}"
    print(header, flush=True)
    print("  " + "-" * 80, flush=True)

    for make_fn in benchmarks:
        r = run_one(make_fn, use_jax=True, verbose=1)

        status = "Y" if r["converged"] else "N"
        n = str(r["n_total"]) if r["n_total"] is not None else "ERR"
        t = f"{r['time']:.1f}" if r["time"] is not None else "N/A"

        if r.get("err_str"):
            info = r["err_str"]
        elif r["error"] is not None:
            info = f"mean_err={r['error']:.4f}"
        elif r["mc_mean"] is not None:
            info = f"mean={np.array2string(r['mc_mean'], precision=2, max_line_width=40)}"
        else:
            info = "N/A"

        row = f"  {r['name']:<22} {status:<6} {n:<7} {t:<10} {info:<30}"
        print(row, flush=True)

    print("  " + "-" * 80, flush=True)
    print("  Done.", flush=True)
