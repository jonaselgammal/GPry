"""
Benchmark: InputWarping (approach 1), LearnedFeatureGP (approach 3), and both combined.

Tests on 5 hard benchmarks + 1 sanity check (15D correlated Gaussian).
"""
import sys
import os
import time
import tempfile
import shutil
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_benchmarks import (
    run_benchmark,
    make_neals_funnel_10d,
    make_gaussian_shells_2d,
    make_rastrigin_2d,
    make_banana_5d,
    make_gaussian_mixture_5d,
)
from gpry.preprocessing import NormalizeBounds, InputWarping, PipelineX


# ============================================================
# 15D correlated Gaussian (sanity check)
# ============================================================
def make_correlated_gaussian_15d():
    """15D Gaussian with significant correlations — should be easy for GP."""
    d = 15
    rng = np.random.RandomState(123)
    # Random rotation to create correlations
    A = rng.randn(d, d)
    cov = A @ A.T / d  # positive definite, condition number ~O(d)
    # Make it well-conditioned but correlated
    eigvals = np.linalg.eigvalsh(cov)
    cov = cov * (1.0 / eigvals.max())  # max eigenvalue = 1
    cov += 0.1 * np.eye(d)  # ensure reasonable condition number
    true_mean = np.zeros(d)
    rv = multivariate_normal(true_mean, cov)

    def logLkl(*args):
        x = np.array(args)
        return rv.logpdf(x)

    bounds = [[-6, 6]] * d
    return logLkl, bounds, true_mean, "Corr. Gaussian 15D"


# ============================================================
# Configurations
# ============================================================
BENCHMARKS = [
    make_gaussian_shells_2d,     # ring geometry
    make_rastrigin_2d,           # highly multimodal
    make_banana_5d,              # curved degeneracy
    make_gaussian_mixture_5d,    # multimodal high-D
    make_neals_funnel_10d,       # extreme scale variation
    make_correlated_gaussian_15d,  # sanity check
]


def get_configs(bounds):
    """Return dict of {config_name: surrogate_cfg} for each approach."""
    d = len(bounds)
    configs = {}

    # Baseline: standard GP (no warping, no learned features)
    configs["Baseline"] = {
        "regressor": {"kernel": "RBF", "use_jax": False},
    }

    # Approach 1: InputWarping
    configs["InputWarp"] = {
        "preprocessing_X": PipelineX([
            NormalizeBounds(bounds),
            InputWarping(concentration=0.5, min_points=max(15, 2 * d)),
        ]),
        "regressor": {"kernel": "RBF", "use_jax": False},
    }

    # Approach 3: LearnedFeatureGP
    configs["LearnedFeat"] = {
        "regressor": {
            "backend": "learned_features",
            "kernel": "RBF",
            "noise_level": 0.01,
            "length_scale_prior": [1e-2, 1e1],
            "n_restarts_optimizer": 1,
            "n_train_steps": 150,
        },
    }

    # Approach 1+3: InputWarping + LearnedFeatureGP
    configs["Warp+Learned"] = {
        "preprocessing_X": PipelineX([
            NormalizeBounds(bounds),
            InputWarping(concentration=0.5, min_points=max(15, 2 * d)),
        ]),
        "regressor": {
            "backend": "learned_features",
            "kernel": "RBF",
            "noise_level": 0.01,
            "length_scale_prior": [1e-2, 1e1],
            "n_restarts_optimizer": 1,
            "n_train_steps": 150,
        },
    }

    return configs


def run_one(make_fn, config_name, surrogate_cfg, max_finite=None, max_total=None):
    """Run a single benchmark with a given config."""
    logLkl, bounds, true_mean, name = make_fn()
    d = len(bounds)

    if max_finite is None:
        max_initial = int(30 * d ** 1.5)
        max_finite = max(200, max_initial + int(30 * d ** 1.2))
    if max_total is None:
        max_initial = int(30 * d ** 1.5)
        max_total = max(max_finite * 3, max_initial * 4)

    options = {"max_finite": max_finite, "max_total": max_total}

    result = run_benchmark(
        logLkl, bounds, name,
        surrogate_cfg=surrogate_cfg,
        options=options,
        verbose=0,
    )

    status = "CONV" if result["converged"] else "NCONV"
    n = str(result["n_total"]) if result["n_total"] is not None else "ERR"
    t = f"{result['time']:.1f}" if result["time"] is not None else "N/A"

    if result.get("error"):
        err_str = result["error"][:50]
    elif result.get("mean_mc") is not None and true_mean is not None:
        mean_err = np.max(np.abs(result["mean_mc"] - true_mean))
        err_str = f"{mean_err:.4f}"
    elif result.get("mean_mc") is not None:
        err_str = f"mean={np.array2string(result['mean_mc'], precision=2, max_line_width=40)}"
    else:
        err_str = "N/A"

    return {
        "benchmark": name,
        "config": config_name,
        "status": status,
        "n_total": n,
        "time": t,
        "error": err_str,
    }


if __name__ == "__main__":
    print("=" * 100, flush=True)
    print("  Approach Comparison: Baseline vs InputWarping vs LearnedFeatures vs Both", flush=True)
    print("=" * 100, flush=True)

    header = f"  {'Benchmark':<22} {'Config':<14} {'Status':<8} {'Evals':<7} {'Time':<10} {'Error':<30}"
    print(header, flush=True)
    print("  " + "-" * 95, flush=True)

    for make_fn in BENCHMARKS:
        logLkl, bounds, true_mean, name = make_fn()
        d = len(bounds)
        max_initial = int(30 * d ** 1.5)
        max_finite = max(200, max_initial + int(30 * d ** 1.2))
        # Cap budget to keep runtimes reasonable for comparison
        max_finite = min(max_finite, 500)
        max_total = max(max_finite * 3, max_initial * 4)

        configs = get_configs(bounds)
        for config_name, surrogate_cfg in configs.items():
            try:
                r = run_one(make_fn, config_name, surrogate_cfg,
                            max_finite=max_finite, max_total=max_total)
                row = (f"  {r['benchmark']:<22} {r['config']:<14} {r['status']:<8} "
                       f"{r['n_total']:<7} {r['time']:<10} {r['error']:<30}")
            except Exception as e:
                row = (f"  {name:<22} {config_name:<14} {'ERR':<8} "
                       f"{'---':<7} {'---':<10} {str(e)[:30]:<30}")
            print(row, flush=True)
        print("  " + "." * 95, flush=True)

    print("  " + "-" * 95, flush=True)
    print("  Done.", flush=True)
