"""
Compare GP backends on benchmark suite.

Tests standard GP vs RandomFeatureGP, GibbsGPR, and the
HyperparameterScheduler across benchmark likelihoods spanning multimodal,
non-stationary, ill-conditioned, and real-world-inspired problems.

Also includes additional hard likelihoods not in the original benchmark suite:
- Double banana (two curved degeneracies)
- Spike + plateau (narrow spike inside broad background — the "snap" problem)
- Warped Rosenbrock 6D
- Levy function 4D (multimodal with irregular basin sizes)

Usage:
    python test_backend_comparison.py [benchmark_filter ...] [--backends b1,b2,...]

Examples:
    python test_backend_comparison.py                # all benchmarks, all backends
    python test_backend_comparison.py banana gmm     # filter benchmarks
    python test_backend_comparison.py --backends standard,gibbs  # filter backends
"""

import sys
import os
import time
import tempfile
import shutil
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_benchmarks import (
    ALL_BENCHMARKS,
    run_benchmark,
)


# ===========================================================================
# Additional hard likelihoods
# ===========================================================================

def make_double_banana_2d():
    """Two banana-shaped modes pointing in opposite directions.

    Tests whether non-stationary backends can handle two curved degeneracies
    with different orientations — a case where the GP needs very different
    length scales in different regions.
    """
    b = 0.1
    sigma1_sq, sigma2_sq = 100.0, 1.0

    def logLkl(x_0, x_1):
        # Banana mode 1: centered at origin
        x2_tw1 = x_1 - b * (x_0 ** 2 - 100.0)
        lp1 = -0.5 * (x_0 ** 2 / sigma1_sq + x2_tw1 ** 2 / sigma2_sq)
        # Banana mode 2: reflected and shifted
        x2_tw2 = -x_1 - b * (x_0 ** 2 - 100.0) + 15.0
        lp2 = -0.5 * (x_0 ** 2 / sigma1_sq + x2_tw2 ** 2 / sigma2_sq) - 1.0
        return logsumexp([lp1, lp2])

    bounds = [[-30, 30], [-30, 30]]
    return logLkl, bounds, None, "Double Banana 2D"


def make_spike_plateau_3d():
    """Broad Gaussian plateau with a narrow spike at the origin.

    This is the "terrible → snap to mode" problem: the GP first sees the
    broad plateau and models it as flat. The spike is much narrower and
    higher, requiring the GP to dramatically adjust its length scales
    once it finds it. Non-stationary kernels (Gibbs, DKL) should help here.
    """
    # Broad plateau
    cov_broad = np.diag([25.0, 25.0, 25.0])
    rv_broad = multivariate_normal(np.zeros(3), cov_broad)
    # Narrow spike (10x narrower, 5x taller in log space)
    cov_spike = np.diag([0.25, 0.25, 0.25])
    rv_spike = multivariate_normal(np.zeros(3), cov_spike)

    def logLkl(x_0, x_1, x_2):
        pt = np.array([x_0, x_1, x_2])
        # The spike is much more peaked
        lp_broad = rv_broad.logpdf(pt) + np.log(0.3)
        lp_spike = rv_spike.logpdf(pt) + np.log(0.7) + 5.0
        return logsumexp([lp_broad, lp_spike])

    bounds = [[-15, 15], [-15, 15], [-15, 15]]
    true_mean = np.zeros(3)
    return logLkl, bounds, true_mean, "Spike+Plateau 3D"


def make_warped_rosenbrock_6d():
    """6D Rosenbrock (banana valley) — a standard hard benchmark.

    The Rosenbrock function has a narrow curved valley that becomes
    increasingly hard to fit with stationary kernels as d grows.
    """
    def logLkl(x_0, x_1, x_2, x_3, x_4, x_5):
        xs = np.array([x_0, x_1, x_2, x_3, x_4, x_5])
        val = 0.0
        for i in range(5):
            val += 100.0 * (xs[i + 1] - xs[i] ** 2) ** 2 + (1.0 - xs[i]) ** 2
        # Scale so it's a reasonable log-likelihood
        return -val / 50.0

    bounds = [[-3, 3]] * 6
    true_mean = np.ones(6)  # global minimum at (1,1,...,1)
    return logLkl, bounds, true_mean, "Rosenbrock 6D"


def make_levy_4d():
    """4D Levy function — multimodal with irregular basin sizes.

    Has a global minimum at (1,1,1,1) but many local minima with different
    basin sizes, making it hard for stationary GPs that assume uniform
    length scales everywhere.
    """
    def logLkl(x_0, x_1, x_2, x_3):
        xs = np.array([x_0, x_1, x_2, x_3])
        w = 1.0 + (xs - 1.0) / 4.0
        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum(
            (w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2)
        )
        term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
        val = term1 + term2 + term3
        return -val

    bounds = [[-5, 5]] * 4
    true_mean = np.ones(4)
    return logLkl, bounds, true_mean, "Levy 4D"


def make_mixture_of_widths_5d():
    """5D mixture with components having very different widths.

    One tight mode (sigma ~ 0.3) and one broad mode (sigma ~ 3.0).
    This directly tests the non-stationary kernel advantage: a stationary
    kernel must compromise between the two scales.
    """
    mean_tight = np.array([2.0, 2.0, 0.0, 0.0, 0.0])
    mean_broad = np.array([-2.0, -2.0, 0.0, 0.0, 0.0])
    cov_tight = np.diag([0.09, 0.09, 0.09, 0.09, 0.09])  # sigma=0.3
    cov_broad = np.diag([9.0, 9.0, 9.0, 9.0, 9.0])        # sigma=3.0
    rv_tight = multivariate_normal(mean_tight, cov_tight)
    rv_broad = multivariate_normal(mean_broad, cov_broad)

    def logLkl(x_0, x_1, x_2, x_3, x_4):
        pt = np.array([x_0, x_1, x_2, x_3, x_4])
        return logsumexp([
            np.log(0.5) + rv_tight.logpdf(pt),
            np.log(0.5) + rv_broad.logpdf(pt),
        ])

    bounds = [[-10, 10]] * 5
    true_mean = (mean_tight + mean_broad) / 2.0
    return logLkl, bounds, true_mean, "Mixed Widths 5D"


# ===========================================================================
# Extended benchmark registry
# ===========================================================================

EXTRA_BENCHMARKS = {
    "double_banana_2d": make_double_banana_2d,
    "spike_plateau_3d": make_spike_plateau_3d,
    "rosenbrock_6d": make_warped_rosenbrock_6d,
    "levy_4d": make_levy_4d,
    "mixed_widths_5d": make_mixture_of_widths_5d,
}

FULL_BENCHMARKS = {**ALL_BENCHMARKS, **EXTRA_BENCHMARKS}


# ===========================================================================
# Backend configurations
# ===========================================================================

def make_backend_configs(ndim):
    """Return list of (name, surrogate_cfg, options_extra) tuples for each backend."""
    budget = max(200, int(50 * ndim ** 1.5))
    base_options = {"max_finite": budget, "max_total": budget * 3}

    configs = [
        # 1. Standard GP (baseline)
        ("standard", {
            "regressor": {"kernel": "RBF", "use_jax": False},
        }, base_options),

        # 2. Random Feature GP
        ("rfgp", {
            "regressor": {
                "backend": "random_features",
                "kernel": "RBF",
                "n_restarts_optimizer": 3,
            },
        }, base_options),

        # 3. Gibbs non-stationary kernel
        ("gibbs", {
            "regressor": {
                "backend": "gibbs",
                "kernel": "RBF",
                "n_restarts_optimizer": 3,
            },
        }, base_options),

        # 4. Standard + HyperparameterScheduler
        ("std+scheduler", {
            "regressor": {"kernel": "RBF", "use_jax": False},
        }, {**base_options, "hyperopt_scheduler": {
            "lml_tol": 0.5,
            "restart_decay": 0.8,
            "decay_period": 3,
        }}),

        # 5. Gibbs + scheduler (the expected best combo for non-stationary)
        ("gibbs+sched", {
            "regressor": {
                "backend": "gibbs",
                "kernel": "RBF",
                "n_restarts_optimizer": 3,
            },
        }, {**base_options, "hyperopt_scheduler": {
            "lml_tol": 0.5,
            "restart_decay": 0.8,
        }}),
    ]

    return configs


# ===========================================================================
# Main comparison driver
# ===========================================================================

def run_comparison(benchmark_filter=None, backend_filter=None, verbose=0):
    """Run all backends on all benchmarks and print comparison table."""

    # Select benchmarks
    benchmarks = list(FULL_BENCHMARKS.items())
    if benchmark_filter:
        benchmarks = [(k, f) for k, f in benchmarks
                      if any(s.lower() in k.lower() for s in benchmark_filter)]

    # Collect all results for summary
    all_results = {}

    print("=" * 110)
    print("  GPry Backend Comparison")
    print("=" * 110)

    for bench_key, make_fn in benchmarks:
        logLkl, bounds, true_mean, desc = make_fn()
        ndim = len(bounds)
        configs = make_backend_configs(ndim)

        # Filter backends
        if backend_filter:
            configs = [(n, s, o) for n, s, o in configs
                       if any(b.lower() in n.lower() for b in backend_filter)]

        print(f"\n{'─' * 110}")
        print(f"  {desc} (d={ndim})")
        print(f"{'─' * 110}")
        header = (f"  {'Backend':<16} {'Status':<7} {'Evals':<6} "
                  f"{'Time (s)':<10} {'Mean Err':<12} {'Notes'}")
        print(header)
        print(f"  {'-' * 16} {'-' * 7} {'-' * 6} {'-' * 10} {'-' * 12} {'-' * 30}")

        bench_results = {}

        for cfg_name, surrogate_cfg, options in configs:
            r = run_benchmark(
                logLkl, bounds, desc,
                surrogate_cfg=surrogate_cfg,
                options=options,
                verbose=verbose, seed=42,
            )

            status = "CONV" if r.get("converged") else "NCONV"
            if r.get("error"):
                status = "ERR"
            n = str(r.get("n_total", "?")) if r.get("n_total") is not None else "?"
            t = f"{r['time']:.1f}" if r.get("time") is not None else "N/A"

            if r.get("mean_mc") is not None and true_mean is not None:
                err = np.max(np.abs(r["mean_mc"] - true_mean))
                err_str = f"{err:.4f}"
            elif r.get("mean_mc") is not None:
                err_str = f"~{np.array2string(r['mean_mc'], precision=2)}"
            else:
                err_str = "N/A"

            notes = ""
            if r.get("error"):
                notes = r["error"][:50]

            row = (f"  {cfg_name:<16} {status:<7} {n:<6} "
                   f"{t:<10} {err_str:<12} {notes}")
            print(row)

            bench_results[cfg_name] = r

        all_results[bench_key] = bench_results

    # ===========================================================================
    # Summary table
    # ===========================================================================
    print(f"\n\n{'=' * 110}")
    print("  SUMMARY: Convergence rate and mean error by backend")
    print(f"{'=' * 110}")

    # Collect backend names from first benchmark
    if all_results:
        first_bench = next(iter(all_results.values()))
        backend_names = list(first_bench.keys())
    else:
        backend_names = []

    header = f"  {'Benchmark':<22} " + " ".join(f"{b:<16}" for b in backend_names)
    print(header)
    print("  " + "-" * (22 + 17 * len(backend_names)))

    for bench_key, bench_results in all_results.items():
        logLkl, bounds, true_mean, desc = FULL_BENCHMARKS[bench_key]()
        row = f"  {desc:<22} "
        for bn in backend_names:
            r = bench_results.get(bn)
            if r is None:
                row += f"{'---':<16} "
                continue
            if r.get("error"):
                cell = "ERR"
            elif r.get("converged"):
                t = r["time"]
                if r.get("mean_mc") is not None and true_mean is not None:
                    err = np.max(np.abs(r["mean_mc"] - true_mean))
                    cell = f"OK {t:.0f}s e={err:.3f}"
                else:
                    cell = f"OK {t:.0f}s"
            else:
                cell = f"NC {r.get('time', 0):.0f}s"
            row += f"{cell:<16} "
        print(row)

    print(f"{'=' * 110}")
    return all_results


if __name__ == "__main__":
    args = sys.argv[1:]

    # Parse --backends flag
    backend_filter = None
    benchmark_filter = []
    i = 0
    while i < len(args):
        if args[i] == "--backends" and i + 1 < len(args):
            backend_filter = args[i + 1].split(",")
            i += 2
        else:
            benchmark_filter.append(args[i])
            i += 1

    if not benchmark_filter:
        benchmark_filter = None

    run_comparison(
        benchmark_filter=benchmark_filter,
        backend_filter=backend_filter,
        verbose=0,
    )
