"""
Compare GPry performance: baseline vs new features on benchmark suite.

Runs each benchmark with multiple configurations to assess which
improvements help and which hurt for different likelihood shapes.

Configs tested:
1. baseline       — default settings
2. zeta_ramp      — zeta scheduling (exploration phase)
3. turbo          — TuRBO trust regions
4. active_sub     — ActiveSubspace rotation
5. kitchen_sink   — all features combined
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_benchmarks import (
    make_gaussian_mixture_2d,
    make_gaussian_mixture_5d,
    make_eggbox_2d,
    make_gaussian_shells_2d,
    make_gaussian_shells_5d,
    make_banana_2d,
    make_banana_5d,
    make_cigar_10d,
    make_neals_funnel_10d,
    make_rastrigin_2d,
    make_sn_cosmology_2d,
    run_benchmark,
)
from gpry.preprocessing import SoftClipY, PipelineY, NormalizeY


def make_options(ndim):
    budget = max(200, int(50 * ndim**1.5))
    return {"max_finite": budget, "max_total": budget * 3}


def make_acq_with_ramp(ndim):
    """Build acquisition config dict with zeta scheduling."""
    return {
        "BatchOptimizer": {
            "acq_func": {
                "LogExp": {
                    "zeta_schedule": "ramp",
                    "n_explore": "3d",
                    "zeta_scaling": 0.85,
                }
            }
        }
    }


def make_configs(ndim):
    """Return list of (name, surrogate_extra, acq_cfg) tuples."""
    return [
        # 1. Baseline
        ("baseline", {}, None),
        # 2. Zeta scheduling only
        ("zeta_ramp", {}, make_acq_with_ramp(ndim)),
        # 3. TuRBO trust regions only
        ("turbo", {"turbo": True}, None),
        # 4. ActiveSubspace rotation only (via preprocessing_X not yet wired —
        #    would need PipelineX integration; skip for now, test directly)
        # 5. Zeta + TuRBO
        ("zeta+turbo", {"turbo": True}, make_acq_with_ramp(ndim)),
        # 6. Kitchen sink: zeta + turbo + adaptive noise + noise floor
        ("kitchen_sink", {
            "turbo": True,
            "adaptive_noise": True,
            "noise_floor": True,
        }, make_acq_with_ramp(ndim)),
    ]


if __name__ == "__main__":
    # Select benchmarks. 10D ones are slow, so they're at the end.
    benchmarks = [
        ("GMM 2D", make_gaussian_mixture_2d),
        ("Banana 2D", make_banana_2d),
        ("Shells 2D", make_gaussian_shells_2d),
        ("Eggbox 2D", make_eggbox_2d),
        ("SN Cosmo", make_sn_cosmology_2d),
        ("GMM 5D", make_gaussian_mixture_5d),
        ("Banana 5D", make_banana_5d),
        ("Shells 5D", make_gaussian_shells_5d),
        ("Cigar 10D", make_cigar_10d),
        # ("Funnel 10D", make_neals_funnel_10d),  # very hard, uncomment to test
    ]

    # Allow selecting specific benchmarks via command line
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
        benchmarks = [(n, f) for n, f in benchmarks
                      if any(s.lower() in n.lower() for s in selected)]

    print("=" * 100)
    print("  GPry Benchmark Comparison: Baseline vs Improvements")
    print("=" * 100)

    for bench_name, make_fn in benchmarks:
        logLkl, bounds, true_mean, desc = make_fn()
        ndim = len(bounds)
        options = make_options(ndim)
        configs = make_configs(ndim)

        print(f"\n{'─' * 100}")
        print(f"  {desc} (d={ndim}, budget={options['max_finite']})")
        print(f"{'─' * 100}")
        print(f"  {'Config':<15} {'Status':<8} {'Evals':<7} {'Time':<8} {'Mean Err':<12} {'Notes'}")
        print(f"  {'-'*15} {'-'*8} {'-'*7} {'-'*8} {'-'*12} {'-'*30}")

        for cfg_name, extra_surr, acq_cfg in configs:
            surrogate_cfg = {"regressor": {"kernel": "RBF", "use_jax": False}}
            surrogate_cfg.update(extra_surr)

            r = run_benchmark(
                logLkl, bounds, desc,
                surrogate_cfg=surrogate_cfg,
                options=options,
                gp_acquisition=acq_cfg,
                verbose=0, seed=42,
            )

            status = "CONV" if r.get("converged") else "NCONV"
            n = str(r.get("n_total", "?")) if r.get("n_total") is not None else "ERR"
            t = f"{r['time']:.1f}s" if r.get("time") is not None else "N/A"

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

            print(f"  {cfg_name:<15} {status:<8} {n:<7} {t:<8} {err_str:<12} {notes}")

    print(f"\n{'=' * 100}")
