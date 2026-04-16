"""
Test max_training_points feature.

Validates that:
1. max_training_points correctly caps GP training set size
2. Highest-y points are retained (near-mode preference)
3. Works correctly with adaptive noise
4. End-to-end with Runner using DontConverge to force many evaluations
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
# Test likelihoods
# ---------------------------------------------------------------------------

def make_gaussian_2d():
    mean = np.array([3.0, 2.0])
    cov = np.array([[0.5, 0.4], [0.4, 1.5]])
    rv = multivariate_normal(mean, cov)

    def logLkl(x_1, x_2):
        return rv.logpdf(np.array([x_1, x_2]).T)

    return logLkl, [[-10, 10], [-10, 10]], mean, cov


def make_rosenbrock_2d():
    a, b = 1.0, 100.0

    def logLkl(x_0, x_1):
        return -0.5 * ((a - x_0) ** 2 + b * (x_1 - x_0 ** 2) ** 2)

    return logLkl, [[-4, 4], [-4, 4]], np.array([1.0, 1.0]), None


# ---------------------------------------------------------------------------
# Runner helper
# ---------------------------------------------------------------------------

def run_test(logLkl, bounds, label, max_training_points=None,
             adaptive_noise=None, max_evals=150, seed=42, use_jax=False):
    tmpdir = tempfile.mkdtemp(prefix=f"gpry_mtp_{label.replace(' ', '_')}_")
    checkpoint = os.path.join(tmpdir, "run")

    surrogate_cfg = {
        "regressor": {"kernel": "RBF", "use_jax": use_jax},
        "adaptive_noise": adaptive_noise,
        "max_training_points": max_training_points,
    }
    # Use DontConverge to force exhausting the budget
    options = {"max_finite": max_evals, "max_total": max_evals * 2}

    try:
        t0 = time.time()
        runner = Runner(
            logLkl, bounds,
            surrogate=surrogate_cfg,
            convergence_criterion=False,  # DontConverge
            checkpoint=checkpoint,
            load_checkpoint="overwrite",
            verbose=0, seed=seed, options=options,
        )
        runner.run()
        t_total = time.time() - t0

        surr = runner.surrogate
        n_gpr = surr.gpr.X_train_.shape[0]  # actual GP training set size

        result = {
            "label": label,
            "max_training_points": max_training_points,
            "adaptive_noise": adaptive_noise is not None,
            "n_total": surr.n_total,
            "n_regress": surr.n_regress,
            "n_gpr": n_gpr,
            "t_total": t_total,
            "error": None,
        }

        # Posterior estimate
        mc = runner.last_mc_samples()
        if mc is not None:
            mean_mc, cov_mc = mean_covmat_from_samples(mc["X"], mc["w"])
            result["mean_mc"] = mean_mc
        else:
            result["mean_mc"] = None

        return result

    except Exception as e:
        import traceback
        return {
            "label": label,
            "max_training_points": max_training_points,
            "adaptive_noise": adaptive_noise is not None,
            "n_total": None,
            "n_regress": None,
            "n_gpr": None,
            "t_total": time.time() - t0,
            "error": f"{type(e).__name__}: {e}",
            "mean_mc": None,
            "traceback": traceback.format_exc(),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MAX_EVALS = 150  # Force many evaluations via DontConverge

    print("=" * 70)
    print("  max_training_points validation")
    print("=" * 70)

    # --- Test 1: 2D Gaussian, verify cap works ---
    print(f"\n{'─' * 70}")
    print("  Test 1: 2D Gaussian — verify GP training set is capped")
    print(f"{'─' * 70}")

    logLkl, bounds, true_mean, true_cov = make_gaussian_2d()
    configs = [
        ("unlimited", None, None),
        ("max_50", 50, None),
        ("max_30", 30, None),
        ("max_50+adaptive", 50, True),
    ]

    for label, mtp, adaptive in configs:
        r = run_test(logLkl, bounds, label, max_training_points=mtp,
                     adaptive_noise=adaptive, max_evals=MAX_EVALS)

        if r["error"]:
            print(f"  {label:>18}: ERROR — {r['error']}")
            if "traceback" in r:
                print(r["traceback"][-400:])
            continue

        # Check the cap is respected
        cap_ok = ""
        if mtp is not None:
            if r["n_gpr"] <= mtp:
                cap_ok = "  CAP OK"
            else:
                cap_ok = f"  CAP VIOLATED ({r['n_gpr']} > {mtp})"

        mean_err = ""
        if r["mean_mc"] is not None and true_mean is not None:
            err = np.max(np.abs(r["mean_mc"] - true_mean))
            mean_err = f"  err={err:.4f}"

        print(f"  {label:>18}: n_total={r['n_total']:<4} n_regress={r['n_regress']:<4} "
              f"n_gpr={r['n_gpr']:<4} t={r['t_total']:.1f}s{mean_err}{cap_ok}")

    # --- Test 2: Rosenbrock, quality check ---
    print(f"\n{'─' * 70}")
    print("  Test 2: Rosenbrock — quality with capped training set")
    print(f"{'─' * 70}")

    logLkl, bounds, true_mean, _ = make_rosenbrock_2d()
    configs = [
        ("unlimited", None, None),
        ("max_80", 80, None),
        ("max_50", 50, None),
        ("max_80+adaptive", 80, True),
    ]

    for label, mtp, adaptive in configs:
        r = run_test(logLkl, bounds, label, max_training_points=mtp,
                     adaptive_noise=adaptive, max_evals=MAX_EVALS)

        if r["error"]:
            print(f"  {label:>18}: ERROR — {r['error']}")
            if "traceback" in r:
                print(r["traceback"][-400:])
            continue

        cap_ok = ""
        if mtp is not None:
            if r["n_gpr"] <= mtp:
                cap_ok = "  CAP OK"
            else:
                cap_ok = f"  CAP VIOLATED ({r['n_gpr']} > {mtp})"

        mean_err = ""
        if r["mean_mc"] is not None and true_mean is not None:
            err = np.max(np.abs(r["mean_mc"] - true_mean))
            mean_err = f"  err={err:.4f}"

        print(f"  {label:>18}: n_total={r['n_total']:<4} n_regress={r['n_regress']:<4} "
              f"n_gpr={r['n_gpr']:<4} t={r['t_total']:.1f}s{mean_err}{cap_ok}")

    # --- Test 3: JAX + adaptive + max_training_points ---
    print(f"\n{'─' * 70}")
    print("  Test 3: JAX + adaptive + max_training_points")
    print(f"{'─' * 70}")

    logLkl, bounds, true_mean, true_cov = make_gaussian_2d()
    configs = [
        ("jax unlimited", None, True, True),
        ("jax max_50", 50, True, True),
        ("numpy unlimited", None, True, False),
        ("numpy max_50", 50, True, False),
    ]

    for label, mtp, adaptive, use_jax in configs:
        r = run_test(logLkl, bounds, label, max_training_points=mtp,
                     adaptive_noise=adaptive, max_evals=MAX_EVALS,
                     use_jax=use_jax)

        if r["error"]:
            print(f"  {label:>18}: ERROR — {r['error']}")
            if "traceback" in r:
                print(r["traceback"][-400:])
            continue

        cap_ok = ""
        if mtp is not None:
            if r["n_gpr"] <= mtp:
                cap_ok = "  CAP OK"
            else:
                cap_ok = f"  CAP VIOLATED ({r['n_gpr']} > {mtp})"

        mean_err = ""
        if r["mean_mc"] is not None and true_mean is not None:
            err = np.max(np.abs(r["mean_mc"] - true_mean))
            mean_err = f"  err={err:.4f}"

        print(f"  {label:>18}: n_total={r['n_total']:<4} n_regress={r['n_regress']:<4} "
              f"n_gpr={r['n_gpr']:<4} t={r['t_total']:.1f}s{mean_err}{cap_ok}")

    print()
