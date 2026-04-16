"""
Test input warping on various likelihoods: simple Gaussian, Rosenbrock,
and a log-scale parameter likelihood.

Compares convergence and accuracy with and without InputWarping.
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
from gpry.preprocessing import NormalizeBounds, InputWarping, PipelineX


# ---------------------------------------------------------------------------
# Test likelihoods (no cobaya dependency)
# ---------------------------------------------------------------------------

def make_gaussian_2d():
    """Simple 2D Gaussian — warping should NOT help here."""
    mean = np.array([3.0, 2.0])
    cov = np.array([[0.5, 0.4], [0.4, 1.5]])
    rv = multivariate_normal(mean, cov)

    def logLkl(x_1, x_2):
        return rv.logpdf(np.array([x_1, x_2]).T)

    return logLkl, [[-10, 10], [-10, 10]], mean, "Gaussian 2D"


def make_rosenbrock_2d():
    """Rosenbrock function — curved banana shape, warping may help."""
    a, b = 1.0, 100.0

    def logLkl(x_0, x_1):
        return -0.5 * ((a - x_0) ** 2 + b * (x_1 - x_0 ** 2) ** 2)

    # Mode at (1, 1)
    return logLkl, [[-4, 4], [-4, 4]], np.array([1.0, 1.0]), "Rosenbrock 2D"


def make_loggaussian_2d():
    """Log-scale Gaussian — first parameter is in log10 scale.
    This is strongly non-stationary and warping SHOULD help."""
    mean_log = np.array([0.5, 2.0])  # mean in log10 space for dim 0
    cov = np.array([[0.3, 0.0], [0.0, 0.5]])
    rv = multivariate_normal(mean_log, cov)

    def logLkl(x_0, x_1):
        # x_0 is in log10 scale, x_1 is linear
        return rv.logpdf(np.array([10 ** x_0, x_1]).T)

    return logLkl, [[-2, 2], [-5, 5]], None, "LogGaussian 2D"


def make_himmelblau_2d():
    """Himmelblau function — 4 local minima, multimodal."""
    a, b = 11.0, 7.0

    def logLkl(x_0, x_1):
        return -0.5 * ((x_0 ** 2 + x_1 - a) ** 2 + (x_0 + x_1 ** 2 - b) ** 2)

    return logLkl, [[-5, 5], [-5, 5]], None, "Himmelblau 2D"


# ---------------------------------------------------------------------------
# Runner helper
# ---------------------------------------------------------------------------

def run_test(logLkl, bounds, label, use_warping, seed=42, max_evals=200,
             concentration=0.3):
    """Run GPry with or without input warping."""
    tmpdir = tempfile.mkdtemp(prefix=f"gpry_warp_{label.replace(' ', '_')}_")
    checkpoint = os.path.join(tmpdir, "run")

    bounds_arr = np.array(bounds)
    if use_warping:
        preproc = PipelineX([
            NormalizeBounds(bounds_arr),
            InputWarping(concentration=concentration, min_points=15),
        ])
    else:
        preproc = NormalizeBounds(bounds_arr)

    surrogate_cfg = {
        "regressor": {"kernel": "RBF", "use_jax": False},
        "preprocessing_X": preproc,
    }
    options = {"max_finite": max_evals, "max_total": max_evals * 2}

    try:
        t0 = time.time()
        runner = Runner(
            logLkl, bounds,
            surrogate=surrogate_cfg,
            checkpoint=checkpoint,
            load_checkpoint="overwrite",
            verbose=0, seed=seed, options=options,
        )
        runner.run()
        t_total = time.time() - t0

        result = {
            "label": label,
            "warping": use_warping,
            "converged": runner.has_converged,
            "n_total": runner.surrogate.n_total,
            "t_total": t_total,
        }

        mc = runner.last_mc_samples()
        if mc is not None:
            mean_mc, cov_mc = mean_covmat_from_samples(mc["X"], mc["w"])
            result["mean_mc"] = mean_mc
        else:
            result["mean_mc"] = None

        # Get warping params if available
        if hasattr(runner.surrogate.preprocessing_X, 'preprocessors'):
            for p in runner.surrogate.preprocessing_X.preprocessors:
                if hasattr(p, '_a'):
                    result["warp_a"] = p._a.copy()
                    result["warp_b"] = p._b.copy()
                    result["warp_activated"] = p._activated
                    result["warp_frozen"] = p._frozen

        return result

    except Exception as e:
        import traceback
        return {
            "label": label, "warping": use_warping,
            "converged": False, "n_total": None,
            "t_total": time.time() - t0, "mean_mc": None,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        make_gaussian_2d(),
        make_rosenbrock_2d(),
        make_loggaussian_2d(),
        make_himmelblau_2d(),
    ]

    print("=" * 70)
    print("  Input Warping Comparison: with vs without")
    print("=" * 70)

    for logLkl, bounds, true_mean, name in test_cases:
        print(f"\n{'─' * 70}")
        print(f"  {name}")
        print(f"{'─' * 70}")

        for use_warping in [False, True]:
            warp_label = "warped" if use_warping else "plain"
            r = run_test(logLkl, bounds, name, use_warping, max_evals=150)

            status = "CONV" if r["converged"] else "NCONV"
            n = r["n_total"] if r["n_total"] is not None else "ERR"

            line = f"  {warp_label:>7}: {status:>5}  evals={n:<4}  time={r['t_total']:.1f}s"

            if r.get("mean_mc") is not None and true_mean is not None:
                mean_err = np.max(np.abs(r["mean_mc"] - true_mean))
                line += f"  mean_err={mean_err:.4f}"
            elif r.get("mean_mc") is not None:
                line += f"  mean={np.array2string(r['mean_mc'], precision=3)}"

            if "warp_a" in r:
                line += f"  a={np.array2string(r['warp_a'], precision=2)}"
                line += f" b={np.array2string(r['warp_b'], precision=2)}"
                line += f"  act={'Y' if r.get('warp_activated') else 'N'}"

            if "error" in r:
                line += f"  ERROR: {r['error'][:60]}"
                if "traceback" in r:
                    print(r["traceback"][-500:])

            print(line)

    print()
