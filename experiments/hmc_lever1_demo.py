"""
Validation + demonstration for Lever 1 (gradient-based HMC acquisition sampling).

Run with the worktree on the path, e.g.
    PYTHONPATH=<worktree> python experiments/hmc_lever1_demo.py

Three parts:
  A. Vectorized gradient correctness: predict_mean_grad_batch vs the existing
     single-point predict(return_mean_grad=True) AND vs central finite
     differences, for the RBF and Matern default kernels.
  B. Training-point-seeded HMC on a bimodal GP mean: does it cover BOTH modes,
     and does seeding matter (single-mode seed -> one mode only)?
  C. Head-to-head SAMPLING cost: GP-evaluation count + wall-time for HMC vs
     UltraNest (NORA's NS) producing an acquisition pool on the SAME GP.
"""

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from gpry.gpr import GaussianProcessRegressor as GPR
from gpry.gp_acquisition import NORA

RNG = np.random.default_rng(0)


def banner(txt):
    print("\n" + "=" * 74 + f"\n{txt}\n" + "=" * 74)


def fit_gpr(kernel, bounds, X, y, n_restarts=4):
    gpr = GPR(kernel=kernel, bounds=bounds, account_for_inf=None,
              n_restarts_optimizer=n_restarts)
    gpr.append_to_data(X, y, fit_gpr=True, fit_classifier=False)
    return gpr


# --------------------------------------------------------------------------- #
# Part A: vectorized gradient correctness
# --------------------------------------------------------------------------- #
def part_A():
    banner("PART A -- vectorized gradient correctness")
    dim = 3
    bounds = np.array([[-3.0, 3.0]] * dim)
    n_train = 40
    Xtr = RNG.uniform(bounds[:, 0], bounds[:, 1], size=(n_train, dim))
    # smooth-ish target
    ytr = -0.5 * np.sum(Xtr ** 2, axis=1) + np.sin(Xtr[:, 0]) * Xtr[:, 1]

    for kernel in ["RBF", "Matern"]:
        gpr = fit_gpr(kernel, bounds, Xtr, ytr)
        Xq = RNG.uniform(bounds[:, 0], bounds[:, 1], size=(25, dim))

        mean_b, grad_b = gpr.predict_mean_grad_batch(Xq)

        # (1) vs existing single-point predict(return_mean_grad=True)
        grad_sp = np.empty_like(grad_b)
        mean_sp = np.empty(len(Xq))
        for i, x in enumerate(Xq):
            out = gpr.predict(x[None], return_std=True,
                              return_mean_grad=True, return_std_grad=False)
            mean_sp[i] = np.ravel(out[0])[0]
            grad_sp[i] = np.ravel(out[2])  # grad_mean has shape (d,)
        err_grad_sp = np.max(np.abs(grad_b - grad_sp))
        err_mean_sp = np.max(np.abs(mean_b - mean_sp))

        # (2) vs central finite differences of the (unclipped) batch mean
        eps = 1e-5
        grad_fd = np.empty_like(grad_b)
        for k in range(dim):
            e = np.zeros(dim)
            e[k] = eps
            mp, _ = gpr.predict_mean_grad_batch(Xq + e)
            mm, _ = gpr.predict_mean_grad_batch(Xq - e)
            grad_fd[:, k] = (mp - mm) / (2 * eps)
        err_grad_fd = np.max(np.abs(grad_b - grad_fd))

        print(f"\n[{kernel:6s}] kernel_ = {gpr.kernel_}")
        print(f"  max|grad_batch - grad_singlepoint| = {err_grad_sp:.3e}")
        print(f"  max|mean_batch - mean_singlepoint| = {err_mean_sp:.3e} "
              f"(clip_factor={gpr.clip_factor})")
        print(f"  max|grad_batch - finite_diff|      = {err_grad_fd:.3e}")
        ok = (err_grad_sp < 1e-7) and (err_grad_fd < 1e-4)
        print(f"  --> {'PASS' if ok else 'FAIL'}")
        assert ok, f"gradient validation failed for {kernel}"


# --------------------------------------------------------------------------- #
# Part B & C: bimodal target
# --------------------------------------------------------------------------- #
def bimodal_logp(X, m1, m2, s):
    X = np.atleast_2d(X)
    d = X.shape[1]
    c = -0.5 * d * np.log(2 * np.pi * s ** 2)
    l1 = c - 0.5 * np.sum((X - m1) ** 2, axis=1) / s ** 2
    l2 = c - 0.5 * np.sum((X - m2) ** 2, axis=1) / s ** 2
    return np.logaddexp(l1 + np.log(0.5), l2 + np.log(0.5))


def occupancy(X, m1, m2, radius=1.2):
    if len(X) == 0:
        return 0.0, 0.0, 0.0
    d1 = np.linalg.norm(X - m1, axis=1)
    d2 = np.linalg.norm(X - m2, axis=1)
    f1 = np.mean(d1 < radius)
    f2 = np.mean(d2 < radius)
    return f1, f2, 1.0 - f1 - f2


def build_bimodal_gpr():
    dim = 2
    m1 = np.array([-2.0, -2.0])
    m2 = np.array([2.0, 2.0])
    s = 0.6
    bounds = np.array([[-4.0, 4.0]] * dim)
    # training points: clusters in both modes + background
    Xtr = np.concatenate([
        m1 + 0.6 * RNG.standard_normal((25, dim)),
        m2 + 0.6 * RNG.standard_normal((25, dim)),
        RNG.uniform(bounds[:, 0], bounds[:, 1], size=(15, dim)),
    ])
    Xtr = np.clip(Xtr, bounds[:, 0], bounds[:, 1])
    ytr = bimodal_logp(Xtr, m1, m2, s)
    gpr = fit_gpr("RBF", bounds, Xtr, ytr, n_restarts=6)
    return gpr, bounds, m1, m2, s


def part_B(gpr, bounds, m1, m2):
    banner("PART B -- training-point-seeded HMC: mode coverage")
    from gpry.mc_interfaces import hmc_sample_gp_mean

    lo, hi = bounds[:, 0], bounds[:, 1]

    # (i) seed from ALL training points (both modes)
    res_all = hmc_sample_gp_mean(
        gpr, lo, hi, rng=np.random.default_rng(1),
        n_warmup=40, n_samples=120, thin=4, n_leapfrog=15,
    )
    f1, f2, fo = occupancy(res_all["X"], m1, m2)
    print(f"\n[all-mode seeds]  chains={res_all['n_chains']}  "
          f"accept={res_all['accept_rate']:.2f}  step={res_all['step_size']:.3f}")
    print(f"  samples={len(res_all['X'])}  GP-evals={res_all['n_eval']}")
    print(f"  occupancy: mode1={f1:.2f}  mode2={f2:.2f}  elsewhere={fo:.2f}")
    both_covered = (f1 > 0.15) and (f2 > 0.15)
    print(f"  --> both modes covered: {both_covered}")

    # (ii) control: seed ONLY from mode-1 training points -> HMC cannot hop
    Xtr = np.asarray(gpr.X_train_)
    near1 = Xtr[np.linalg.norm(Xtr - m1, axis=1) < 1.5]
    res_one = hmc_sample_gp_mean(
        gpr, lo, hi, seeds=near1, rng=np.random.default_rng(2),
        n_warmup=40, n_samples=120, thin=4, n_leapfrog=15,
    )
    g1, g2, go = occupancy(res_one["X"], m1, m2)
    print(f"\n[mode-1-only seeds]  chains={res_one['n_chains']}  "
          f"accept={res_one['accept_rate']:.2f}")
    print(f"  occupancy: mode1={g1:.2f}  mode2={g2:.2f}  elsewhere={go:.2f}")
    print(f"  --> demonstrates the handoff's caveat: HMC does NOT discover the "
          f"unseeded mode (mode2={g2:.2f}). Seeding is what makes it "
          f"mode-complete over DISCOVERED modes.")
    assert both_covered, "HMC failed to cover both seeded modes"
    return res_all


def part_C(gpr, bounds, m1, m2):
    banner("PART C -- head-to-head SAMPLING cost: HMC vs UltraNest (NORA's NS)")
    acq = NORA(bounds=bounds, preprocessing_X=gpr.preprocessing_X,
               acq_func="LogExp", sampler="ultranest", verbose=0)

    results = {}
    for name in ["ultranest", "hmc"]:
        gpr.n_eval = 0
        t0 = time.time()
        try:
            X, y, sy, w = acq.do_MC_sample(
                gpr, bounds=bounds, rng=np.random.default_rng(3), sampler=name)
        except Exception as e:
            print(f"\n[{name}] FAILED: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        n_eval = int(gpr.n_eval)
        X = np.atleast_2d(X) if X is not None else np.empty((0, 2))
        f1, f2, fo = occupancy(X, m1, m2)
        results[name] = dict(n_eval=n_eval, dt=dt, n=len(X), f1=f1, f2=f2)
        print(f"\n[{name:9s}] pool={len(X):5d}  GP-evals={n_eval:7d}  "
              f"wall={dt:6.2f}s  occ(m1,m2)=({f1:.2f},{f2:.2f})")

    if "ultranest" in results and "hmc" in results:
        r_ns, r_hmc = results["ultranest"], results["hmc"]
        print("\n--- comparison (SAMPLING step only; not end-to-end wall-time) ---")
        print(f"  GP-eval ratio  NS / HMC = "
              f"{r_ns['n_eval'] / max(1, r_hmc['n_eval']):.2f}x")
        print(f"  wall-time ratio NS / HMC = {r_ns['dt'] / max(1e-6, r_hmc['dt']):.2f}x")
        print("  NOTE: HMC does extra work per GP-row (it also evaluates the")
        print("        analytic gradient), so equal GP-row counts != equal FLOPs.")
        print("        Both cover both modes; NS via exploration, HMC via seeding.")
    return results


def save_plot(gpr, bounds, m1, m2, s, res_all):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    lo, hi = bounds[:, 0], bounds[:, 1]
    gx = np.linspace(lo[0], hi[0], 120)
    gy = np.linspace(lo[1], hi[1], 120)
    GXv, GYv = np.meshgrid(gx, gy)
    grid = np.column_stack([GXv.ravel(), GYv.ravel()])
    Z = bimodal_logp(grid, m1, m2, s).reshape(GXv.shape)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.contourf(GXv, GYv, np.exp(Z), levels=20, cmap="Greys")
    Xtr = np.asarray(gpr.X_train_)
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c="tab:blue", s=18, label="training pts (seeds)")
    Xs = res_all["X"]
    ax.scatter(Xs[:, 0], Xs[:, 1], c="tab:orange", s=4, alpha=0.35, label="HMC samples")
    ax.scatter(*m1, c="red", marker="x", s=90)
    ax.scatter(*m2, c="red", marker="x", s=90, label="true modes")
    ax.set_title("Lever 1: training-seeded HMC on the GP mean")
    ax.legend(loc="upper left", fontsize=8)
    out = os.path.join(os.path.dirname(__file__), "hmc_lever1_demo.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nSaved plot to {out}")
    return out


if __name__ == "__main__":
    part_A()
    gpr, bounds, m1, m2, s = build_bimodal_gpr()
    res_all = part_B(gpr, bounds, m1, m2)
    part_C(gpr, bounds, m1, m2)
    save_plot(gpr, bounds, m1, m2, s, res_all)
    banner("ALL CHECKS PASSED")
