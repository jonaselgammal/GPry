# Restart-efficiency study

## Question

The GP hyperparameter fit dominates the acquisition loop at high dimension:

| run (v3, NUTS) | loop | fit | fit share |
|---|---|---|---|
| d=16 | 444 s | 231 s | **52%** |
| d=30 | 11822 s | 10009 s | **85%** |

A bench measurement on a real d=16 surrogate found that *every* restart
strategy converges to the same optimum (LML = 460.2, ls = 36.51), and that the
two informed starts do all the work:

| n_restarts | time | LML evals | LML |
|---|---|---|---|
| 34 (default) | 28.1 s | 4778 | 460.2 |
| 2 (informed only) | **0.6 s** | 94 | 460.2 |

47x faster for an identical fit. This study asks whether that survives across
dimension, and on targets where a wrong length scale actually breaks the
posterior rather than only on easy Gaussians.

## Arms

| arm | `restart_strategy` | `n_restarts_optimizer` |
|---|---|---|
| S0 | uniform | 10 + 2d (current default) — **control** |
| S1 | uniform | 8 |
| S2 | uniform | 2 (informed starts only) |
| S3 | local | 8 |
| S4 | screen | 8 |

`local` draws log-normal around the covariance-based guess; `screen` draws 8x
and ranks by a single gradient-free LML, keeping the best.

## Targets

d=8/16/30 use the anisotropic Gaussian (speed/scaling). **d=5 is deliberately
non-Gaussian** so the study measures robustness, not just speed.

GPry's own `Curved_degeneracy` and `Ring` are 2d-only, and its n-dim
`Himmelblau` loops `range(0, len(X)//2, 2)`, which silently leaves dimensions
unconstrained — so both d=5 targets are built from scratch in `common.py`.
Both have exact closed-form samples, so the truth side never needs a sampler.

- **`make_curved`** — chained twisted Gaussian: `x_i = z_i + b(z_{i-1}^2 - s_{i-1}^2)`.
  The twist is *centred*, so the target has a **diagonal covariance**: the
  coordinates are linearly uncorrelated but strongly dependent. A surrogate
  that collapses to a Gaussian blob therefore scores *perfectly* on every
  moment metric. `evaluate_recovery_curved` scores in the base coordinates
  instead (exactly Gaussian under the truth), trimmed at |z/s| <= 8 because the
  inverse twist is a chained quadratic that amplifies tail error (exact samples
  reach |z/s| ~ 5; a wrong posterior reaches ~1e7). A bounded curvature
  correlation is reported alongside.
- **`make_multimode`** — K unit modes on mutually orthogonal directions, so
  *every* dimension separates the modes.

## Difficulty calibration (control arm, S0)

A target the control cannot solve cannot discriminate between restart budgets,
so both d=5 targets were calibrated against S0 before the study ran. Both
first attempts failed, and fixing them took two rounds.

### Banana

| config | control result | verdict |
|---|---|---|
| chained (n_twist=4), b=0.5 | KL_z 2.38, no convergence in 600 pts | too hard |
| chained, b=0.35 | KL_z 1.79, no convergence in 600 pts | too hard |
| chained, b=0.2 | KL_z 0.279 — but a Gaussian **blob** scores 0.036 | useless |
| **single bend (n_twist=1), b=1.2** | **KL_z 0.0017 / 0.0012 / 0.0026** | **used** |

The chained twist had *no usable window*: weak enough to solve meant nearly
Gaussian (the blob beat the GP 8x), strong enough to be interesting meant
unsolvable. Chaining compounds the curvature across every coordinate. The
standard single-bend banana at b=1.2 is strongly non-Gaussian (a blob scores
KL_z = 2.74) yet the control recovers it to KL_z ~ 0.002 — a ~1000x margin —
with curvature 0.248/0.223/0.226 against a true 0.226, converging at n=70-90.

### Multimode

| sep | control result |
|---|---|
| 6.0 | 3/4 modes, w_relerr 0.96 |
| 5.0 | 4/4, 3/4, 4/4 across seeds; w_relerr 0.76 / 0.95 / 0.29 |
| **4.0** | **4/4 modes on every seed** |

At sep=4 under *natural* convergence the criterion still fired at wildly
different points (n = 95 / 115 / 280, w_relerr 0.79 / 0.10 / 0.056), so
quality-at-convergence measured *when the criterion tripped*, not the fit under
test. Running to a matched fixed budget (`DontConverge`, n=300) collapses that:

| | natural | fixed budget |
|---|---|---|
| w_relerr | 0.79 / 0.10 / 0.056 | **0.062 / 0.108 / 0.060** |

The banana does **not** get a fixed budget: with convergence disabled it runs
past GP saturation into `GPAcquisitionError` ("Acquisition returning no
values"), which is just a different arbitrary stopping rule.

Settings live in `RST_B` / `RST_NTWIST` / `RST_SEP` / `RST_FIXED`; the value
used is recorded in each result JSON.

## Open problem: d=30 rails against the length-scale ceiling

The d=30 arm fails — runs die at n=180-240 with `GPAcquisitionError` and
KL ~ 280-310 (garbage). The fitted kernel shows why:

```
d=30 (fails):  length_scale=[1.05, 100, 100, 100, 100, 1.04, 1, 100, ..., 100]
d=16 (works):  length_scale=[29.3, 30.2, 31.2, 27.9, 32.4, 35.7, ...]
```

Ten of the thirty length scales are pinned **exactly at 100**, the upper bound
of the `[1e-3, 1e2]` default merged in PR #4. A Gaussian log-posterior is
quadratic, so an RBF legitimately wants a very long correlation length; d=16
sits at ~30, comfortably inside, but d=30 wants more than the ceiling allows,
the fit rails, and acquisition then finds no candidate and dies.

Note this arm is *uninformative rather than wrong*: several arms return
bit-identical results (S0/S1/S3 seed 2 all give LML=-13.7367, KL=276.0591)
because they all die at the same point, before the restart budget matters.

## Layout

- `experiments/cluster/run_restart_study.py` — one run
- `experiments/cluster/gen_manifest_restart.py` → `manifest_restart.txt` (125 tasks)
- `experiments/cluster/restart.sbatch` — array job, partition **cpu36**
- `experiments/cluster/aggregate_restart.py` — cost + quality-vs-control table

Manifest order is d=30 first (longest, ~3.7 h at S0) to fill the wall-clock tail.
