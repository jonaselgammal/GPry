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

---

# RESULTS (125 runs, all complete)

## 1. Quality is preserved by every arm, everywhere

Median headline metric per arm; all five arms are statistically
indistinguishable in every case:

| target | S0 (control) | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| curved d=5 (KL_z) | 0.0019 | 0.0021 | 0.0023 | 0.0017 | 0.0015 |
| multimode d=5 (w_relerr) | 0.062 | 0.062 | 0.064 | 0.072 | 0.057 |
| gauss d=8 (KL) | 0.0016 | 0.0021 | 0.0016 | 0.0021 | 0.0018 |
| gauss d=16 (KL) | 0.0058 | 0.0063 | 0.0057 | 0.0057 | 0.0061 |
| gauss d=30 (KL) | 0.0209 | 0.0196 | 0.0185 | 0.0199 | 0.0189 |

**All 25 multimode runs recovered 4/4 modes**, in every arm.

## 2. The fit gets much cheaper — but the loop barely does

| case | fit share of loop | Amdahl ceiling | S2 fit speedup | S2 **loop** speedup |
|---|---|---|---|---|
| curved d=5 | 5.3% | 1.06x | 5.02x | **1.04x** |
| gauss d=8 | 6.8% | 1.07x | 4.84x | **1.04x** |
| gauss d=16 | 13.3% | 1.15x | 3.85x | **1.40x** |
| multimode d=5 | 29.1% | 1.41x | 8.69x | **1.48x** |
| gauss d=30 | 53.9% | 2.17x | 1.64x | **1.26x** |

**The 47x microbenchmark did not translate.** That number came from timing a
single full fit on one saved d=16 surrogate. In a real run the full fit happens
only every `round(2*sqrt(d))` iterations — every other iteration uses the cheap
`fit_simple_every` path with `n_restarts=1` — and the fit is only 5-54% of the
loop. There was never 47x available end to end.

(d=16 and multimode slightly EXCEED their Amdahl ceiling because the cheaper
arm also converged in fewer points, cutting acquisition work too; the ceiling
assumes fixed non-fit work.)

## 3. Failures are a background rate, not an arm effect

6 of 125 runs died with `GPAcquisitionError`, spread across arms *including the
control*:

| arm | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| failures | 1 | 1 | 1 | 2 | 0 |

At 5 seeds per cell this is indistinguishable from a ~5% background failure
rate. It does **not** support "fewer restarts is less robust" — nor the
converse.

## 4. Side-finding: the length-scale ceiling is causal for those failures

Failed runs have 10-12 of 30 length scales pinned at exactly 100, the ceiling
of the merged `[1e-3, 1e2]` default; successful ones sit at a median ~43 with
**none** railed. Re-running the identical failing case (d=30, S0, seed 3) with
only the ceiling changed:

| ls ceiling | outcome | n | KL | fit time |
|---|---|---|---|---|
| 1e2 (merged default) | **FAILS** | 240 | 311.7 | 14 s |
| 1e3 | converges | 690 | 0.0191 | 508 s |
| 1e5 | converges | 690 | 0.0186 | 1842 s |

Widening to 1e3 fixes it at no extra cost relative to a normal successful fit
(~591 s); 1e5 costs 3.6x more. **Suggested: raise the default ceiling to 1e3.**

## Correction to an earlier read of this data

An interim report here claimed "the d=30 arm is failing" and suspected a
regression from PR #4. That was wrong, from a selection effect: the first three
d=30 results to appear were the three failures, because a failing run dies in
~3 minutes while a healthy one takes ~20. With all 25 in, **21/25 succeed**.
Nor is it a regression — pre-fix v3 managed 2/3 seeds at d=30 and needed n=990;
post-fix it is 21/25 at n=660-720.
