# HANDOFF — GPry gradient-based (NUTS) acquisition

Updated 2026-08-19. Supersedes `notes/TODO.md`, which is stale (dated 08-12; most of
its items are resolved or were refuted — see "Refuted" below).

## Goal & phase

Make GP-surrogate Bayesian inference (GPry) tractable at high dimension by replacing
nested-sampling acquisition with gradient-based (NUTS) acquisition. Archetype: blend of
Research + Scientific-code, now entering **write-up + code-cleanup** phase. The methodology
question is settled; remaining work is a paper, a defensible code structure, and final runs.

## Done (verified, with evidence)

- **NUTS acquisition wins at d=16** (only). Acquisition-loop time on the converged runs,
  `results/v3/v3_out/`: NUTS 444/437/450 s vs UltraNest 3265/3406 s (**7.4x**) and BlackJAX
  3566/3599/3275 s, at equal KL (~0.013-0.014). **CORRECTED 08-19** -- the previously
  circulated "~11x" was wrong, and see the two RED FLAGS below before using any of this.
- **Final MC matched to the acquisition sampler** (was hardcoded UltraNest in every arm,
  making the final MC 79% of wall time). `gpry/run.py` `mc={"nuts"/"hmc"}`.
- **Restart-efficiency study, 125 runs.** (Failure count is **5**/125, not the 6 quoted in
  some prose; the per-arm table S0:1 S1:1 S2:1 S3:2 S4:0 is correct.) All 5 restart strategies give indistinguishable
  quality at d in {5,8,16,30}; all 25 multimode runs found 4/4 modes.
  `results/restart/`, `notes/RESTART_STUDY.md`.
- **Final head-to-head, 50 runs, CLEAN single-rank** (`results/final/{BASE,PROP}`, job 476949;
  the contaminated 4-rank original is preserved at `results/final_4rank/`):

  | case | fit BASE->PROP | fit speedup | loop speedup | quality BASE->PROP |
  |---|---|---|---|---|
  | gauss d=30 | 547 -> 100 s | **5.46x** | 1.39x | 0.0205 -> 0.0192 |
  | gauss d=16 | 17.4 -> 1.8 s | **9.45x** | 1.08x | 0.0058 -> 0.0054 |
  | gauss d=8 | 3.4 -> 0.3 s | **10.53x** | 1.13x | 0.0016 -> 0.0021 |
  | curved d=5 | 4.6 -> 0.6 s | **8.09x** | 1.13x | 0.0019 -> 0.0017 |
  | multimode d=5 | 69.5 -> 6.7 s | **10.30x** | 1.39x | 0.0623 -> 0.0515 |

- **Contention inflation MEASURED** (clean vs 4-rank, same seeds/config) -- was `[assumed]`, now
  `[derived]`. Quality and `n_total` are bit-identical between the two; only timings moved. The
  FIT inflated consistently 1.06-1.26x in all ten cells (BLAS-heavy, memory-bandwidth bound);
  LOOP times scattered +-25% with no consistent sign, i.e. acquisition run-to-run variance
  rather than contention. The BASE/PROP ratios survived, since both arms were equally
  contended. Cross-validation: d=16 PROP clean loop 132.4 s vs the independent h2h campaign's
  132.1 s for the identical config -- 0.2% agreement between two separate clean campaigns.

- **R1 post-fix sampler head-to-head** (`results/h2h/`, job 476793). **THE TWO HIGH-d
  CELLS ARE QUARANTINED — see the sentinel bug below.** Clean cells only:

  | case | loop ratio | per-point | masked prior vol | status |
  |---|---|---|---|---|
  | multimode d=5 | **7.63x** | 7.88x | 0.000000 | **CLEAN — matched budget n=300, use as headline** |
  | gauss d=8 | 3.02x | 2.90x | ~0 | clean |
  | curved d=5 | 3.31x | n.q. | up to 0.19 | CONTAMINATED |
  | gauss d=16 | 29.6x | 14.60x | ~2e-4 | **QUARANTINED** (understated; ~111x counterfactual) |
  | gauss d=30 | 1.43x | 1.06x | 0.18-0.50 | **QUARANTINED** (UltraNest never sampled) |

## THE SENTINEL BUG — invalidates every UltraNest comparison at high d

`GPAcquisition._do_mc_sample_ultranest` set `surrogate.minus_inf_value = -1e-300` to give
UltraNest a finite stand-in for -inf. **-1e-300 is zero to ~300 digits: the LARGEST
log-posterior, not the smallest.** Nested sampling maximises into the classifier-masked
region and stops there (`Explored until L=-1e-300`) after ~2 e-folds instead of ~30.

Reproduced directly on `h2h/gauss_d30_S2_ultranest_seed1_surrogate.pkl`: an all-masked
batch returns `-1e-300`, the same points in a mixed batch return `-59.23`, and real values
there span `[-418, -234]`. (Both paths differ because `predict` early-returns on a fully
masked batch, skipping the clipper.)

Gated by the masked fraction of the prior: **~0 at d=8, ~2e-4 at d=16, 0.18-0.50 at d=30**
— invisible at low d, total at high d. Hence:
- **d=30**: UltraNest quits after ~2 e-folds. Its 931 ms/point is the CHEAPEST cell in the
  campaign, below d=5 — not a scaling regime, an early exit. A direct counterfactual with
  the sentinel repaired ran **>=223x more evaluations and did not terminate**.
- **d=16**: finding the trap is a coin flip — 0.4M evals if found, 3.3M if not. **That
  bimodality IS the 6x UltraNest d=16 spread.** With no iteration escaping, UltraNest's
  median loop would be ~14669 s against NUTS's 132 s, i.e. **~111x, not 29.6x** [counterfactual].

**FIXED** on this branch (`d612bff`) by anchoring the stand-in to the training-data range,
with a regression test. The bug is on `origin/main` and needs its own upstream PR.

**Consequences**: do NOT start Group B until the high-d cells are re-run — B1 and B2 would
both be measured against a broken NS arm. Point counts are contaminated too (roughly half of
UltraNest's d=30 candidate pool comes from the meaningless plateau), so "NUTS needs 690 vs
990 points" is not clean either.

- **NUTS itself is fine** (H1 refuted): replaying a production acquisition step on the saved
  surrogates gives mean tree size 8.8 -> 12.8 from d=8 to d=30, **zero divergences at every
  d**, acceptance flat at 0.87-0.91, and near-d-independent per-point cost (735/619/877 ms at
  d=8/16/30). The per-iteration growth is GP size and padding capacity, not sampler decay.

- **`evals_acquire` does not count the NUTS arm's GP evaluations** — `nuts_acquire` uses the
  JAX path, which bypasses `surrogate.predict`, so the counter reads ~0 for it and the saved
  numbers reflect only downstream candidate ranking. **Any eval-count comparison between arms
  in the saved data is invalid.** True NUTS cost is ~44-62k GP evals per acquisition step.

- **False convergence is real and UltraNest-only**: `h2h gauss_d30_ultranest_seed3` reports
  `converged=True, error=None` at **n=180 with KL=158.69** -- GPry's criterion declaring success
  on a posterior three orders of magnitude wrong. Zero such cases in 200 NUTS runs. Caught only
  by the aggregator's recovery gate; without it that run would have entered the cost median as
  a fast UltraNest win.

- **Per-point cost is `median(per-run t_acquire/n_total)`**, NOT median-over-median. The
  latter pairs one run's time with another's point count and yields a value no run has.
  (The handoff previously carried ratio-of-medians values here; corrected 08-20.)

- **Fit is no longer the bottleneck**: falls from 5-57% of the loop to 0.7-16.7%. Further
  work on it is capped at 1.20x (d=30), ~1.02x elsewhere. Remaining d=30 cost is
  acquisition: 621 s of a 758 s loop.
- **`length_scale_prior` bug** found, fixed, PR'd, merged to main (PR #4).

## INFRASTRUCTURE TRAP: `--exclusive` without `--ntasks=1` runs N copies of every task

`#SBATCH --exclusive` allocates all 36 cores. With `--cpus-per-task=8` and no explicit
`--ntasks`, `srun` derives `ntasks = floor(36/8) = 4` and launches **four identical ranks**,
racing to write the same output files; the surviving JSON is whichever rank finished last.
`--exclusive` was added to make timings trustworthy and did the exact opposite.

Every sbatch MUST carry, and `srun` must repeat:

    #SBATCH --ntasks=1
    #SBATCH --nodes=1
    srun --ntasks=1 --nodes=1 python ...

Audit by counting banner lines per task log (`grep -c "^\[ARM\]"`); it must be 1.

| campaign | ranks/task | verdict |
|---|---|---|
| `rst_*` (restart study, 125 runs) | 1 | clean |
| `ftol_*`, `v3_*` | 1 | clean |
| **`final_476048` (`results/final/`)** | **4** | **timings contaminated** |

What survives: all four ranks agreed BIT-IDENTICALLY on `n_total`, `converged`, `evals_fit`,
`LML` to full precision, and `KL` -- NUTS is deterministic. So every QUALITY conclusion from
`results/final/` stands. What does not: the wall-clock numbers describe 4 processes x 8 threads
on a 36-core node, not the intended lone process. The 1-5% spread BETWEEN ranks does not bound
the inflation, because all four were equally contended.

The BASE/PROP ratios (5.07x fit, 1.48x loop) are probably close, since both arms were equally
affected, but that is an inference, not a measurement. Re-run `final.sbatch` (now fixed) to
recover real absolute times. The clean NUTS arm of job 476793 uses the same configuration as
`results/final/PROP`, so comparing them will quantify the inflation directly.

## RED FLAGS on the headline comparison (raised 08-19, must be resolved before the paper)

**1. At d=30 UltraNest BEAT NUTS in v3.** The one converged UltraNest run: 1840 s loop,
KL 0.0447. NUTS: 11822 s and 13150 s, KL 0.0475/0.0477. That is UltraNest **6.4x faster at
equal-or-better accuracy**. Any claim of a NUTS advantage at d=30 is currently unsupported
and contradicted by our own data.

**2. "UltraNest fails at d>=16 in three distinct ways" is NOT substantiated -- do not repeat
it.** `results/v3` shows ONE failure mode: length scales railed at **1e5**, the PRE-PR#4
bound. It hit both arms, including `d30_nuts_seed2` (9 of 30 railed, KL 291) and
`d16_ultranest_seed1` (5 railed, KL 57.9). It is a shared symptom of the `length_scale_prior`
bug, not an UltraNest-specific weakness.

**3. Therefore: every NUTS-vs-nested comparison we own is PRE-PR#4.** The bug made every fit
optimise over 10 decades instead of 4, and it plausibly hurt NUTS most (its d=30 loop was
~85% fit; post-fix the same case runs in 758 s vs 11822 s pre-fix, ~15x faster). The post-fix
campaigns (`results/restart`, `results/final`) are **NUTS-only**. There is **no post-fix
head-to-head at any dimension**. This is the blocking run for the paper.

## Cleanup done so far (branch `nuts-acquisition-v2`, all behaviour-verified)

Test suite went 0 -> 26 targeted tests; the 6 pre-existing failures are unchanged
throughout (identical on pristine `origin/main`, so they are stale, not regressions).

| commit | what | verification |
|---|---|---|
| `2fbe91f` | contract tests pinning the JAX backends vs the numpy surrogate | JAX is exact to machine precision when K is well conditioned |
| `b106b5f` | `_extract_stationary_kernel` dispatched on `k1`/`k2`, so a **Sum was evaluated as a Product** -- wrong number, no error. Now rejects what it cannot represent | end-to-end A/B **bit-identical**: LML=170.69418794091632 (17 digits), same n, evals, KL |
| `81544a8` | y-preprocessor affinity **verified**, not assumed (a nonlinear one would silently mis-scale the target); 3 hand-rolled probes -> 1 helper | accepts NormalizeY, rejects a SoftClipY stand-in |
| `6e55908` | `jax_enable_x64` centralised **and verified it took effect** (silently ignored if JAX is already locked to 32-bit; float32 error here is 6100%) | idempotent, suite unchanged |
| `6bba3af` | three byte-identical copies of the JAX RBF/Matern math -> one `_jax_k_vec` | end-to-end A/B **bit-identical** again |
| `2409bf3` | kernel-level tests vs `gpry.kernels` | agrees to **1e-12** for RBF and Matern nu in {0.5,1.5,2.5}, d in {1,3,7} |

Deliberately NOT done yet, and why:
- **attribute injection** (`sampler.jax_loglike = ...` -> a declared interface): cosmetic, and
  it touches the exact BlackJAX path the running campaign uses. Defer until R1 lands.
- **array-namespace kernels** (the real fix, which makes composition fall out of the kernel
  algebra instead of being rejected): the invariant it must preserve is now pinned by
  `2409bf3`. Do it after R1.

Corrected earlier claim: the module-level singletons (`_JAX_LOGLIKE_RUNNER`, `_NUTS_RUNNER`)
were described as "not re-entrant for two surrogates". That is WRONG -- the data is passed as
arguments, so one runner serves any number of surrogates, which was the whole fix for the
recompilation/OOM bug. The design is correct; do not "fix" it.

## In flight

Three parallel tracks:
1. **Code cleanup (main session).** Restructure the JAX GP log-likelihood; see Decisions.
   No behaviour change intended; must not regress.
2. **Paper agent.** `paper/main.tex` exists as scaffolding with 5 figures. Needs rewriting
   around the settled results.
3. **Runs/plotting agent.** Final paper runs + figures.

## Decisions

- **One paper, not several.** [Informed] Headline = tractability/robustness at high d (d=30
  works; UltraNest fails at d>=16). Fit-cost work is a *section*: alone it is 1.1-1.5x
  end-to-end and the mechanism is optimisation folklore. The interesting fit result is that
  all 68 random restarts collapse to one attractor 836 nats from the optimum.
- **Frame claims as tractability, not raw speed.** [Informed] Speed claims burned us twice
  ("67x" JAX, "47x" restart bench), and post-change the loop is acquisition-dominated.
- **Hybrid JAX, not 100% JAX.** [Informed] Keep numpy as source of truth; write stationary
  kernels once against an array namespace (`xp = array_namespace(x)`) so the same source runs
  under numpy (fit, CPU) and JAX (NUTS, optionally GPU). Rejected full JAX rewrite: end-to-end
  JAX measured wall-SLOWER, the hot path is already JAX, and it would force the hyperparameter
  fit off scipy L-BFGS-B (where jaxopt already burned us silently).
- **GPU: hybrid boundary is fine.** [Derived] CPU->GPU payload is ~170 KB, ~20x per run; the
  JAX NUTS path is callback-free. Real risk is fp64: consumer GPUs run fp64 at 1/32-1/64 of
  fp32, and x64 is mandatory (float32 destroys the GP mean at amp~1e6). Mitigation: the
  mixed-precision f32-solve/f64-reduce kernel already PoC'd.
- **d=5 targets built from scratch.** [Ad-hoc, validated] GPry's `Curved_degeneracy`/`Ring`
  are 2d-only and its n-dim `Himmelblau` silently leaves dims unconstrained. Calibrated
  against the control: banana `n_twist=1, b=1.2`; multimode `sep=4.0` with a matched fixed
  budget (`DontConverge`, n=300).

## Open questions / risks

- **The attractor claim is now measured at the FULL budget, and it is not "all"**
  (`results/paper_runs/attractor/attractor_d30_full68.json`, regenerated 08-20). Of the 68
  random restarts on that surrogate: **64 land on the degenerate attractor** at LML=-797.44,
  **2 reach the optimum** at +38.47, and 2 land elsewhere (-93.26, -129.77). Both informed
  starts reach +38.465. Gap = 835.91 nats.
  The earlier "all 68 collapse" was an extrapolation from 12 instrumented restarts and is
  FALSE — random restarts do occasionally work. The argument for dropping them is that they
  are **redundant given the informed starts** (which find the optimum every time), not that
  they never work; the direct evidence remains that S2 matched S0's quality across 125 runs.
  Caveat: 64/68 is one draw sequence on one surrogate, so phrase it as "64 of the 68 restarts
  of this fit", never as a rate.
- **~5-16% `GPAcquisitionError` failure rate at d=30**, present in the BASELINE too. We can
  say it is a background rate unrelated to our changes; we cannot say why. A referee will ask.
- **Surface mismatch**: NUTS acquisition applies the SVM mask but NOT `clip_factor`, so it
  does not target a bit-identical function to the NS arm. Defensible; must be stated.
- **`optimizer_ftol=1e-5` caveat**: on 2 of 8 synthetic starts a restart stopped 3.4 / 12.0
  nats early. Never observed end-to-end (31 runs). `1e-6` gives 1.9x instead of 3x if you
  want margin.
- **Silent-wrongness in `_extract_stationary_kernel`**: it cannot distinguish `Sum` from
  `Product` (both expose `k1`/`k2`), so a genuine sum of two stationary kernels computes the
  wrong thing with NO error. Only safe today because the default is `C*RBF + White` and White
  contributes nothing. This is the top cleanup target.
- **Three proposed defaults NOT yet applied to main**: ceiling 1e2->1e3;
  `n_restarts_optimizer` 10+2d -> 2; `optimizer_ftol` None -> 1e-5.

## Refuted (do not re-litigate)

- Near-duplicate training points / cond(K) explaining the fit cost — refuted by measurement;
  healthy UltraNest GPs have the same cond ~1e14.
- "NS is 9x more expensive on NUTS-built GPs" — retracted, was one favourable seed.
- "d=30 is broken / PR #4 regressed it" — wrong, a selection effect (failing runs finish in
  ~3 min, healthy in ~20, so failures appeared first). 21/25 succeed.
- The analytic LML gradient being wrong — it is correct; the check used `eps=1e-5`, inside
  the FD noise floor. FD matches to 5 s.f. at `eps=1e-2/1e-3`.
- `fit_simple_every` as a speed lever — erratic pass/fail across seeds; the per-iteration fit
  is load-bearing.

## Dev commands & file map

Env: **`gpry-nuts`** conda env (base silently imports a different checkout).
`/opt/homebrew/Caskroom/miniconda/base/envs/gpry-nuts/bin/python`, with
`PYTHONPATH=/Users/jeg/Documents/GPry-NUTS/GPry`.

Cluster: `ssh jeg@ssh1.ux.uis.no` -> `ssh gorina11` for `sbatch` (login node has too-old
GLIBC; run anything Python via `srun -p cpu36`). Partition **cpu36 must be pinned**. Env
`/bhome/jeg/envs/gpry_v2`. Checkout `/bhome/jeg/gpry_nuts_tests/GPry_v2` (sync via
`git fetch && git reset --hard origin/nuts-acquisition-v2`). Use `--exclusive` for any timing
run: co-scheduling swung per-eval cost 45->107 ms at identical n and d.

| path | what |
|---|---|
| `GPry/` | the package, branch `nuts-acquisition-v2` (rebased on `origin/main`) |
| `GPry/gpry/mc_interfaces.py` | NEW, 776 lines — NUTS/HMC backend + JAX GP loglike |
| `GPry/experiments/cluster/run/` | scripts that EXECUTE science: `run_restart_study.py`, `compare_samplers.py`, `common.py` |
| `GPry/experiments/cluster/slurm/` | sbatch + manifests + manifest generators |
| `GPry/experiments/cluster/analyse/` | read finished products, produce numbers (`aggregate_restart.py`, `replot_corner.py`) |
| `GPry/experiments/cluster/legacy/` | the superseded 2026-08-10 Tier-B harness, kept whole (it deploys flat via its own `setup_env.sh`) |
| `GPry/experiments/cluster/run/common.py` | targets + evaluators (`make_curved`, `make_multimode`, exact reference samples) |
| `results/paper_runs/` | everything the paper cites: `h2h/ final/ final_4rank/ restart/ v3/` + summary CSVs |
| `results/paper_runs/plot_utils.py` | **single source of truth for every plotting setting**; sizes derive from the MEASURED `\textwidth` = 469.755 pt; `LAYOUT` switches 1- vs 2-column in one line |
| `results/other_runs/` | superseded campaigns and legacy readers; never cite as current |
| `notes/RESTART_STUDY.md` | full design, calibration, results, corrections |
| `paper/main.tex` | scaffolding + 5 figures |

**Plotting rule (non-negotiable): corner plots always show ALL dimensions, never a subset.**

Verification rule: never report that something ran/passed without pasting the real command and
its real output. Label unrun claims **NOT RUN / UNVERIFIED**.
