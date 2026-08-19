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
- **Final head-to-head, 50 runs on EXCLUSIVE nodes** (`results/final/{BASE,PROP}`):
  | case | fit speedup | loop speedup | quality BASE->PROP | fails |
  |---|---|---|---|---|
  | gauss d=30 | 5.07x | 1.48x | 0.0205 -> 0.0192 | 1 -> 0 |
  | gauss d=16 | 9.41x | 1.20x | 0.0058 -> 0.0054 | 0 -> 0 |
  | gauss d=8 | 10.91x | 1.11x | 0.0016 -> 0.0021 | 0 -> 0 |
  | curved d=5 | 8.38x | 1.15x | 0.0019 -> 0.0017 | 0 -> 0 |
  | multimode d=5 | 10.58x | 1.48x | 0.0623 -> 0.0515 | 0 -> 0 |
  BASE = current defaults; PROP = S2 (2 restarts) + ceiling 1e3 + `optimizer_ftol=1e-5`.
- **Fit is no longer the bottleneck**: falls from 5-57% of the loop to 0.7-16.7%. Further
  work on it is capped at 1.20x (d=30), ~1.02x elsewhere. Remaining d=30 cost is
  acquisition: 621 s of a 758 s loop.
- **`length_scale_prior` bug** found, fixed, PR'd, merged to main (PR #4).

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

- **The 836-nat attractor was measured on 12 restarts, not 68.** 68 is the default budget
  (10+2d-2) at d=30; only 12 were instrumented. Say "all 12 measured, of the 68 drawn".
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
| `GPry/experiments/cluster/` | harness: `run_restart_study.py`, `compare_samplers.py`, `common.py`, manifests, sbatch |
| `GPry/experiments/cluster/common.py` | targets + evaluators (`make_curved`, `make_multimode`, exact reference samples) |
| `results/final/{BASE,PROP}` | the 50-run head-to-head |
| `results/restart/` | the 125-run restart study |
| `notes/RESTART_STUDY.md` | full design, calibration, results, corrections |
| `paper/main.tex` | scaffolding + 5 figures |

**Plotting rule (non-negotiable): corner plots always show ALL dimensions, never a subset.**

Verification rule: never report that something ran/passed without pasting the real command and
its real output. Label unrun claims **NOT RUN / UNVERIFIED**.
