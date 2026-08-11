# Fixed-budget NUTS high-d recovery test (gorina cluster)

**Question:** does GPry with fixed-budget NUTS acquisition build a GP that
recovers the posterior mode/covariance in high-d? Evaluated with a **reliable**
method (well-tuned NUTS-many-chains + a sampler-free Laplace covariance
cross-check) — **not** UltraNest, which we established fails to sample high-d GPs.

**Design:** the expensive acquisition run checkpoints (pickles) the GP at target
point-counts; evaluation is a separate, cheap, re-runnable step on the saved GPs.
Runs 3 dims × 3 seeds = **9 parallel jobs** (the seeds address the single-seed
caveat from earlier).

Point budgets (`manifest.txt`): **d=30 → 2500** (checkpoints 1500, 2500 — kept as
you asked), **d=16 → 1200** (750, 1200), **d=12 → 800** (500, 800).

Everything lives on beegfs at `/bhome/$USER/gpry_nuts_tests/`
(= `~/Documents/cluster/gpry_nuts_tests/` on your Mac).

---

## 1. One-time setup — **on ssh1** (internet + conda)
```bash
ssh ssh1.ux.uis.no
cd /bhome/$USER/gpry_nuts_tests
bash setup_env.sh          # env on beegfs, clones GPry hamilton_mc, installs jax+blackjax
```
Should end with `ENV OK jax ... | blackjax ...`.

## 2. Run the acquisitions (9 parallel jobs) — **on gorina11**
```bash
ssh gorina11
cd /bhome/$USER/gpry_nuts_tests
sbatch --partition cpu36 acq.sbatch      # array 1-9; each checkpoints its GP(s)
squeue -u $USER                          # d=30 tasks ~2-3h, d=12/16 faster
```
Each task writes `runs/d<d>_seed<seed>/`: `gp_n<ckpt>.pkl`, `truth.npz`,
`conv.npz`, `meta.json`. Logs in `logs/acq_*.out`.

## 3. Evaluate recovery (cheap, re-runnable) — **on gorina11**
```bash
cd /bhome/$USER/gpry_nuts_tests
sbatch --partition cpu36 eval.sbatch     # array 1-9; one eval per run dir
```
Writes per checkpoint: `eval.json`, `corner_n<ckpt>.png`, `marginals_n<ckpt>.png`.
`logs/eval_*.out` print a one-line `RECOVERED=True/False` per checkpoint.
Re-run this step any time without redoing step 2. Single run interactively:
```bash
srun -p cpu36 --cpus-per-task 8 --pty bash -c \
  'conda activate /bhome/$USER/envs/gpry_nuts && cd /bhome/$USER/gpry_nuts_tests && python run_eval.py runs/d30_seed1'
```

## 4. Read results (from your Mac via the mount, or on the cluster)
`eval.json` per run dir — the numbers that matter:
- `kl_nuts` — KL(GP posterior ‖ truth); small = recovered.
- `max_mean_in_sigma` — posterior-mean distance from the true mode, in σ (mode found ≈ < 0.25).
- `std_relerr_nuts` — median relative error of recovered marginal σ's.
- `std_relerr_laplace` — sampler-free covariance cross-check; may be `NaN` on GPs where the SVM/clip flattens the surface (expected — trust `_nuts`).
- `n_divergent` — NUTS divergences (should be 0).

Plots: `marginals_n*.png` (all-d 1-D marginals, truth=black vs recovered=orange),
`corner_n*.png` (triangle, first ≤10 params). Roll-up:
```bash
cd /bhome/$USER/gpry_nuts_tests && for f in runs/*/eval.json; do echo "$f"; cat "$f"; echo; done
```

---

## Notes / knobs
- Fixed NUTS budget = 32 chains × (80 warmup + 60 draws), with the high-d
  tunings baked in (`pad_multiple=512`, `max_num_doublings=7` — the tree-depth
  cap that stops a sparse/stiff early 30d GP from blowing up to ~70 s/call). Set
  in `run_acquisition.py`.
- `cpu36` (8×36 cores, 4-day limit) fits the 9 jobs; `cpu64` also works. Work is
  CPU-bound (GP Cholesky), so no GPU needed.
- Tested with jax 0.6.x + blackjax (window_adaptation / nuts / num_integration_steps
  / max_num_doublings API). If the installed blackjax differs, those calls in
  `gpry/mc_interfaces.py` are the ones to check.
- Change budgets/seeds by editing `manifest.txt` and `--array` in the sbatch files.
