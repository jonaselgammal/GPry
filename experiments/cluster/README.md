# Cluster harness

Reorganised 2026-08-20. Nothing was deleted: everything that used to sit flat in
this directory is still here, just filed by role. Superseded material is in
`legacy/`.

The split is **running vs. analysing**:

| dir | contains | touches the cluster? | reads `results/`? |
|---|---|---|---|
| `run/` | the scripts that execute science | yes (launched by `slurm/`) | no — output dir is an argv |
| `slurm/` | `*.sbatch`, `manifest_*.txt`, `gen_manifest_*.py` | yes | no |
| `analyse/` | aggregators, re-plotters, figure builders, fit diagnostics | no | yes |
| `legacy/` | the superseded 2026-08-10 Tier-B harness | historical | historical |

```
run/      common.py               targets + evaluators (make_curved, make_multimode,
                                  exact reference samples, GP save/load, NUTS read-out)
          run_restart_study.py    one run of the restart / final / h2h campaigns
          compare_samplers.py     acquisition-sampler head-to-head (v3 campaign)

slurm/    restart.sbatch          125-run restart-efficiency study  -> manifest_restart.txt
          final.sbatch            50-run BASE-vs-PROP head-to-head  -> manifest_final.txt
          h2h.sbatch              50-run post-PR#4 sampler h2h      -> manifest_h2h.txt
          cmp.sbatch cmp_bj.sbatch cmp_v3.sbatch     compare_samplers campaigns
          bj_diag.sbatch          one-off BlackJAX/LLVM memory diagnostic
          gen_manifest_{restart,final,h2h}.py        regenerate a manifest in place

analyse/  aggregate_restart.py    cost + quality-vs-control table for a flat results dir
          make_figures.py         all paper figures -> results/figures/ + paper/figures/
          replot_corner.py        all-dimension corner plot from a saved read-out/GP
          diag_refit_{cost,analyse}.py   why the hyperparameter refit costs what it does

legacy/   the 2026-08-10 Tier-B campaign: setup_env.sh, run_acquisition{,_2mode}.py,
          run_eval{,_2mode}.py, acq*.sbatch, eval*.sbatch, manifest{,_2mode}.txt,
          gen_manifest_2mode.py, aggregate_results.py, phase1_baseline.py, README.md
          (its own README documents that workflow). Pre-PR#4 and pre-matched-final-MC:
          kept for provenance, do not quote its timings.
```

## Submitting

Every sbatch `cd`s to the cluster checkout's `experiments/cluster/` itself and
then refers to `slurm/<manifest>` and `run/<script>`. Submit from
`experiments/cluster/` so that `#SBATCH --output=logs/...` lands there:

```bash
cd /bhome/$USER/gpry_nuts_tests/GPry_v2/experiments/cluster
mkdir -p logs
sbatch --array=1-50 slurm/h2h.sbatch
```

Every sbatch that uses `--exclusive` MUST also carry `--ntasks=1 --nodes=1` and
repeat them on `srun` — without them srun derives `ntasks = floor(36/8) = 4` and
runs four racing copies of the job. Audit with `grep -c "^\[ARM\]" <task log>`;
it must print 1.

## Imports

`run/` is the import root: `common.py` sits there, and a script launched as
`python run/foo.py` gets `run/` on `sys.path[0]` automatically. Scripts in
`analyse/` and `legacy/` prepend `../run` to `sys.path` themselves.

## Where the results are

`analyse/` scripts are the only ones with baked-in paths, and they anchor on the
project root (four levels up from this checkout's `experiments/cluster/<sub>/`):

- `results/paper_runs/` — `final/`, `restart/`, `h2h/`, `v3/` (the defensible campaigns)
- `results/other_runs/` — `raw/`, `legacy_csv/`, ... (superseded campaigns)
- `results/figures/` and `paper/figures/` — figure output

`aggregate_restart.py`, `replot_corner.py` and the `diag_*` scripts take their
directory as an argument and have no baked-in paths at all.
