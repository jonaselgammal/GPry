#!/usr/bin/env bash
#SBATCH --job-name=gph2h
#SBATCH --output=logs/h2h_%A_%a.out
#SBATCH --time=08:00:00
#SBATCH --exclusive
# CRITICAL (found 2026-08-19): --exclusive allocates ALL cores on the node, and
# srun then derives ntasks = floor(cores / cpus-per-task) = floor(36/8) = 4. Without
# an explicit --ntasks=1 this launches FOUR identical ranks of the script, which race
# to write the same output files and contend for the node. The surviving JSON is
# whichever rank finished last. Job 476785 task 4 produced two different complete
# results this way (n=180/KL=143.5 and n=990/KL=0.0188).
# --exclusive was added to make timings trustworthy; unpinned it did the opposite.
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=cpu36
# POST-PR#4 acquisition-sampler head-to-head: {nuts, ultranest} x 5 cases x 5 seeds.
#
# --exclusive: co-scheduling swung per-eval cost 45->107 ms at identical n and d,
# which is enough to invent or hide a 2x effect. Both arms get a whole node.
#
# --time=08:00:00 with a 7.5 h in-process soft limit (RST_MAXWALL). Rationale: the
# slowest run this project has ever measured is 13497 s (3.75 h, v3 d=30 NUTS,
# pre-fix), and the slowest UltraNest run is 5745 s (1.6 h). 8 h is >2x the worst
# observation and keeps a pathological task off an exclusive node overnight. The
# soft limit fires FIRST so the task still writes a result JSON with
# error="TIMEOUT" -- a timeout is a result (the configuration did not finish in
# the budget), not a missing data point.
set -eo pipefail
set +u
source "$HOME/bhome/anaconda3/etc/profile.d/conda.sh"
conda activate /bhome/$USER/envs/gpry_v2
export TMPDIR=/bhome/$USER/tmp
export MPLCONFIGDIR=$TMPDIR/mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export JAX_COMPILATION_CACHE_DIR=$TMPDIR/jaxc_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK \
       OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
cd /bhome/$USER/gpry_nuts_tests/GPry_v2/experiments/cluster
mkdir -p logs
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/manifest_h2h.txt)
read -r KIND D ARM SEED SAMPLER <<< "$LINE"
# PROP defaults, identical in both arms; only RST_SAMPLER differs.
export RST_FTOL=1e-5 RST_LSMAX=1e3
export RST_B=1.2 RST_NTWIST=1 RST_SEP=4.0 RST_FIXED=multimode
export RST_SAMPLER=$SAMPLER
export RST_MAXWALL=27000
echo "task ${SLURM_ARRAY_TASK_ID}: target=$KIND d=$D arm=$ARM seed=$SEED sampler=$SAMPLER node=$(hostname)"
srun --ntasks=1 --nodes=1 python run/run_restart_study.py "$KIND" "$D" "$ARM" "$SEED" \
     /bhome/$USER/gpry_nuts_tests/h2h_out
