#!/usr/bin/env bash
#SBATCH --job-name=gpcmp
#SBATCH --output=logs/cmp_%A_%a.out
#SBATCH --array=1-18
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu36
set -eo pipefail
set +u
source "$HOME/bhome/anaconda3/etc/profile.d/conda.sh"
conda activate /bhome/$USER/envs/gpry_v2
export TMPDIR=/bhome/$USER/tmp
export MPLCONFIGDIR=$TMPDIR/mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK \
       OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
# JAX/XLA on a shared node: several array tasks landing together exhausted memory
# (MemoryError; "Failed to materialize symbols"). Keep XLA single-device, bound its
# compile threads, and give it a task-private compilation cache.
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JAX_COMPILATION_CACHE_DIR=$TMPDIR/jaxcache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
cd /bhome/$USER/gpry_nuts_tests/GPry_v2/experiments/cluster
mkdir -p logs
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/manifest_cmp.txt)
read -r D SAMPLER SEED <<< "$LINE"
echo "task ${SLURM_ARRAY_TASK_ID}: d=$D sampler=$SAMPLER seed=$SEED"
srun python run/compare_samplers.py "$D" "$SAMPLER" "$SEED" 0 /bhome/$USER/gpry_nuts_tests/cmp_out
