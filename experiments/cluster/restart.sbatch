#!/usr/bin/env bash
#SBATCH --job-name=gprst
#SBATCH --output=logs/rst_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=cpu36
set -eo pipefail
set +u
ulimit -v unlimited 2>/dev/null || true
source "$HOME/bhome/anaconda3/etc/profile.d/conda.sh"
conda activate /bhome/$USER/envs/gpry_v2
export TMPDIR=/bhome/$USER/tmp
export MPLCONFIGDIR=$TMPDIR/mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export JAX_COMPILATION_CACHE_DIR=$TMPDIR/jaxcache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK \
       OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
cd /bhome/$USER/gpry_nuts_tests/GPry_v2/experiments/cluster
mkdir -p logs
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" manifest_restart.txt)
read -r KIND D ARM SEED <<< "$LINE"
echo "task ${SLURM_ARRAY_TASK_ID}: target=$KIND d=$D arm=$ARM seed=$SEED"
srun python run_restart_study.py "$KIND" "$D" "$ARM" "$SEED" \
     /bhome/$USER/gpry_nuts_tests/restart_out
