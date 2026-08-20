#!/usr/bin/env bash
#SBATCH --job-name=gpcmpbj
#SBATCH --output=logs/cmpbj_%A_%a.out
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=cpu36
set -eo pipefail
set +u
# JAX's CPU backend JITs through LLVM, which mmaps executable memory and reserves
# a large virtual address space. Slurm often propagates a virtual-memory ulimit
# from --mem, and LLVM then fails allocations of even a few hundred bytes
# ("Unable to allocate section memory"). NUTS survives it; BlackJAX-NS compiles a
# much bigger graph (num_inner_steps slice steps) and does not.
ulimit -v unlimited 2>/dev/null || echo "note: could not raise ulimit -v (now: $(ulimit -v))"
ulimit -s unlimited 2>/dev/null || true
echo "ulimit -v = $(ulimit -v)  |  node = $(hostname)"
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
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" manifest_bj.txt)
read -r D SAMPLER SEED <<< "$LINE"
echo "task ${SLURM_ARRAY_TASK_ID}: d=$D sampler=$SAMPLER seed=$SEED"
srun python compare_samplers.py "$D" "$SAMPLER" "$SEED" 0 /bhome/$USER/gpry_nuts_tests/cmp_out
