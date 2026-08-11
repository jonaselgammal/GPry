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
cd /bhome/$USER/gpry_nuts_tests/GPry_v2/experiments/cluster
mkdir -p logs
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" manifest_cmp.txt)
read -r D SAMPLER SEED <<< "$LINE"
echo "task ${SLURM_ARRAY_TASK_ID}: d=$D sampler=$SAMPLER seed=$SEED"
srun python compare_samplers.py "$D" "$SAMPLER" "$SEED" 0 /bhome/$USER/gpry_nuts_tests/cmp_out
