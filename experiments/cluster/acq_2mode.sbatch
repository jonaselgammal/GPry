#!/usr/bin/env bash
#SBATCH --job-name=gpmm_acq
#SBATCH --output=logs/mm_acq_%A_%a.out
#SBATCH --array=1-30
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
set -eo pipefail
set +u
source "$HOME/bhome/anaconda3/etc/profile.d/conda.sh"
conda activate /bhome/$USER/envs/gpry_nuts
export TMPDIR=/bhome/$USER/tmp
# Per-task matplotlib cache: a shared tex.cache races between array
# tasks and crashed an eval job (FileNotFoundError on a .dvi).
export MPLCONFIGDIR=$TMPDIR/mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK \
       OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /bhome/$USER/gpry_nuts_tests
mkdir -p logs runs_mm
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" manifest_2mode.txt)
read -r D SEP WR MAXT CKPTS SAMPLER SEED <<< "$LINE"
echo "task ${SLURM_ARRAY_TASK_ID}: d=$D sep=$SEP wr=$WR max=$MAXT ckpts=$CKPTS sampler=$SAMPLER seed=$SEED"
srun python run_acquisition_2mode.py "$D" "$SEP" "$WR" "$MAXT" "$CKPTS" "$SAMPLER" "$SEED" runs_mm
