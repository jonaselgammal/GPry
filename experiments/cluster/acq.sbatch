#!/usr/bin/env bash
#SBATCH --job-name=gpnuts_acq
#SBATCH --output=logs/acq_%A_%a.out
#SBATCH --array=1-9
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
set -eo pipefail
set +u
source "$HOME/bhome/anaconda3/etc/profile.d/conda.sh"
conda activate /bhome/$USER/envs/gpry_nuts
export TMPDIR=/bhome/$USER/tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK \
       OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /bhome/$USER/gpry_nuts_tests
mkdir -p logs runs
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" manifest.txt)
read -r D MAXT CKPTS SEED <<< "$LINE"
echo "task ${SLURM_ARRAY_TASK_ID}: d=$D max_total=$MAXT ckpts=$CKPTS seed=$SEED"
srun python run_acquisition.py "$D" "$MAXT" "$CKPTS" "$SEED" runs
