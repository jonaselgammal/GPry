#!/usr/bin/env bash
#SBATCH --job-name=gpnuts_eval
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --array=1-9
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -eo pipefail
set +u
source "$HOME/bhome/anaconda3/etc/profile.d/conda.sh"
conda activate /bhome/$USER/envs/gpry_nuts
export TMPDIR=/bhome/$USER/tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK \
       OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /bhome/$USER/gpry_nuts_tests
mkdir -p logs
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" manifest.txt)
read -r D MAXT CKPTS SEED <<< "$LINE"
RUNDIR="runs/d${D}_seed${SEED}"
echo "eval ${SLURM_ARRAY_TASK_ID}: ${RUNDIR}"
srun python run_eval.py "${RUNDIR}"
