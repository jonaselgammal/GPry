#!/usr/bin/env bash
#SBATCH --job-name=gpfinal
#SBATCH --output=logs/final_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=cpu36
# --exclusive: the earlier study co-scheduled four jobs per node, which swung
# per-eval cost 45->107 ms at identical n and d and made wall-clock timings
# incomparable. Both arms here get a whole node so the timings mean something.
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
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" manifest_final.txt)
read -r KIND D ARM SEED FTOL LSMAX <<< "$LINE"
unset RST_FTOL RST_LSMAX
[ "$FTOL"  != "0" ] && export RST_FTOL=$FTOL
[ "$LSMAX" != "0" ] && export RST_LSMAX=$LSMAX
export RST_B=1.2 RST_NTWIST=1 RST_SEP=4.0 RST_FIXED=multimode
LABEL=$([ "$ARM" = "S0" ] && echo BASE || echo PROP)
echo "task ${SLURM_ARRAY_TASK_ID}: $LABEL target=$KIND d=$D arm=$ARM seed=$SEED ftol=$FTOL lsmax=$LSMAX node=$(hostname)"
srun python run_restart_study.py "$KIND" "$D" "$ARM" "$SEED" \
     /bhome/$USER/gpry_nuts_tests/final_${LABEL}
