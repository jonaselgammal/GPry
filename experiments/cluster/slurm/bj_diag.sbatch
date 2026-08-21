#!/usr/bin/env bash
#SBATCH --job-name=bjdiag
#SBATCH --output=logs/bjdiag_%j.out
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=cpu36
set +u
echo "node=$(hostname)"
echo "ulimit -v=$(ulimit -v) -l=$(ulimit -l) -s=$(ulimit -s) -n=$(ulimit -n)"
echo "vm.max_map_count=$(cat /proc/sys/vm/max_map_count 2>/dev/null)"
grep -E "MemTotal|MemAvailable" /proc/meminfo
echo "cgroup mem limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)"
source "$HOME/bhome/anaconda3/etc/profile.d/conda.sh"
conda activate /bhome/$USER/envs/gpry_v2
export TMPDIR=/bhome/$USER/tmp
export MPLCONFIGDIR=$TMPDIR/mpl_$SLURM_JOB_ID; mkdir -p "$MPLCONFIGDIR"
cd /bhome/$USER/gpry_nuts_tests/GPry_v2/experiments/cluster
# `common` now lives in run/; the inline python -c below imports it.
export PYTHONPATH="$PWD/run:${PYTHONPATH:-}"
for CFG in "baseline" "omp1" "nothunk" "omp1_nothunk"; do
  echo "=========== CONFIG: $CFG ==========="
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  unset XLA_FLAGS
  case $CFG in
    omp1) export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ;;
    nothunk) export XLA_FLAGS="--xla_cpu_use_thunk_runtime=false" ;;
    omp1_nothunk) export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
                  export XLA_FLAGS="--xla_cpu_use_thunk_runtime=false" ;;
  esac
  timeout 300 python -c "
import warnings; warnings.filterwarnings('ignore')
import numpy as np, common as C, gpry.ns_interfaces as nsint
from gpry.mc_interfaces import build_jax_gp_loglike
tgt=C.make_gaussian(8); b=np.asarray(tgt['bounds'],float)
import glob
p=sorted(glob.glob('/bhome/$USER/gpry_nuts_tests/cmp_out/d8_nuts_seed1_surrogate.pkl'))
sur=C.load_gp(p[0])
si=nsint._ns_interfaces['blackjax'](b,1); si.set_prior(b)
si.set_precision(nlive=200,num_repeats=40,precision_criterion=0.01,nprior=2000,max_ncalls=50000)
si.jax_loglike=build_jax_gp_loglike(sur)
X,y,w,lZ,lZs=si.run(lambda Z: sur.predict(np.atleast_2d(Z),return_std=False,validate=False),out_dir=None,seed=1)
print('BJ_OK n=',len(X))
" 2>&1 | grep -E "BJ_OK|Cannot allocate|LLVM ERROR|Error" | head -3
done
