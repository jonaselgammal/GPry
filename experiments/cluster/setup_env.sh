#!/usr/bin/env bash
# Run on ssh1.ux.uis.no (internet + conda), AFTER cloning the repo, e.g.:
#   mkdir -p /bhome/$USER/gpry_nuts_tests && cd /bhome/$USER/gpry_nuts_tests
#   git clone --branch hamilton_mc https://github.com/jonaselgammal/GPry.git GPry
#   bash GPry/experiments/cluster/setup_env.sh
# Creates a conda env on beegfs, installs GPry + jax + blackjax, and copies the
# run scripts flat into /bhome/$USER/gpry_nuts_tests/ so you submit from there.
set -euo pipefail
export BH=/bhome/$USER
export TMPDIR=$BH/tmp PIP_CACHE_DIR=$BH/.cache/pip CONDA_PKGS_DIRS=$BH/conda_pkgs
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$BH/envs"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../GPry/experiments/cluster
REPO="$(cd "$HERE/../.." && pwd)"                       # .../GPry
PROJ="$BH/gpry_nuts_tests"

source "$HOME/bhome/anaconda3/etc/profile.d/conda.sh"
conda create -p "$BH/envs/gpry_nuts" python=3.11 -y || true
conda activate "$BH/envs/gpry_nuts"

# GPry pulls scikit-learn/dill/ultranest/getdist/pandas/scipy/matplotlib/h5py.
# Only extras the NUTS backend needs: jax + blackjax (CPU is fine here).
pip install -e "$REPO" "jax[cpu]" blackjax

mkdir -p "$PROJ"
cp "$HERE"/{common.py,run_acquisition.py,run_eval.py,manifest.txt,acq.sbatch,eval.sbatch,README.md} "$PROJ"/

python - <<'PY'
import gpry, jax, blackjax, getdist, dill, scipy, sklearn
from gpry.mc_interfaces import nuts_acquire   # NUTS backend present
print("ENV OK  jax", jax.__version__, "| blackjax", blackjax.__version__)
PY
echo "Setup complete. Env: $BH/envs/gpry_nuts   Submit jobs from: $PROJ"
