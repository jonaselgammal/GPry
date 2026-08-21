"""
Why is the GP hyperparameter refit more expensive on NUTS-acquired training sets?

Phase 1 saw ~8x more LML evaluations per FULL fit at d=16 (7066/6200 vs 756/828)
at matched n. The refit is sampler-INDEPENDENT code, so the cause must be the
training data itself -- or the observation is a fluke of one run.

Stage 1: run both acquisitions, save the surrogate + training set.
Stage 2 (separate script): controlled, identical refits on both datasets with
         many seeds, plus geometry/conditioning diagnostics.

Usage: python diag_refit_cost.py <d> <sampler> <seed> [max_total] [outdir]
"""
import os
import sys
import json
import time
import warnings
from copy import deepcopy

import numpy as np

# `common.py` lives in ../run (this script moved into analyse/ on 2026-08-20).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "run"))
import common as C

from gpry.run import Runner


def main():
    d = int(sys.argv[1])
    sampler = sys.argv[2].lower()
    seed = int(sys.argv[3])
    max_total = int(sys.argv[4]) if len(sys.argv) > 4 else 360
    outdir = sys.argv[5] if len(sys.argv) > 5 else "diag_refit"
    os.makedirs(outdir, exist_ok=True)

    tgt = C.make_gaussian(d)
    n_initial = 3 * d
    r = Runner(
        tgt["logLkl"], tgt["bounds"].tolist(), ref_bounds=tgt["ref_bounds"].tolist(),
        gpr="RBF", gp_acquisition={"NORA": {"sampler": sampler, "mc_every": 1}},
        mc={"nested": {}},
        options={"n_initial": n_initial,
                 "max_initial": min(3 * n_initial, max_total),
                 "max_total": max_total},
        seed=seed, verbose=1, checkpoint=None,
    )
    if sampler == "nuts":
        r.acquisition._nuts_kwargs = {"pad_multiple": 512, "max_num_doublings": 7}
    t0 = time.time()
    try:
        r.run()
    except Exception:
        import traceback
        print("run() ended:\n" + traceback.format_exc(), flush=True)
    wall = time.time() - t0

    tag = f"d{d}_{sampler}_seed{seed}"
    sur = r.surrogate
    # Save the surrogate AND the raw training set (transformed space, which is
    # what the regressor and the hyperparameter fit actually see).
    C.save_gp(deepcopy(sur), os.path.join(outdir, f"{tag}_surrogate.pkl"))
    np.savez(
        os.path.join(outdir, f"{tag}_train.npz"),
        X_train_=np.asarray(sur.gpr.X_train_, dtype=float),
        y_train_=np.asarray(sur.gpr.y_train_, dtype=float),
        X_orig=np.asarray(sur.X, dtype=float),
        y_orig=np.asarray(sur.y, dtype=float),
        theta=np.asarray(sur.gpr.kernel_.theta, dtype=float),
    )
    r.progress.data.to_csv(os.path.join(outdir, f"{tag}_iters.csv"), index=False)
    with open(os.path.join(outdir, f"{tag}_meta.json"), "w") as f:
        json.dump({"d": d, "sampler": sampler, "seed": seed, "wall_s": round(wall, 1),
                   "n_total": int(sur.n_total),
                   "kernel": str(sur.gpr.kernel_)}, f, indent=2)
    print(f"[DIAG] saved {tag}: n_total={sur.n_total} wall={wall:.0f}s "
          f"kernel={sur.gpr.kernel_}", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
