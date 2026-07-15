"""
EXPENSIVE step: run GPry with fixed-budget NUTS acquisition on a d-dim Gaussian,
checkpointing (pickling) the GP at target point-counts so evaluation can be done
cheaply and repeatedly afterwards (run_eval.py).

Usage:
    python run_acquisition.py <d> <max_total> <ckpt1,ckpt2,...> <run_seed> [outroot]

e.g. python run_acquisition.py 30 2500 1500,2500 1
Writes to  <outroot>/d<d>_seed<seed>/  : truth.npz, gp_n<ckpt>.pkl, conv.npz, meta.json
"""
import os
import sys
import time
import json
from copy import deepcopy

import numpy as np

import common as C

from gpry.run import Runner


def main():
    d = int(sys.argv[1])
    max_total = int(sys.argv[2])
    ckpts = sorted(int(x) for x in sys.argv[3].split(","))
    seed = int(sys.argv[4])
    outroot = sys.argv[5] if len(sys.argv) > 5 else "runs"
    outdir = os.path.join(outroot, f"d{d}_seed{seed}")
    os.makedirs(outdir, exist_ok=True)

    tgt = C.make_gaussian(d)
    np.savez(os.path.join(outdir, "truth.npz"), mean=tgt["mean"], cov=tgt["cov"],
             marg=tgt["marg"], bounds=tgt["bounds"])

    done = set()

    def cb(runner):
        n = runner.gpr.n_total
        for ck in ckpts:
            if ck not in done and n >= ck:
                done.add(ck)
                C.save_gp(deepcopy(runner.gpr), os.path.join(outdir, f"gp_n{ck}.pkl"))
                print(f"[ckpt] saved GP at n_total={n} (target {ck})", flush=True)

    n_initial = max(3 * d, 60)
    r = Runner(tgt["logLkl"], tgt["bounds"].tolist(), ref_bounds=tgt["ref_bounds"].tolist(),
               gpr="RBF", gp_acquisition={"NORA": {"sampler": "nuts", "mc_every": 1}},
               mc={"nested": {}}, callback=cb,
               options={"n_initial": n_initial, "max_initial": 3 * n_initial,
                        "max_total": max_total},
               seed=seed, verbose=1, checkpoint=None)
    r.generate_mc_sample = lambda *a, **k: None       # skip GPry's own (unguarded) final MC
    r.diagnose_last_mc_sample = lambda *a, **k: None
    r.acquisition._nuts_kwargs = {"pad_multiple": 512, "max_num_doublings": 7}

    t0 = time.time()
    try:
        r.run()
    except Exception as e:
        print(f"run() ended: {type(e).__name__}: {str(e)[:80]}", flush=True)
    wall = time.time() - t0
    # ensure the final point-count is also checkpointed
    cb(r)

    conv = {"values": [], "n_evals": []}
    for cc in r.convergence:
        if type(cc).__name__ == "GaussianKL":
            v, npost, _ = cc.get_history()
            conv = {"values": list(map(float, v)), "n_evals": list(map(float, npost))}
            break
    np.savez(os.path.join(outdir, "conv.npz"),
             v=np.array(conv["values"]), n=np.array(conv["n_evals"]))
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump({"d": d, "seed": seed, "max_total": max_total, "ckpts": ckpts,
                   "n_total_final": int(r.gpr.n_total), "wall_s": round(wall, 1),
                   "checkpointed": sorted(done)}, f, indent=2)
    print(f"DONE d={d} seed={seed}: n_total={r.gpr.n_total} in {wall:.0f}s -> {outdir}",
          flush=True)


if __name__ == "__main__":
    main()
