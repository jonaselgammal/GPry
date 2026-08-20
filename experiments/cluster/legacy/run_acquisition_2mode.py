"""
EXPENSIVE step (Tier-B multimodal): run GPry on a two-mode Gaussian mixture
separated by `sep` sigma, with EITHER NUTS or nested-sampling (ultranest)
acquisition, checkpointing the GP so evaluation is cheap/repeatable afterwards.

Usage:
    python run_acquisition_2mode.py <d> <sep> <wr> <max_total> <ckpts> <sampler> <seed> [outroot]

  <wr>      = weight ratio w0/w1 (1.0 = equal modes)
  <ckpts>   = comma-separated point-counts to checkpoint, e.g. 200,300
  <sampler> = nuts | ultranest   (the acquisition proposal sampler -- the thing
              under test; evaluation is sampler-independent, done in run_eval_2mode.py)

Writes  <outroot>/mm_d<d>_sep<sep>_wr<wr>_<sampler>_seed<seed>/ :
        truth.npz, gp_n<ckpt>.pkl, conv.npz, meta.json
"""
import os
import sys
import time
import json
from copy import deepcopy

import numpy as np

import common as C

from gpry.run import Runner


def tag(d, sep, wr, sampler, seed):
    s = f"{sep:g}".replace(".", "p")
    w = f"{wr:g}".replace(".", "p")
    return f"mm_d{d}_sep{s}_wr{w}_{sampler}_seed{seed}"


def main():
    d = int(sys.argv[1])
    sep = float(sys.argv[2])
    wr = float(sys.argv[3])
    max_total = int(sys.argv[4])
    ckpts = sorted(int(x) for x in sys.argv[5].split(","))
    sampler = sys.argv[6].lower()
    seed = int(sys.argv[7])
    outroot = sys.argv[8] if len(sys.argv) > 8 else "runs_mm"
    outdir = os.path.join(outroot, tag(d, sep, wr, sampler, seed))
    os.makedirs(outdir, exist_ok=True)

    tgt = C.make_two_mode(d, sep, weight_ratio=wr)
    np.savez(os.path.join(outdir, "truth.npz"),
             means=tgt["means"], cov=tgt["cov"], marg=tgt["marg"],
             weights=tgt["weights"], sep=tgt["sep"], weight_ratio=tgt["weight_ratio"],
             bounds=tgt["bounds"])

    done = set()

    def cb(runner):
        n = runner.surrogate.n_total
        for ck in ckpts:
            if ck not in done and n >= ck:
                done.add(ck)
                C.save_gp(deepcopy(runner.surrogate), os.path.join(outdir, f"gp_n{ck}.pkl"))
                print(f"[ckpt] saved GP at n_total={n} (target {ck})", flush=True)

    # GPry's standard initial design is 3*d. An earlier version of this
    # harness used max(3*d, 60); that floor is ~10x oversized at d=2 and
    # systematically inflated the low-d end of every cost curve.
    n_initial = 3 * d
    r = Runner(tgt["logLkl"], tgt["bounds"].tolist(), ref_bounds=tgt["ref_bounds"].tolist(),
               gpr="RBF", gp_acquisition={"NORA": {"sampler": sampler, "mc_every": 1}},
               mc={"nested": {}}, callback=cb,
               options={"n_initial": n_initial, "max_initial": min(3 * n_initial, max_total),
                        "max_total": max_total},
               seed=seed, verbose=1, checkpoint=None)
    r.generate_mc_sample = lambda *a, **k: None       # skip GPry's own (unguarded) final MC
    r.diagnose_last_mc_sample = lambda *a, **k: None
    if sampler == "nuts":
        r.acquisition._nuts_kwargs = {"pad_multiple": 512, "max_num_doublings": 7}

    t0 = time.time()
    try:
        r.run()
    except Exception:
        import traceback
        print("run() ended with exception:\n" + traceback.format_exc(), flush=True)
    wall = time.time() - t0
    cb(r)
    # GPry's convergence criterion can stop a run BEFORE the first checkpoint
    # (e.g. an easy d=2 target converging at n~140 with ckpts at 200/300), which
    # would leave the run unscoreable. Always save the final surrogate too.
    C.save_gp(deepcopy(r.surrogate), os.path.join(outdir, "gp_final.pkl"))
    print(f"[ckpt] saved final surrogate at n_total={r.surrogate.n_total}", flush=True)

    conv = {"values": [], "n_evals": []}
    for cc in r.convergence:
        if type(cc).__name__ == "GaussianKL":
            try:
                v, npost, _ = cc.get_history()
                conv = {"values": list(map(float, v)), "n_evals": list(map(float, npost))}
            except Exception:
                pass
            break
    np.savez(os.path.join(outdir, "conv.npz"),
             v=np.array(conv["values"]), n=np.array(conv["n_evals"]))
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump({"d": d, "sep": sep, "weight_ratio": wr, "sampler": sampler,
                   "seed": seed, "max_total": max_total, "ckpts": ckpts,
                   "n_total_final": int(r.surrogate.n_total), "wall_s": round(wall, 1),
                   "checkpointed": sorted(done)}, f, indent=2)
    print(f"DONE {tag(d, sep, wr, sampler, seed)}: n_total={r.surrogate.n_total} "
          f"in {wall:.0f}s -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
