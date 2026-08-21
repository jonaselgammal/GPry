"""
C1 / hypothesis 1: does the NUTS acquisition step degrade from d=16 to d=30?

Replays ONE production NUTS acquisition step (the settings the h2h campaign
used: n_warmup=80, n_samples=60, max_chains=32, pad_multiple=512,
max_num_doublings=7) on each saved final surrogate, and reports the sampler
diagnostics the campaign did NOT persist: mean tree size, divergences,
acceptance, and the true number of GP-mean evaluations.

Note: `evals_acquire` in the campaign CSVs does NOT include these evaluations
for the NUTS arm -- the JAX GP log-likelihood bypasses `surrogate.predict`, so
the counter only sees the downstream candidate ranking (~1800/iteration
regardless of d).  Eval counts are therefore NOT comparable between arms in the
saved data; this script recovers the NUTS side.
"""
import os
import sys
import time
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run"))
import common as C  # noqa: E402
from gpry.mc_interfaces import nuts_acquire  # noqa: E402

KW = dict(n_warmup=80, n_samples=60, thin=1, max_chains=32,
          pad_multiple=512, max_num_doublings=7)


def probe(path, repeat=2):
    sur = C.load_gp(path)
    b = np.asarray(sur.bounds)
    rng = np.random.default_rng(0)
    out = None
    for k in range(repeat):          # first call pays JIT compilation
        n0 = sur.n_eval
        t0 = time.time()
        X, _, _, w, info = nuts_acquire(sur, b, rng=rng, return_info=True, **KW)
        dt = time.time() - t0
        out = {"file": os.path.basename(path), "d": int(sur.d),
               "n_regress": int(sur.n_regress), "call": k,
               "wall_s": round(dt, 2),
               "gp_evals_nuts": int(info["n_eval"]),
               "mean_tree_size": round(float(info["mean_tree_size"]), 2),
               "divergences": int(info["divergences"]),
               "accept_rate": round(float(info["accept_rate"]), 4),
               "n_chains": int(info["n_chains"]),
               "capacity": int(info["capacity"]),
               "n_candidates": int(len(X)),
               "predict_evals_counted": int(sur.n_eval - n0)}
        print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        probe(p)
