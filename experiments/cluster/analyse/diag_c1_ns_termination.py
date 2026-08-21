"""
C1 diagnosis: replay ONE NORA/UltraNest acquisition step on a saved final
surrogate, and record how much nested-sampling work it actually does.

Motivation: in the h2h campaign UltraNest costs 585k-1371k surrogate
evaluations per acquisition iteration at d=16 (with ~3.4M spikes), but only
87k-195k at d=30 -- despite nlive growing 400 -> 750 and the slice length
5d growing 80 -> 150.  That inversion, not the d=16 number, is what makes the
d=30 speedup look small.  This replays the step outside the loop so UltraNest's
own termination bookkeeping (ncall, niter, logz, logzerr) is visible.

Usage:  python diag_c1_ns_termination.py <surrogate.pkl> [<surrogate.pkl> ...]
"""
import os
import sys
import time
import json
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run"))
import common as C  # noqa: E402

from gpry.ns_interfaces import InterfaceUltraNest  # noqa: E402


def replay(path, nlive_per_training=3, nlive_max_per_dim=25, num_repeats_per_dim=5,
           frac_remain=0.01, max_ncalls=None):
    sur = C.load_gp(path)
    d = sur.d
    bounds = np.asarray(sur.bounds)
    n_regress = sur.n_regress
    nlive = int(min(nlive_per_training * n_regress, nlive_max_per_dim * d))
    nsteps = int(num_repeats_per_dim * d)

    iface = InterfaceUltraNest(bounds, verbosity=1)
    iface.set_precision(nlive=nlive, precision_criterion=frac_remain,
                        num_repeats=nsteps, max_ncalls=max_ncalls)

    def logp(X):
        prev = sur.minus_inf_value
        sur.minus_inf_value = -1e-300
        out = sur.predict(np.atleast_2d(X), return_std=False, validate=False)
        sur.minus_inf_value = prev
        return out

    n0 = sur.n_eval
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y, w, logZ, logZstd = iface.run(logp, out_dir=None)
    dt = time.time() - t0
    devals = sur.n_eval - n0
    res = iface.last_ultranest_result
    try:
        iface.delete_output()
    except Exception:
        pass
    return {
        "file": os.path.basename(path),
        "d": d,
        "n_train": int(n_regress),
        "nlive": nlive,
        "nsteps_slice": nsteps,
        "wall_s": round(dt, 1),
        "surrogate_evals": int(devals),
        "ultranest_ncall": int(res.get("ncall", -1)),
        "ultranest_niter": int(res.get("niter", -1)),
        "logz": float(res.get("logz", np.nan)),
        "logzerr": float(res.get("logzerr", np.nan)),
        "ess": float(res.get("ess", np.nan)),
        "n_mc_samples": int(len(X)) if X is not None else 0,
        # how many nlive-cycles of compression: niter / nlive
        "compression_efolds": round(int(res.get("niter", 0)) / nlive, 2),
        # evals normalised by the configured per-replacement work
        "evals_per_nlive_nsteps": round(devals / (nlive * nsteps), 3),
    }


if __name__ == "__main__":
    out = []
    for p in sys.argv[1:]:
        r = replay(p)
        out.append(r)
        print(json.dumps(r), flush=True)
    print("\n=== summary ===")
    hdr = ("file", "d", "n_train", "nlive", "nsteps_slice", "wall_s",
           "surrogate_evals", "ultranest_ncall", "ultranest_niter",
           "compression_efolds", "evals_per_nlive_nsteps", "logz", "logzerr", "ess")
    print(",".join(hdr))
    for r in out:
        print(",".join(str(r[k]) for k in hdr))
