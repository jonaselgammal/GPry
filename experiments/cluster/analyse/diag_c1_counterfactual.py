"""
C1 counterfactual: what does the UltraNest acquisition step cost at d=30 once
the `-1e-300` sentinel is no longer the global maximum?

`GPAcquisition._do_mc_sample_ultranest` sets `surrogate.minus_inf_value` to
-1e-300 (a number equal to 0 to ~300 digits).  When a whole vectorized batch
falls inside the SVM-masked region, `SurrogateModel.predict` takes its
`np.all(~finite)` early-return, which SKIPS the upper clipper, so those points
come back as -1e-300 -- about 59 nats ABOVE every clipped real value.  At d=30
that masked region is ~26-33% of the prior box, so nested sampling climbs into
it and stops.

Here the same surrogate is sampled twice: once as the campaign did, and once
with the sentinel mapped below the training minimum (what -inf is supposed to
mean).  The difference is the size of the artefact.

Usage: python diag_c1_counterfactual.py <surrogate.pkl> [max_ncalls]
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


def run(sur, fixed, max_ncalls):
    floor = float(np.min(sur._y[sur._i_regress])) - 100.0

    def logp(X):
        X = np.atleast_2d(X)
        prev = sur.minus_inf_value
        sur.minus_inf_value = -1e-300
        y = sur.predict(X, return_std=False, validate=False)
        sur.minus_inf_value = prev
        if fixed:
            y = np.where(y > -1e-6, floor, y)   # sentinel -> a real -inf stand-in
        return y

    nlive = int(min(3 * sur.n_regress, 25 * sur.d))
    iface = InterfaceUltraNest(np.asarray(sur.bounds), verbosity=1)
    iface.set_precision(nlive=nlive, precision_criterion=0.01,
                        num_repeats=5 * sur.d, max_ncalls=max_ncalls)
    n0 = sur.n_eval
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iface.run(logp, out_dir=None)
    dt = time.time() - t0
    r = iface.last_ultranest_result
    try:
        iface.delete_output()
    except Exception:
        pass
    return {"sentinel": "fixed" if fixed else "as-run (-1e-300)",
            "nlive": nlive, "wall_s": round(dt, 1),
            "evals": int(sur.n_eval - n0), "niter": int(r["niter"]),
            "efolds": round(r["niter"] / nlive, 2),
            "max_logl": float(r["maximum_likelihood"]["logl"]),
            "logzerr": float(r["logzerr"]),
            "hit_max_ncalls": bool(max_ncalls and (sur.n_eval - n0) >= max_ncalls)}


if __name__ == "__main__":
    path = sys.argv[1]
    max_ncalls = int(float(sys.argv[2])) if len(sys.argv) > 2 else 3_000_000
    sur = C.load_gp(path)
    print(json.dumps({"file": os.path.basename(path), "d": int(sur.d),
                      "n_regress": int(sur.n_regress),
                      "max_ncalls_cap": max_ncalls}), flush=True)
    for fixed in (False, True):
        print(json.dumps(run(sur, fixed, max_ncalls)), flush=True)
