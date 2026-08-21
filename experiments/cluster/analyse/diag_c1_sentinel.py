"""
C1 mechanism probe: how often does the UltraNest acquisition step land on the
`minus_inf_value` sentinel?

`GPAcquisition._do_mc_sample_ultranest` replaces the surrogate's -inf with
**-1e-300**, i.e. a number numerically indistinguishable from 0.  Every real
GP-mean value on these targets is in [-560, -20], so the sentinel is not a
floor but the GLOBAL MAXIMUM of the function UltraNest is asked to explore.
Any SVM-masked region inside the prior box is therefore an attracting plateau.

Reports, per saved surrogate: the uniform-prior volume of the masked region,
the fraction of the acquisition run's evaluations that land on it, and where
nested sampling stopped.
"""
import os
import sys
import warnings
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run"))
import common as C  # noqa: E402
from gpry.ns_interfaces import InterfaceUltraNest  # noqa: E402


def probe(path, n_uniform=2_000_000, run_ns=True):
    sur = C.load_gp(path)
    b = np.asarray(sur.bounds)
    d = sur.d
    rng = np.random.default_rng(0)
    # prior volume of the SVM-masked ("infinite") region
    n_masked = 0
    chunk = 200_000
    for _ in range(n_uniform // chunk):
        X = rng.uniform(b[:, 0], b[:, 1], size=(chunk, d))
        Xt = sur.preprocessing_X.transform(X)
        n_masked += int((~sur.infinities_classifier.is_finite_X(Xt, validate=False)).sum())
    out = {"file": os.path.basename(path), "d": d, "n_regress": int(sur.n_regress),
           "masked_prior_volume_frac": n_masked / n_uniform}
    if not run_ns:
        return out
    stat = {"n": 0, "tot": 0}

    def logp(X):
        X = np.atleast_2d(X)
        prev = sur.minus_inf_value
        sur.minus_inf_value = -1e-300
        y = sur.predict(X, return_std=False, validate=False)
        sur.minus_inf_value = prev
        stat["tot"] += len(X)
        stat["n"] += int((y > -1e-6).sum())
        return y

    nlive = int(min(3 * sur.n_regress, 25 * d))
    iface = InterfaceUltraNest(b, verbosity=1)
    iface.set_precision(nlive=nlive, precision_criterion=0.01, num_repeats=5 * d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iface.run(logp, out_dir=None)
    r = iface.last_ultranest_result
    try:
        iface.delete_output()
    except Exception:
        pass
    out.update({"nlive": nlive, "ns_evals": stat["tot"],
                "frac_evals_on_sentinel": round(stat["n"] / stat["tot"], 4),
                "niter": int(r["niter"]), "efolds": round(r["niter"] / nlive, 2),
                "max_logl_found": float(r["maximum_likelihood"]["logl"]),
                "logzerr": float(r["logzerr"])})
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(probe(p)), flush=True)
