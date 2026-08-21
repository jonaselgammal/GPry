"""
C1 follow-up: is UltraNest's acquisition-time nested sampling terminating
EARLY at d=30?

RESULT (2026-08-20): frac_remain is NOT what stops it -- 1e-2 and 1e-4 give
2.14 and 2.10 e-folds.  The real cause is the -1e-300 sentinel; see
`diag_c1_sentinel.py`.  NOTE: the importance-sampling log Z below is NOT
trustworthy on these surrogates (IS ESS came out at ~5 of 4e5 draws); use it
only as a smoke test, never as a reference value.

Two independent handles on the same saved surrogate:
  (a) an importance-sampling estimate of log Z of the GP mean over the prior
      box, using a Gaussian proposal built from the training set;
  (b) UltraNest run at the production frac_remain (0.01) and at a much
      stricter one, so the drift in log Z with tolerance is visible.

Usage: python diag_c1_ns_logz.py <surrogate.pkl> [frac_remain ...]
"""
import os
import sys
import time
import json
import warnings

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run"))
import common as C  # noqa: E402
from gpry.ns_interfaces import InterfaceUltraNest  # noqa: E402
from gpry.tools import mean_covmat_from_evals  # noqa: E402


def surrogate_logp(sur):
    def logp(X):
        prev = sur.minus_inf_value
        sur.minus_inf_value = -1e-300
        out = sur.predict(np.atleast_2d(X), return_std=False, validate=False)
        sur.minus_inf_value = prev
        return out
    return logp


def is_logz(sur, n=400000, inflate=1.6, seed=0):
    """Importance-sampling log Z of the GP mean over the prior box."""
    b = np.asarray(sur.bounds)
    X_tr = np.asarray(sur.X_regress)
    y_tr = np.asarray(sur.y_regress)
    mu, cov = mean_covmat_from_evals(X_tr, y_tr)
    cov = cov * inflate ** 2
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((n, sur.d))
    X = mu + Z @ L.T
    inside = np.all((X >= b[:, 0]) & (X <= b[:, 1]), axis=1)
    X = X[inside]
    sign, logdet = np.linalg.slogdet(cov)
    dx = X - mu
    sol = np.linalg.solve(cov, dx.T).T
    logq = -0.5 * np.einsum("ij,ij->i", dx, sol) - 0.5 * logdet - 0.5 * sur.d * np.log(2 * np.pi)
    logp = surrogate_logp(sur)(X)
    lw = logp - logq
    lz = logsumexp(lw) - np.log(n)          # n, not len(X): outside-box mass -> 0
    # crude MC error
    w = np.exp(lw - lw.max())
    ess = w.sum() ** 2 / (w ** 2).sum()
    return float(lz), float(ess), int(len(X)), int(n)


def run_un(sur, frac_remain):
    d = sur.d
    nlive = int(min(3 * sur.n_regress, 25 * d))
    nsteps = int(5 * d)
    iface = InterfaceUltraNest(np.asarray(sur.bounds), verbosity=1)
    iface.set_precision(nlive=nlive, precision_criterion=frac_remain, num_repeats=nsteps)
    n0 = sur.n_eval
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y, w, logZ, logZstd = iface.run(surrogate_logp(sur), out_dir=None)
    dt = time.time() - t0
    res = iface.last_ultranest_result
    try:
        iface.delete_output()
    except Exception:
        pass
    return {"frac_remain": frac_remain, "nlive": nlive, "nsteps": nsteps,
            "wall_s": round(dt, 1), "evals": int(sur.n_eval - n0),
            "niter": int(res["niter"]), "efolds": round(res["niter"] / nlive, 2),
            "logz_boxcorrected": float(logZ), "logzerr": float(res["logzerr"]),
            "ess": float(res["ess"])}


if __name__ == "__main__":
    path = sys.argv[1]
    fracs = [float(x) for x in sys.argv[2:]] or [1e-2, 1e-4]
    sur = C.load_gp(path)
    lz, ess, nin, ntot = is_logz(sur)
    print(json.dumps({"file": os.path.basename(path), "d": sur.d,
                      "n_regress": int(sur.n_regress),
                      "IS_logz": round(lz, 3), "IS_ess": round(ess, 1),
                      "IS_inside_box": nin, "IS_n": ntot}), flush=True)
    for fr in fracs:
        print(json.dumps(run_un(sur, fr)), flush=True)
