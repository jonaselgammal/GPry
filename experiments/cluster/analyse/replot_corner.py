"""Regenerate an all-dimension corner plot WITHOUT re-running the acquisition loop.

Two sources, in order of preference:

  <tag>_readout.npz   the saved NUTS read-out chain (free, instant)
  <tag>_surrogate.pkl the saved GP; the read-out is re-sampled from it (~seconds
                      to a minute, but still no acquisition loop)

Runs before 2026-08-19 have only the .pkl, because the read-out chain was not
being persisted; new runs get both.

Usage:
    python replot_corner.py <run.json> [out.png]
    python replot_corner.py <results_dir>            # every run in the dir
    python replot_corner.py <results_dir> --cache-only   # only write the .npz,
                                                         # leave the archived PNGs alone

The plot always shows ALL dimensions -- never a subset.
"""
import os
import sys
import glob
import json

import numpy as np

# `common.py` and `run_restart_study.py` live in ../run (this script moved
# into analyse/ on 2026-08-20).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "run"))
import common as C                                            # noqa: E402
import run_restart_study as RS                                 # noqa: E402
from run_restart_study import corner_all_dims                  # noqa: E402


def build_target(r):
    """Rebuild the exact target a run saw, from the run's own JSON.

    `run_restart_study.build_target` reads RST_B / RST_SEP into module-level
    constants at IMPORT time, so setting the environment here would be silently
    ignored. The parameter actually used is recorded per run as `target_param`,
    so it is passed explicitly instead.
    """
    kind, d, p = r["target"], r["d"], r.get("target_param")
    if kind == "gauss":
        return C.make_gaussian(d)
    if kind == "curved":
        return C.make_curved(d, b=float(p), n_twist=RS.CURVED_NTWIST)
    if kind == "multimode":
        return C.make_multimode(d, n_modes=RS.MULTIMODE_K, sep=float(p))
    raise ValueError(f"unknown target {kind!r}")


def readout_for(base, kind, tgt):
    """Return (X, w) for `base` (the path prefix without extension)."""
    npz = f"{base}_readout.npz"
    if os.path.exists(npz):
        z = np.load(npz)
        return z["X"], z["w"], "readout.npz"
    pkl = f"{base}_surrogate.pkl"
    if os.path.exists(pkl):
        sur = C.load_gp(pkl)
        X, w, _ = C.nuts_corner_samples(sur, tgt["bounds"], n_chains=64,
                                        n_warmup=300, n_samples=300)
        # Cache it so the next re-plot is free.
        np.savez_compressed(npz, X=np.asarray(X), w=np.asarray(w, float))
        return X, w, "surrogate.pkl (cached to readout.npz)"
    raise FileNotFoundError(f"neither {npz} nor {pkl} exists")


def one(js_path, out=None, cache_only=False):
    base = js_path[:-len(".json")]
    r = json.load(open(js_path))
    kind, d, arm = r["target"], r["d"], r["arm"]
    tgt = build_target(r)
    X, w, src = readout_for(base, kind, tgt)
    if cache_only:
        print(f"  {os.path.basename(base)}: read-out cached ({len(X)} pts, d={d}) "
              f"from {src}")
        return
    out = out or f"{base}_corner.png"
    corner_all_dims(kind, tgt, X, w, out, arm)
    print(f"  {os.path.basename(out)}  (d={d}, all {d} dims, from {src})")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cache_only = "--cache-only" in sys.argv
    tgt = args[0]
    if os.path.isdir(tgt):
        files = [f for f in sorted(glob.glob(os.path.join(tgt, "*.json")))
                 if not f.endswith("_iters.json")]
        verb = "caching read-outs for" if cache_only else "replotting"
        print(f"{verb} {len(files)} runs from {tgt}")
        for f in files:
            try:
                one(f, cache_only=cache_only)
            except Exception as exc:
                print(f"  [warn] {os.path.basename(f)}: {exc!r}")
    else:
        one(tgt, args[1] if len(args) > 1 else None, cache_only=cache_only)


if __name__ == "__main__":
    main()
