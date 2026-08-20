"""
Aggregate the cluster run outputs (eval.json + meta.json) into tidy tables and
the paper figures. Reproducible: point RAW at the pulled run dirs.

Multimodal (Tier B): runs_mm/mm_d<d>_sep<sep>_wr<wr>_<sampler>_seed<seed>/
Unimodal  (recovery scaling): runs/d<d>_seed<seed>/
"""
import json, glob, os, re, csv
import numpy as np
from scipy.stats import chi2

RAW = os.path.join(os.path.dirname(__file__), "raw")
OUT = os.path.dirname(__file__)
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)


def chi_tail(d, thr=5.0):
    """Expected fraction of PERFECT unit-Gaussian samples with ||x-mode||>thr."""
    return float(chi2.sf(thr ** 2, d))


# --------------------------------------------------------------------------- #
def load_multimodal():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "runs_mm/mm_*/eval.json"))):
        tag = os.path.basename(os.path.dirname(f))
        m = re.match(r"mm_d(\d+)_sep(\w+?)_wr(\w+)_(\w+)_seed(\d+)", tag)
        d, sep, samp, seed = int(m[1]), float(m[2].replace("p", ".")), m[4], int(m[5])
        js = json.load(open(f))
        if not js:
            continue
        ck = sorted(js, key=lambda k: int(k[1:]))[-1]
        r = js[ck]
        meta = {}
        mp = os.path.join(os.path.dirname(f), "meta.json")
        if os.path.exists(mp):
            meta = json.load(open(mp))
        spur_corr = max(0.0, r["spurious_frac"] - chi_tail(d))
        rows.append(dict(d=d, sep=sep, sampler=samp, seed=seed, ckpt=int(ck[1:]),
                         n_modes=r["n_modes_found"], w_relerr=r["w_relerr"],
                         wass_x0=r["wass_x0"], spurious_raw=r["spurious_frac"],
                         spurious_corr=round(spur_corr, 4), ess=r["ess"],
                         div=r["n_divergent"],
                         wall_s=meta.get("wall_s"), n_final=meta.get("n_total_final")))
    return rows


def load_unimodal():
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "runs/d*_seed*/eval.json"))):
        tag = os.path.basename(os.path.dirname(f))
        m = re.match(r"d(\d+)_seed(\d+)", tag)
        d, seed = int(m[1]), int(m[2])
        js = json.load(open(f))
        if not js:
            continue
        ck = sorted(js, key=lambda k: int(k[1:]))[-1]
        r = js[ck]
        rows.append(dict(d=d, seed=seed, ckpt=int(ck[1:]), kl=r["kl_nuts"],
                         max_mean_sig=r["max_mean_in_sigma"],
                         std_relerr=r["std_relerr_nuts"], div=r["n_divergent"]))
    return rows


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    mm = load_multimodal(); uni = load_unimodal()
    write_csv(mm, os.path.join(OUT, "multimodal_tierB_seed1.csv"))
    write_csv(uni, os.path.join(OUT, "unimodal_recovery.csv"))

    # ---- console summary --------------------------------------------------- #
    print("=== MULTIMODAL Tier-B (seed 1), final checkpoint per config ===")
    print(f"{'d':>2} {'sep':>4} {'sampler':>9} {'modes':>6} {'w_relerr':>8} "
          f"{'wassx0':>7} {'spur_raw':>8} {'spur_corr':>9} {'wall_s':>7}")
    for r in sorted(mm, key=lambda x: (x["d"], x["sep"], x["sampler"])):
        print(f"{r['d']:>2} {r['sep']:>4} {r['sampler']:>9} {r['n_modes']:>4}/2 "
              f"{r['w_relerr']:>8} {r['wass_x0']:>7} {r['spurious_raw']:>8} "
              f"{r['spurious_corr']:>9} {str(r['wall_s']):>7}")

    print("\n=== TIMING: median wall_s per run, by (d, sampler) ===")
    for d in sorted(set(r["d"] for r in mm)):
        for s in ("nuts", "ultranest"):
            ws = [r["wall_s"] for r in mm if r["d"] == d and r["sampler"] == s
                  and r["wall_s"] is not None]
            nf = [r["n_final"] for r in mm if r["d"] == d and r["sampler"] == s
                  and r["n_final"] is not None]
            if ws:
                print(f"  d={d:>2} {s:>9}: median={np.median(ws):8.1f}s  "
                      f"range=[{min(ws):.0f},{max(ws):.0f}]  n_final~{int(np.median(nf))}")
        ws_n = [r["wall_s"] for r in mm if r["d"] == d and r["sampler"] == "nuts" and r["wall_s"]]
        ws_u = [r["wall_s"] for r in mm if r["d"] == d and r["sampler"] == "ultranest" and r["wall_s"]]
        if ws_n and ws_u:
            print(f"     -> ultranest/nuts wall ratio (median) = {np.median(ws_u)/np.median(ws_n):.1f}x")

    print("\n=== UNIMODAL recovery scaling (final ckpt) ===")
    for r in sorted(uni, key=lambda x: (x["d"], x["seed"])):
        print(f"  d={r['d']:>2} seed{r['seed']} n={r['ckpt']}: KL={r['kl']} "
              f"maxmean={r['max_mean_sig']}sig std_relerr={r['std_relerr']} div={r['div']}")


if __name__ == "__main__":
    main()
