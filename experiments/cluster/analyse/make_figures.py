"""Build the paper figures from the aggregated CSVs. Saves PDF+PNG to figures/."""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(__file__)
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
C = {"nuts": "tab:red", "ultranest": "tab:blue"}
LB = {"nuts": "NUTS", "ultranest": "nested (UltraNest)"}


def read(path):
    with open(path) as fh:
        return [ {k: (float(v) if v not in ("", "None") and _num(v) else v)
                  for k, v in row.items()} for row in csv.DictReader(fh) ]


def _num(v):
    try: float(v); return True
    except: return False


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=140)
    plt.close(fig)


mm = read(os.path.join(OUT, "multimodal_tierB_seed1.csv"))
uni = read(os.path.join(OUT, "unimodal_recovery.csv"))


# --- Fig 1: cost vs dimension (the headline) ------------------------------- #
def fig_timing():
    ds = sorted(set(int(r["d"]) for r in mm))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for s in ("nuts", "ultranest"):
        med, lo, hi = [], [], []
        for d in ds:
            w = [r["wall_s"] for r in mm if int(r["d"]) == d and r["sampler"] == s
                 and isinstance(r["wall_s"], float)]
            med.append(np.median(w)); lo.append(min(w)); hi.append(max(w))
        med = np.array(med) / 60
        ax.plot(ds, med, "o-", color=C[s], label=LB[s], lw=1.8)
        ax.fill_between(ds, np.array(lo) / 60, np.array(hi) / 60, color=C[s], alpha=0.15)
    ax.set_xlabel("dimension $d$"); ax.set_ylabel("wall time per run [min]")
    ax.set_yscale("log"); ax.set_xticks(ds)
    ax.set_title("Acquisition cost at matched budget (2-mode target)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    save(fig, "fig_timing_vs_dim")


# --- Fig 2: weight error & marginal fidelity vs separation ------------------ #
def fig_accuracy():
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True)
    for col, d in enumerate((8, 16)):
        for s in ("nuts", "ultranest"):
            pts = sorted([r for r in mm if int(r["d"]) == d and r["sampler"] == s],
                         key=lambda r: r["sep"])
            sep = [r["sep"] for r in pts]
            axes[0, col].plot(sep, [100 * r["w_relerr"] for r in pts], "o-",
                              color=C[s], label=LB[s])
            axes[1, col].plot(sep, [r["wass_x0"] for r in pts], "o-", color=C[s])
        axes[0, col].set_title(f"$d={d}$")
        axes[1, col].set_xlabel(r"mode separation [$\sigma$]")
    axes[0, 0].set_ylabel("mode-weight error [%]")
    axes[1, 0].set_ylabel(r"$W_1$ on bimodal axis")
    axes[0, 0].legend(fontsize=8)
    for ax in axes.ravel(): ax.grid(alpha=0.3)
    fig.suptitle("Both samplers find both modes; NUTS fidelity $\\geq$ nested, esp. at large separation")
    save(fig, "fig_accuracy_vs_sep")


# --- Fig 3: unimodal recovery scaling -------------------------------------- #
def fig_unimodal():
    ds = sorted(set(int(r["d"]) for r in uni))
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    kl_med = [np.median([r["kl"] for r in uni if int(r["d"]) == d]) for d in ds]
    kl_lo = [min(r["kl"] for r in uni if int(r["d"]) == d) for d in ds]
    kl_hi = [max(r["kl"] for r in uni if int(r["d"]) == d) for d in ds]
    ax.plot(ds, kl_med, "o-", color="tab:green", lw=1.8, label="median over 3 seeds")
    ax.fill_between(ds, kl_lo, kl_hi, color="tab:green", alpha=0.2)
    ax.axhline(0.05, ls="--", color="grey", lw=1, label="recovery threshold (0.05)")
    ax.set_xlabel("dimension $d$"); ax.set_ylabel(r"$D_{\mathrm{KL}}(\mathrm{GP}\,\|\,\mathrm{truth})$")
    ax.set_title("Unimodal posterior recovery with NUTS acquisition")
    ax.set_xticks(ds); ax.legend(); ax.grid(alpha=0.3)
    save(fig, "fig_unimodal_scaling")


fig_timing(); fig_accuracy(); fig_unimodal()
print("figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    print("  ", f)
