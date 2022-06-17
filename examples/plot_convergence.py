"""
Plots convergence statistics for runs generated with run_likelihood.py
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, random_correlation
from numpy.random import default_rng
import warnings
from time import time
import pickle

# GPry things needed for building the model
from gpry.acquisition_functions import LogExp
from gpry.gpr import GaussianProcessRegressor
from gpry.kernels import RBF, ConstantKernel as C
from gpry.preprocessing import Normalize_y, Normalize_bounds
from gpry.convergence import KL_from_draw, KL_from_MC_training, KL_from_draw_approx, \
    ConvergenceCheckError, KL_from_draw_approx_alt, ConvergenceCriterionGaussianMCMC
from gpry.gp_acquisition import GP_Acquisition
from gpry.tools import kl_norm
from gpry.mpi import is_main_process, mpi_comm
from cobaya.model import get_model
from gpry.plots import getdist_add_training, plot_convergence
import matplotlib.pyplot as plt
from cobaya.model import get_model
from gpry.tools import kl_norm
import getdist.plots as gdplt
from cobaya.run import run as cobaya_run
from getdist.mcsamples import MCSamplesFromCobaya
from gpry.run import run, mcmc

import matplotlib
rcparams = {
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'font.size': 14, # was 10
    'legend.fontsize': 14, # was 10
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
}
matplotlib.rcParams.update(rcparams)
# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Supported likelihoods:
# curved_degeneracy, loggaussian, ring, Rosenbrock, Himmelblau, spike
likelihood = "Rosenbrock_2d"

plot_approximate_kl = False

with open(f"{likelihood}/run_params.pkl", "rb") as f:
    run_params = pickle.load(f)

n_repeats = run_params["n_repeats"]
n_accepted_evals = run_params["n_accepted_evals"]


# Correct counter
### Here we need to extract the right iterations to build the mean

fig, ax = plt.subplots()
for n_r in range(n_repeats):
    # Read in convergence history
    with open(f"{likelihood}/history_try_{n_r}.pkl", "rb") as f:
        history = pickle.load(f)["hist"]
    correct_counter_value = history["CorrectCounter_value"]
    total_evals = history["total_evals_correct_counter"]
    accepted_evals = history["accepted_evals_correct_counter"]
    total_evals_for_convergence = history['total_evals_for_convergence']
    accepted_evals_for_convergence = history['accepted_evals_for_convergence']
    plt.plot(accepted_evals, correct_counter_value, linestyle="-", alpha=0.7)
    color = plt.gca().lines[-1].get_color()
    if accepted_evals_for_convergence is not None:
        ax.axvline(x=accepted_evals_for_convergence, color=color, linestyle="-.")
plt.grid()
plt.yscale("log")
plt.ylabel("Correct counter value")
plt.xlabel("N accepted steps")
plt.savefig(f"{likelihood}/total_history_correct_counter.pdf")

# KL divergences
fig, ax = plt.subplots()
for n_r in range(n_repeats):
    # Read in convergence history
    with open(f"{likelihood}/history_try_{n_r}.pkl", "rb") as f:
        history = pickle.load(f)["hist"]
    KL_gauss_true_wrt_gp = history["KL_gauss_true_wrt_gp"]
    KL_gauss_gp_wrt_true = history["KL_gauss_gp_wrt_true"]
    KL_full_true_wrt_gp = history["KL_full_true_wrt_gp"]
    total_evals = history["total_evals_kl"]
    accepted_evals = history["accepted_evals_kl"]
    total_evals_for_convergence = history['total_evals_for_convergence']
    accepted_evals_for_convergence = history['accepted_evals_for_convergence']
    plt.plot(accepted_evals, KL_full_true_wrt_gp, linestyle="-", alpha=0.7)
    color = plt.gca().lines[-1].get_color()
    if plot_approximate_kl:
        plt.plot(accepted_evals, KL_gauss_gp_wrt_true, color=color, linestyle=":", alpha=0.7)
        plt.plot(accepted_evals, KL_gauss_true_wrt_gp, color=color, linestyle="--", alpha=0.7)
    if accepted_evals_for_convergence is not None:
        ax.axvline(x=accepted_evals_for_convergence, color=color, linestyle="-.")
plt.grid()
plt.yscale("log")
plt.ylabel("KL divergence")
plt.xlabel("N accepted steps")
plt.savefig(f"{likelihood}/total_history_kl.pdf")
plt.close()

################################################################################

# Means for KL divergences
with open(f"{likelihood}/history_total.pkl", "rb") as f:
    history = pickle.load(f)

KL_gauss_true_wrt_gp_total = history["KL_gauss_true_wrt_gp"]
KL_gauss_gp_wrt_true_total = history["KL_gauss_gp_wrt_true"]
KL_full_true_wrt_gp_total = history["KL_full_true_wrt_gp"]
Correct_counter_total = history["Correct_counter"]
total_evals_for_convergence_total = history['total_evals_for_convergence']
accepted_evals_for_convergence_total = history['accepted_evals_for_convergence']
l_con, m_con, u_con = np.nanquantile(accepted_evals_for_convergence_total, [0.25, 0.5, 0.75])

accepted_evals_all = history["accepted_evals"]


# Kl divergences
KL_gauss_true_wrt_gp_total[np.isinf(KL_gauss_true_wrt_gp_total)] = np.nan
KL_gauss_gp_wrt_true_total[np.isinf(KL_gauss_gp_wrt_true_total)] = np.nan
KL_full_true_wrt_gp_total[np.isinf(KL_full_true_wrt_gp_total)] = np.nan
KL_full_true_wrt_gp_total = np.abs(KL_full_true_wrt_gp_total)
fig, ax = plt.subplots()
lower, median, upper = np.nanquantile(KL_full_true_wrt_gp_total, [0.25, 0.5, 0.75], axis=1)
plt.plot(accepted_evals, median, lw=1.5, zorder=9, label='MCMC True vs GP')
color = plt.gca().lines[-1].get_color()
plt.fill_between(accepted_evals, lower, upper, alpha=0.2, color=color, zorder=8)
if plot_approximate_kl:
    lower, median, upper = np.nanquantile(KL_gauss_true_wrt_gp_total, [0.25, 0.5, 0.75], axis=1)
    plt.plot(accepted_evals, median, lw=1.5, zorder=9, label='Gauss True vs GP')
    color = plt.gca().lines[-1].get_color()
    plt.fill_between(accepted_evals, lower, upper, alpha=0.2, color=color, zorder=8)
    lower, median, upper = np.nanquantile(KL_gauss_gp_wrt_true_total, [0.25, 0.5, 0.75], axis=1)
    plt.plot(accepted_evals, median, lw=1.5, zorder=9, label='Gauss GP vs True')
    color = plt.gca().lines[-1].get_color()
    plt.fill_between(accepted_evals, lower, upper, alpha=0.2, color=color, zorder=8)

ax.axvline(x=m_con, color="grey", linestyle=":", zorder=11)
ax.axvspan(l_con, u_con, alpha=0.2, color="grey", zorder=10)

plt.grid()
if plot_approximate_kl:
    plt.legend()
plt.yscale("log")
plt.ylabel("KL divergence")
plt.xlabel("N accepted steps")
plt.savefig(f"{likelihood}/mean_history_kl.pdf")

# Correct counter
Correct_counter_total[np.isinf(Correct_counter_total)] = np.nan

fig, ax = plt.subplots()
lower, median, upper = np.nanquantile(Correct_counter_total, [0.25, 0.5, 0.75], axis=1)
plt.plot(accepted_evals, median, lw=1.5, zorder=9)
color = plt.gca().lines[-1].get_color()
plt.fill_between(accepted_evals, lower, upper, alpha=0.2, color=color, zorder=8)

ax.axvline(x=m_con, color="grey", linestyle=":", zorder=11)
ax.axvspan(l_con, u_con, alpha=0.2, color="grey", zorder=10)

plt.grid()
plt.yscale("log")
plt.ylabel("Relative difference")
plt.xlabel("N accepted steps")
plt.savefig(f"{likelihood}/mean_history_correct_counter.pdf")

# both in one plot
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax1 = plt.subplots()
lower, median, upper = np.nanquantile(KL_full_true_wrt_gp_total, [0.25, 0.5, 0.75], axis=1)
color = colors[0]
ax1.plot(accepted_evals, median, lw=1.5, zorder=9, color=color)
ax1.fill_between(accepted_evals, lower, upper, alpha=0.2, zorder=8, color=color)
ax1.set_yscale("log")
ax1.set_xlabel("N accepted steps")
ax1.set_ylabel('Kl-divergence', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
lower, median, upper = np.nanquantile(Correct_counter_total, [0.25, 0.5, 0.75], axis=1)
color = colors[1]
ax2.plot(accepted_evals, median, lw=1.5, zorder=9, color=color)
ax2.fill_between(accepted_evals, lower, upper, alpha=0.2, zorder=8, color=color)
ax2.set_yscale("log")
ax2.set_ylabel('Relative difference', color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.axvline(x=m_con, color="grey", linestyle=":", zorder=11)
ax1.axvspan(l_con, u_con, alpha=0.2, color="grey", zorder=10)

plt.grid()
plt.tight_layout()
plt.savefig(f"{likelihood}/mean_history_both.pdf")
