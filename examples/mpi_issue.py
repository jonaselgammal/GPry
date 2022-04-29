"""
Tests gaussians in different dimensions with different numbers of Kriging
Believer steps to see how fast they converge.
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, random_correlation
import scipy.stats
from numpy.random import default_rng
import warnings
from time import time
import pickle
import sys
import math

# GPry things needed for building the model
from gpry.mpi import mpi_comm, mpi_size, mpi_rank, is_main_process, get_random_state, \
    split_number_for_parallel_processes, multiple_processes, multi_gather_array
from gpry.plots import getdist_add_training, plot_convergence
from gpry.tools import kl_norm
from gpry.run import run, mcmc, mc_sample_from_gp

import getdist.plots as gdplt
from getdist.gaussian_mixtures import GaussianND
from getdist.mcsamples import MCSamplesFromCobaya

from cobaya.model import get_model
from cobaya.run import run as cobaya_run

import matplotlib.pyplot as plt
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

# Runs likelihoods with n-dimensions and m kriging believer steps to compare performance.
# Run until correctcounter says it's converged and to be safe also create triangle plot
# and calculate KL divergence.
fname = "test"
dims =  [2, 4, 6, 8, 10] # 8, 10, 12, 16 # dimensions to loop over
kriging_steps = [4, 8]

# Give option to turn on/off different parts of the code
n_repeats = 15 # number of repeats for checking convergence
prior_size_in_std = 5.
rminusone = 0.01 # R-1 for the MCMC
plot_final_contours = True # whether to plot final contours
info_text_in_plot = True

verbose = 3 # Verbosity of the BO loop

final_dict = {}

for d in dims:
    final_dict[d] = {}
    dim_dict = final_dict[d]
    for k in kriging_steps:
        dim_dict[k] = {"n_accepted": [], "n_total":[],
            "correct_counter_value":[], "KL_gauss_true_wrt_gp": [],
            "KL_gauss_gp_wrt_true":[]}
        kriging_dict = dim_dict[k]
        for n_r in range(n_repeats):
            if is_main_process:
                rng = default_rng()
                std = rng.uniform(size=d)
                eigs = rng.uniform(size=d)
                eigs = eigs / np.sum(eigs) * d
                corr = random_correlation.rvs(eigs) if d > 1 else [[1]]
                cov = np.multiply(np.outer(std, std), corr)
                mean = np.zeros_like(std)
            mean, std, cov = mpi_comm.bcast((mean, std, cov) if is_main_process else None)
            print(mean)
            print(std)

            input_params = [f"x_{di}" for di in range(d)]
            rv = multivariate_normal(mean, cov)
            def f(**kwargs):
                X = [kwargs[p] for p in input_params]
                return np.log(rv.pdf(X))

            # Interface with Gpry
            # Define the likelihood and the prior of the model
            info = {"likelihood": {"f": {"external": f,
                                         "input_params": input_params}}}
            param_dict = {}
            for di, p_d in enumerate(input_params):
                param_dict[p_d] = {
                    "prior": {"min": mean[di] - prior_size_in_std * std[di],
                              "max": mean[di] + prior_size_in_std * std[di]}}
            info["params"] = param_dict
            # info = mpi_comm.bcast(info if is_main_process else None)
            print(info)

            model = get_model(info)
            # model = mpi_comm.bcast(model if is_main_process else None)
            print(model)

            # Run model until it's converged
            _, gpr, acquisition, convergence, options = run(model, verbose=verbose, options={"n_points_per_acq":k}, convergence_criterion="CorrectCounter", convergence_options={"threshold": 0.01}) # , convergence_criterion="ConvergenceCriterionGaussianMCMC", convergence_options={"threshold": 0.5/d}

            # Run MCMC
            updated_info2, sampler2 = mcmc(model, gpr, options = {"mcmc": {"Rminus1_stop": rminusone, "max_tries": 10000}})

            # Plot triangle plot for comparison
            if is_main_process:
                s2 = sampler2.products()["sample"]
                gdsamples2 = MCSamplesFromCobaya(updated_info2, s2)
                s2.detemper()
                s2mean, s2cov = s2.mean(), s2.cov()
                kriging_dict['KL_gauss_true_wrt_gp'].append(kl_norm(mean,cov, s2mean, s2cov))
                kriging_dict['KL_gauss_gp_wrt_true'].append(kl_norm(s2mean,s2cov, mean, cov))
                kriging_dict['correct_counter_value'].append(convergence.values[-1])
                kriging_dict['n_accepted'].append(gpr.n_accepted_evals)
                kriging_dict['n_total'].append(gpr.n_total_evals)
                del s2, sampler2
                if plot_final_contours:
                    gauss_plot=GaussianND(mean, cov, names=input_params)
                    gdplot = gdplt.get_subplot_plotter(width_inch=5)
                    gdplot.triangle_plot([gauss_plot, gdsamples2], list(info["params"]),
                                         filled=[False, True], legend_labels=['True', 'GPry'])
                    getdist_add_training(gdplot, model, gpr)
                    if info_text_in_plot:
                        n_d = model.prior.d()
                        info_text = ("$n_{tot}=%i$\n $n_{accepted}=%i$\n $d_{KL}=%.2e$\n $d_{CC}=%.2e$"
                            %(gpr.n_total_evals, gpr.n_accepted_evals, kl_norm(mean,cov, s2mean, s2cov), convergence.values[-1]))
                        ax = gdplot.get_axes(ax=(0, int(math.ceil((n_d-1)/2.))))
                        gdplot.add_text_left(info_text, x=0.2, y=0.5, ax=(0, int(math.ceil((n_d-1)/2.)))) #, transform=ax.transAxes
                        ax.axis('off')
                    plt.savefig(f"{fname}/d{d}_m{k}_t{n_r}.pdf")
                    plt.close()

if multiple_processes:
    if is_main_process:
        with open(f"{fname}/data.pkl", "wb") as f:
            pickle.dump(final_dict, f, pickle.HIGHEST_PROTOCOL)
else:
    with open(f"{fname}/data.pkl", "wb") as f:
        pickle.dump(final_dict, f, pickle.HIGHEST_PROTOCOL)
