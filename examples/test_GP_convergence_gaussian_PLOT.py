"""
This script tests the goodness of the Gaussian KL convergence criterion, using different
implementations.

In particular, tests the accuracy of the covariance recovered (since for well-recovered
covariance matrix, all criteria would be equivalent).
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, random_correlation
from numpy.random import default_rng
import warnings
from time import time
import pickle
import matplotlib.pyplot as plt

# GPry things needed for building the model
from gpry.mpi import is_main_process, mpi_comm
from cobaya.model import get_model
from gpry.plots import getdist_add_training
from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt
from cobaya.run import run
from gpry.run import mcmc
import gpry.run

dim = 20

# Number of times that each combination of d and zeta is run with different
# gaussians
n_repeats = 5
rminusone = 0.01
plot_every_N = 50

# Ratio of prior size to (2x)std of mode
prior_size_in_std = 3

# Print always full dataframes
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

verbose = 1

######################


for n_r in range(n_repeats):

    #########################
    ### SETUP THE PROBLEM ###
    #########################
    # Create random gaussian
    if is_main_process:
        rng = default_rng()
        std = rng.uniform(size=dim)
        eigs = rng.uniform(size=dim)
        eigs = eigs / np.sum(eigs) * dim
        corr = random_correlation.rvs(eigs) if dim > 1 else [[1]]
        cov = np.multiply(np.outer(std, std), corr)
        mean = np.zeros_like(std)
    mean, std, cov = mpi_comm.bcast((mean, std, cov) if is_main_process else None)
    rv = multivariate_normal(mean, cov)
    # Interface with Gpry
    input_params = [f"x_{d}" for d in range(dim)]

    def f(**kwargs):
        X = [kwargs[p] for p in input_params]
        return np.log(rv.pdf(X))

    # Define the likelihood and the prior of the model
    info = {"likelihood": {"f": {"external": f,
                                 "input_params": input_params}}}
    param_dict = {}
    for d, p_d in enumerate(input_params):
        param_dict[p_d] = {
            "prior": {"min": mean[d] - prior_size_in_std * std[d],
                      "max": mean[d] + prior_size_in_std * std[d]}}
    info["params"] = param_dict
    # Initialise stuff
    model = get_model(info)
    
    ###############################
    ### RUN THE COMPARISON MCMC ###
    ###############################
    info_run = info.copy()
    info_run['sampler'] = {"mcmc": {"Rminus1_stop": rminusone, "max_tries": 10000}}
    updated_info1, sampler1 = run(info_run)
    s1 = sampler1.products()["sample"]
    gdsamples1 = MCSamplesFromCobaya(updated_info1, s1)

    def plot_if_necessary(gpr, old_gpr, new_X, new_y, convergence):
      n_conv = convergence.criterion_value(gpr)
      if gpr.n_accepted_evals % plot_every_N != 0 and not (n_conv >= convergence.ncorrect):
        return
      updated_info2, sampler2 = mcmc(model, gpr, options = {"mcmc": {"Rminus1_stop": rminusone, "max_tries": 10000}})
      s2 = sampler2.products()["sample"]
      gdsamples2 = MCSamplesFromCobaya(updated_info2, s2)

      gdplot = gdplt.get_subplot_plotter(width_inch=5)
      gdplot.triangle_plot([gdsamples1,gdsamples2], [f"x_{i}" for i in range(dim)], filled=[True,False], legend_labels = ['TRUE', 'GP'])
      #plt.show()
      plt.savefig(f"triangle_{dim}d_repeat{n_r}_{gpr.n_accepted_evals}.pdf")
      plt.close()
      del s2, sampler2, gdsamples2, gdplot

    gpry.run.run(model, convergence_criterion="CorrectCounter", verbose=verbose,callback = plot_if_necessary,options={'max_accepted':2000, 'max_points':10000})
