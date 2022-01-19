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
from gpry.tools import kl_norm

dim = 24

# Number of times that each combination of d and zeta is run with different
# gaussians
n_repeats = 5
rminusone = 0.01
print_every_N = 80

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
    #gdsamples1 = MCSamplesFromCobaya(updated_info1, s1)
    s1.detemper()
    s1mean, s1cov = s1.mean(), s1.cov()
    x_values = s1.data[s1.sampled_params]
    logp = s1['minuslogpost']
    weights = s1['weight']
    history = {'KL_truth':[],'KL_gauss_wrt_true':[],'KL_gauss_wrt_gp':[],'step':[]}
    true_hist = {'KL_gauss_wrt_true' : kl_norm(mean,cov, s1mean, s1cov),'KL_gauss_wrt_gp':kl_norm(s1mean,s1cov, mean, cov),'KL_truth':0} # The truth doesn't change, so the history has len==1

    def print_if_necessary(gpr, old_gpr, new_X, new_y, convergence):
      n_conv = convergence.criterion_value(gpr)
      if gpr.n_accepted_evals % print_every_N != 0 and not (n_conv >= convergence.ncorrect):
        return
      updated_info2, sampler2 = mcmc(model, gpr, options = {"mcmc": {"Rminus1_stop": rminusone, "max_tries": 10000}})
      print("BEFORE GETSAMPLES")
      s2 = sampler2.products()["sample"]
      s2.detemper()
      print("BEFORE MEANCOV")
      s2mean, s2cov = s2.mean(), s2.cov()
      history['KL_gauss_wrt_true'].append(kl_norm(mean,cov, s2mean, s2cov))
      history['KL_gauss_wrt_gp'].append(kl_norm(s2mean,s2cov, mean, cov))
      print("BEFORE DEL")
      del s2mean, s2cov, s2, sampler2
      print("BEFORE PREDICT")
      y_values = []
      for i in range(0,len(x_values), 256):
        y_values = np.concatenate([y_values,gpr.predict(x_values[i:i+256])])
      logq = np.array(y_values)
      print(logq.shape)
      print(logp.shape)
      print("AFTER PREDICT")
      history['KL_truth'].append(np.sum(weights*(logp-logq))/np.sum(weights))
      history['step'].append(gpr.n_accepted_evals)
      print("AFTER KL compute")

      print(history)
      if n_conv == convergence.ncorrect:
        #np.save("history_{}d_repeat{}_{}.npy".format(dim,n_r,gpr.n_accepted_evals),history)
        data ={'true_hist' : true_hist, 'hist':history,'n_accepted':gpr.n_accepted_evals,'n_tot':gpr.n_total_evals}
        with open("history_{}d_repeat{}.pkl".format(dim,n_r), "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    gpry.run.run(model, convergence_criterion="CorrectCounter", verbose=verbose,callback = print_if_necessary,options={'max_accepted':2000, 'max_points':10000})
    
