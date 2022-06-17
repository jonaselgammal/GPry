# THIS IS THE ONE!!!!!

"""
runs the curved degeneracy example and tracks convergence, plots ...
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

# GPry things needed for building the model
from gpry.acquisition_functions import LogExp
from gpry.gpr import GaussianProcessRegressor
from gpry.kernels import RBF, ConstantKernel as C
from gpry.preprocessing import Normalize_y, Normalize_bounds
from gpry.convergence import KL_from_draw, KL_from_MC_training, KL_from_draw_approx, \
    ConvergenceCheckError, KL_from_draw_approx_alt, ConvergenceCriterionGaussianMCMC, \
    CorrectCounter
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

# Print always full dataframes
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Supported likelihoods:
# curved_degeneracy, loggaussian, ring, Rosenbrock, Himmelblau, spike, gaussian
likelihood = "gaussian"
dim = 15 # The dimensionality of the likelihood, only relevant for loggaussian, Rosenbrock, Himmelblau
input_params = [f"x_{d}" for d in range(dim)]

# Give option to turn on/off different parts of the code
n_repeats = 15 # number of repeats for checking convergence
rminusone = 0.01 # R-1 for the MCMC
min_points_for_kl = 60 # minimum number of points for calculating the KL divergence
evaluate_convergence_every_n = 30 # How many points should be sampled before running the KL
n_accepted_evals = 401 # Number of accepted steps before killing
plot_intermediate_contours = True # whether to plot intermediate (non-converged)
plot_final_contours = True # whether to plot final contours
info_text_in_plot = True

verbose = 1 # Verbosity of the BO loop

accepted_evals = np.arange(min_points_for_kl, n_accepted_evals, evaluate_convergence_every_n)

if likelihood == "curved_degeneracy":
    # curved_degeneracy
    def curved_degeneracy(x_0, x_1):
        return  -(10*(0.45-x_0))**2./4. - (20*(x_1/4.-x_0**4.))**2.

    # Construct model instance
    info = {"likelihood": {"curved_degeneracy": curved_degeneracy}}
    info["params"] = {
        "x_0": {"prior": {"min": -0.5, "max": 1.5}},
        "x_1": {"prior": {"min": -0.5, "max": 2.}}
        }

elif likelihood == "loggaussian":
    # loggaussian
    ndim_log = 2 # Number of log dims
    prior_size_in_std = 5
    rng = default_rng()
    std = rng.uniform(size=dim)
    eigs = rng.uniform(size=dim)
    eigs = eigs / np.sum(eigs) * dim
    corr = random_correlation.rvs(eigs) if dim > 1 else [[1]]
    cov = np.multiply(np.outer(std, std), corr)
    mean = np.zeros_like(std)
    rv = multivariate_normal(mean, cov)
    # Interface with Gpry
    def loggaussian(**kwargs):
        X = [kwargs[p] for p in input_params]
        for j in range(ndim_log):
           X[j] = 10**X[j]
        return np.log(rv.pdf(X))

    # model and prior
    info = {
        "likelihood": {
            "loggaussian": {
                "external":loggaussian,
                "input_params": input_params
            }
        }
    }
    info["params"] = {}
    # 10**(mean[k] + prior_size_in_std*std[k]) if k<ndim_log else
    for k, p in enumerate(input_params):
        p_max = (mean[k] + prior_size_in_std*std[k])
        p_min = (mean[k] - prior_size_in_std*std[k])
        info["params"][p] = {"prior": {"min": p_min, "max": p_max}}

elif likelihood == "ring":
    # ring
    mean_radius = 1
    std = 0.05
    offset = 0
    prior_size_in_std = 5
    def ring(x_0, x_1, mean_radius=mean_radius, std=std, offset=offset):
        return scipy.stats.norm.logpdf(np.sqrt(x_0**2 + x_1**2)+offset*x_0, loc=mean_radius, scale=std)
    # model and prior
    lower_x_0 = offset - mean_radius - prior_size_in_std*std
    upper_x_0 = offset + mean_radius + prior_size_in_std*std
    lower_x_1 = -1*mean_radius - prior_size_in_std*std
    upper_x_1 = mean_radius + prior_size_in_std*std
    info = {"likelihood": {"ring": ring}}
    info["params"] = {
        "x_0": {"prior": {"min": lower_x_0, "max": upper_x_0}},
        "x_1": {"prior": {"min": lower_x_1, "max": upper_x_1}}
        }

elif likelihood == "Himmelblau":
    def himmel(x,y,x1=11.,x2=7.):
        return 0.1*((x*x+y-x1)**2 + (x+y*y - x2)**2)
    def himmelblau(**kwargs):
        X = [kwargs[p] for p in input_params]
        if len(X)%2 ==0:
            chi2 = 0
            for i in range(0,len(X)//2,2):
                chi2+=himmel(X[i],X[i+1])
            return -.5 * chi2
        else:
            return himmelblau(X[:-1]) - 0.5*X[-1]**2
    def lkl_simple(X):
        chi2 = himmel(X[0],X[1])
        for i in range(2,len(X)):
            chi2 += X[i]**2
        return -.5 * chi2

    minx = -4
    maxx = 4
    info = {"likelihood": {"himmelblau": {"external": himmelblau,
                                 "input_params": input_params}}}
    param_dict = {}
    for d, p_d in enumerate(input_params):
        param_dict[p_d] = {
            "prior": {"min": minx,
                      "max": maxx}}
    info["params"] = param_dict

elif likelihood == "Rosenbrock":
    def rosen(x,y,a=1.,b=100.):
        return (a-x)**2+b*(y-x*x)**2
    def rosenbrock(**kwargs):
        X = [kwargs[p] for p in input_params]
        if len(X)%2 ==0:
            chi2 = 0
            for i in range(0,len(X),2):
                chi2+=rosen(X[i],X[i+1])
            return -0.5*chi2
        else:
            return rosenbrock(X[:-1]) - 0.5*X[-1]**2

    def lkl_simple(X):
        chi2 = rosen(X[0],X[1])
        for i in range(2,len(X)):
            chi2 += X[i]**2
        return -0.5*chi2
    minx = -4
    maxx = 4
    info = {"likelihood": {"rosenbrock": {"external": rosenbrock,
                                 "input_params": input_params}}}
    param_dict = {}
    for d, p_d in enumerate(input_params):
        param_dict[p_d] = {
            "prior": {"min": minx,
                      "max": maxx}}
    info["params"] = param_dict

elif likelihood == "spike":
    def sp(x,a=100.,b=2.):
        return -np.log(np.exp(-x*x)+(1.-np.exp(-b*b))*np.exp(-a*(x-b)**2))
    def spike(**kwargs):
        X = [kwargs[p] for p in input_params]
        chi2 = 0
        for i in range(len(X)):
            chi2+=sp(X[i])
        return -0.5*chi2

    minx = -4.
    maxx = 4.

    info = {"likelihood": {"spike": {"external": spike,
                                 "input_params": input_params}}}
    param_dict = {}
    for d, p_d in enumerate(input_params):
        param_dict[p_d] = {
            "prior": {"min": minx,
                      "max": maxx}}
    info["params"] = param_dict

elif likelihood == "gaussian":
    prior_size_in_std = 7.
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

else:
    raise RuntimeError("Likelihood not supported...")


model = get_model(info)

# Get dimensionality and prior bounds
dim = model.prior.d()
prior_bounds = model.prior.bounds()

###############################
### RUN THE COMPARISON MCMC ###
###############################
info_run = info.copy()
info_run['sampler'] = {"mcmc": {"Rminus1_stop": rminusone, "max_tries": 10000}}
updated_info1, sampler1 = cobaya_run(info_run)
s1 = sampler1.products()["sample"]
gdsamples1 = MCSamplesFromCobaya(updated_info1, s1)
s1.detemper()
s1mean, s1cov = s1.mean(), s1.cov()
x_values = s1.data[s1.sampled_params]
logp = s1['minuslogpost']
logp = -logp
weights = s1['weight']

# prepare the array for storing all convergence statistics
KL_gauss_true_wrt_gp_total = np.empty((len(accepted_evals), n_repeats))
KL_gauss_true_wrt_gp_total[:] = np.nan
KL_gauss_gp_wrt_true_total = np.empty((len(accepted_evals), n_repeats))
KL_gauss_gp_wrt_true_total[:] = np.nan
KL_full_true_wrt_gp_total = np.empty((len(accepted_evals), n_repeats))
KL_full_true_wrt_gp_total[:] = np.nan
Correct_counter_total = np.empty((len(accepted_evals), n_repeats))
Correct_counter_total[:] = np.nan
total_evals_for_convergence_total = np.empty(n_repeats)
total_evals_for_convergence_total[:] = np.nan
accepted_evals_for_convergence_total = np.empty(n_repeats)
accepted_evals_for_convergence_total[:] = np.nan

for n_r in range(n_repeats):
    history = {'KL_gauss_true_wrt_gp':[],'KL_gauss_gp_wrt_true':[],'KL_full_true_wrt_gp':[],
        'CorrectCounter_value':[], 'total_evals_kl':[],'accepted_evals_kl':[],
        'total_evals_correct_counter':[],'accepted_evals_correct_counter':[],
        'total_evals_for_convergence': np.nan, 'accepted_evals_for_convergence': np.nan}
    counter = 0
    is_converged = False
    corr_counter_conv = CorrectCounter(model.prior, {})
    def print_and_plot(model, gpr, gp_acquisition, convergence, options,
        old_gpr, new_X, new_y, pred_y):
        global counter
        global accepted_evals
        global logp
        global weights
        global KL_gauss_true_wrt_gp_total
        global KL_gauss_gp_wrt_true_total
        global KL_full_true_wrt_gp_total
        global Correct_counter_total
        global total_evals_for_convergence_total
        global accepted_evals_for_convergence_total
        global corr_counter_conv
        global is_converged
        global info_text_in_plot

        # If CorrectCounter is not converged check convergence
        convergence = corr_counter_conv.is_converged(gpr, gp_2=old_gpr, new_X=new_X, new_y=new_y, pred_y=pred_y)
        history['CorrectCounter_value'] = corr_counter_conv.values
        history['total_evals_correct_counter'] = corr_counter_conv.n_posterior_evals
        history['accepted_evals_correct_counter'] = corr_counter_conv.n_accepted_evals
        if not is_converged and convergence:
            print("CorrectCounter has converged!")
            history["total_evals_for_convergence"] = gpr.n_total_evals
            history["accepted_evals_for_convergence"] = gpr.n_accepted_evals
            total_evals_for_convergence_total[n_r] = gpr.n_total_evals
            accepted_evals_for_convergence_total[n_r] = gpr.n_accepted_evals
            is_converged = True
            if plot_final_contours:
                updated_info2, sampler2 = mcmc(model, gpr, options = {"mcmc": {"Rminus1_stop": rminusone, "max_tries": 10000}})
                # print("BEFORE GETSAMPLES")
                s2 = sampler2.products()["sample"]
                gdsamples2 = MCSamplesFromCobaya(updated_info2, s2)
                s2.detemper()
                y_values = []
                for i in range(0,len(x_values), 256):
                    y_values = np.concatenate([y_values,gpr.predict(x_values[i:i+256])])
                logq = np.array(y_values)
                mask = np.isfinite(logq)
                logp2 = logp[mask]
                logq2 = logq[mask]
                weights2 = weights[mask]
                kl = np.sum(weights*(logp-logq))/np.sum(weights)
                del s2, sampler2
                gdplot = gdplt.get_subplot_plotter(width_inch=5)
                gdplot.triangle_plot([gdsamples1, gdsamples2], list(info["params"]),
                                     filled=[False, True], legend_labels=['MCMC', 'GPry'])
                getdist_add_training(gdplot, model, gpr)
                if info_text_in_plot:
                    n_d = model.prior.d()
                    info_text = ("$n_{tot}=%i$\n $n_{accepted}=%i$\n $d_{KL}=%.2e$\n $d_{CC}=%.2e$"
                        %(gpr.n_total_evals, gpr.n_accepted_evals, kl, corr_counter_conv.values[-1]))
                    ax = gdplot.get_axes(ax=(0, n_d-1))
                    gdplot.add_text_left(info_text, x=0.2, y=0.5, ax=(0, n_d-1)) #, transform=ax.transAxes
                    ax.axis('off')
                plt.savefig(f"{likelihood}/triangle_try_{n_r}_final.pdf")
                plt.close()

        if len(accepted_evals) > counter:
            if gpr.n_accepted_evals < accepted_evals[counter]: # and not (n_conv >= convergence.ncorrect):
                return
        else:
            return
        # Add correct counter value
        Correct_counter_total[counter, n_r] = corr_counter_conv.values[-1]
        # Calculate KL divergence
        updated_info2, sampler2 = mcmc(model, gpr, options = {"mcmc": {"Rminus1_stop": rminusone, "max_tries": 10000}})
        # print("BEFORE GETSAMPLES")
        s2 = sampler2.products()["sample"]
        gdsamples2 = MCSamplesFromCobaya(updated_info2, s2)
        s2.detemper()
        s2mean, s2cov = s2.mean(), s2.cov()
        history['KL_gauss_true_wrt_gp'].append(kl_norm(s1mean,s1cov, s2mean, s2cov))
        history['KL_gauss_gp_wrt_true'].append(kl_norm(s2mean,s2cov, s1mean, s1cov))
        y_values = []
        for i in range(0,len(x_values), 256):
            y_values = np.concatenate([y_values,gpr.predict(x_values[i:i+256])])
        logq = np.array(y_values)
        mask = np.isfinite(logq)
        logp2 = logp[mask]
        logq2 = logq[mask]
        weights2 = weights[mask]
        kl = np.sum(weights2*(logp2-logq2))/np.sum(weights2)
        history['KL_full_true_wrt_gp'].append(kl)
        history['total_evals_kl'].append(gpr.n_total_evals)
        history['accepted_evals_kl'].append(gpr.n_accepted_evals)
        KL_gauss_true_wrt_gp_total[counter, n_r] = kl_norm(s1mean,s1cov, s2mean, s2cov)
        KL_gauss_gp_wrt_true_total[counter, n_r] = kl_norm(s2mean,s2cov, s1mean, s1cov)
        KL_full_true_wrt_gp_total[counter, n_r] = kl
        del s2mean, s2cov, s2, sampler2
        counter += 1
        if plot_intermediate_contours: # and convergence.ncorrect != n_conv:
            gdplot = gdplt.get_subplot_plotter(width_inch=5)
            gdplot.triangle_plot([gdsamples1, gdsamples2], list(info["params"]),
                                 filled=[False, True], legend_labels=['MCMC', 'GPry'])
            getdist_add_training(gdplot, model, gpr)
            if info_text_in_plot:
                n_d = model.prior.d()
                info_text = ("$n_{tot}=%i$\n $n_{accepted}=%i$\n $d_{KL}=%.2e$\n $d_{CC}=%.2e$"
                    %(gpr.n_total_evals, gpr.n_accepted_evals, kl, corr_counter_conv.values[-1]))
                ax = gdplot.get_axes(ax=(0, n_d-1))
                gdplot.add_text_left(info_text, x=0.2, y=0.5, ax=(0, n_d-1)) #, transform=ax.transAxes
                ax.axis('off')
            plt.savefig(f"{likelihood}/triangle_try_{n_r}_{gpr.n_accepted_evals}.pdf")
            plt.close()
            del gdplot, gdsamples2
        data ={'hist':history,'n_accepted':gpr.n_accepted_evals,'n_tot':gpr.n_total_evals}
        with open(f"{likelihood}/history_try_{n_r}.pkl", "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    counter = 0
    run(model, convergence_criterion="DontConverge", verbose=verbose,callback = print_and_plot, options={'max_accepted':n_accepted_evals, 'max_points':10000})

data_2 = {
    'KL_gauss_true_wrt_gp':KL_gauss_true_wrt_gp_total,
    'KL_gauss_gp_wrt_true':KL_gauss_gp_wrt_true_total,
    'KL_full_true_wrt_gp':KL_full_true_wrt_gp_total,
    'Correct_counter':Correct_counter_total,
    'accepted_evals':accepted_evals,
    'total_evals_for_convergence':total_evals_for_convergence_total,
    'accepted_evals_for_convergence':accepted_evals_for_convergence_total}
with open(f"{likelihood}/history_total.pkl", "wb") as f:
    pickle.dump(data_2, f, pickle.HIGHEST_PROTOCOL)

run_params = {
    'n_repeats': n_repeats,
    'min_points_for_kl': min_points_for_kl,
    'evaluate_convergence_every_n': evaluate_convergence_every_n,
    'n_accepted_evals': n_accepted_evals,
    'accepted_evals': accepted_evals
}
with open(f"{likelihood}/run_params.pkl", "wb") as f:
    pickle.dump(run_params, f, pickle.HIGHEST_PROTOCOL)
