"""
Example code for LCDM with the Planck lite likelihood (1 nuisance parameter)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
rcparams = {
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'font.size': 14, # was 10
    'legend.fontsize': 14, # was 10
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
}
matplotlib.rcParams.update(rcparams)

import matplotlib.pyplot as plt
# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from gpry.mpi import is_main_process, mpi_comm
from gpry.plots import getdist_add_training
import gpry.run
# Building the likelihood
# (does programmatically the same as calling cobaya-cosmo-generator)
from cobaya.cosmo_input import create_input
from cobaya.model import get_model
from cobaya.yaml import yaml_load_file
from gpry.run import mc_sample_from_gp
from gpry.mpi import mpi_rank, is_main_process, mpi_comm
from getdist.mcsamples import MCSamplesFromCobaya
from gpry.tools import kl_norm
import getdist.plots as gdplt
import pickle


# Choose classy (CLASS) or camb (CAMB)
theory_code = "classy"
#theory_code = "camb"



# Give option to turn on/off different parts of the code
n_repeats = 5 # number of repeats for checking convergence
rminusone = 0.01 # R-1 for the MCMC
min_points_for_kl = 40 # minimum number of points for calculating the KL divergence
evaluate_convergence_every_n = 20 # How many points should be sampled before running the KL
n_accepted_evals = 251 # Number of accepted steps before killing
plot_intermediate_contours = True # whether to plot intermediate (non-converged)
plot_final_contours = True # whether to plot final contours
info_text_in_plot = True

verbose = 1 # Verbosity of the BO loop

accepted_evals = np.arange(min_points_for_kl, n_accepted_evals, evaluate_convergence_every_n)






preset = "planck_2018_" + theory_code.lower()
info = create_input(preset=preset)
# Substitute high-ell likelihood by Planck-lite
info["likelihood"].pop("planck_2018_highl_plik.TTTEEE")
info["likelihood"]["planck_2018_highl_plik.TTTEEE_lite_native"] = None

# Temporary solution: reduce priors: (CAMB case only)
# See https://wiki.cosmos.esa.int/planck-legacy-archive/images/2/21/Baseline_params_table_2018_95pc_v2.pdf
info["params"]["A_planck"] = 1.00044  # fixed for now
del info['theory']['classy']['extra_args']['non linear']
del info['theory']['classy']['extra_args']['hmcode_min_k_max']
print(info['params'].keys())
info['params']['theta_s_1e2']['prior'] = {'min':1.04,'max':1.044}
info['params']['n_s']['prior'] = {'min':0.94,'max':0.99}
info['params']['omega_b']['prior'] = {'min':0.0215,'max':0.023}
info['params']['omega_cdm']['prior'] = {'min':0.115,'max':0.125}
info['params']['logA']['prior'] = {'min':2.9,'max':3.1}
info['params']['tau_reio']['prior'] = {'min':0.03,'max':0.08}
info['params']['theta_s_1e2']['renames']=['theta']

model = get_model(info)
#print(model)
dim = model.prior.d()
prior_bounds = model.prior.bounds()



###############################
### RUN THE COMPARISON MCMC ###
###############################
ch_name = "chains/Planck_LITE"
if is_main_process:
  print("PART 1 :: THE 'TRUTH'")
  try:
    import yaml
    loaded_info = yaml_load_file(ch_name+".checkpoint")
    if np.all([loaded_info['sampler'][key]['converged'] for key in loaded_info['sampler']]):
      found_converged = True
    else:
      found_converged = False
  except FileNotFoundError as fnfe:
    print("File not found :: ",fnfe)
    found_converged = False
  except Exception as e:
    raise Exception("There was a problem reading something, here's the origianl error message :: ",e) from e
found_converged = mpi_comm.bcast(found_converged if is_main_process else None)
if found_converged:
  from getdist.mcsamples import loadMCSamples
  true_samples = loadMCSamples(ch_name)
else:
  from getdist.mcsamples import MCSamplesFromCobaya
  info_run = info.copy()
  info_run['sampler'] = {"mcmc": {"Rminus1_stop": rminusone, "max_tries": 10000}}
  from cobaya.run import run as run_cobaya
  updated_info1, sampler1 = run_cobaya(info_run,output=ch_name,resume=True)
  s1 = sampler1.products()["sample"]
  if hasattr(s1,"detemper"):
    s1.detemper()
  true_samples = MCSamplesFromCobaya(updated_info1,s1)
s1cov = true_samples.getCov()
s1mean = true_samples.getMeans()
parnames = [x.name for x in true_samples.getParamNames().names]
Npar = 6#len(parnames)
parnames = parnames[:Npar]
s1cov = s1cov[:Npar,:Npar]
s1mean = s1mean[:Npar]
x_values = true_samples.samples[:,:Npar]
print(parnames)
#logp = -0.5*true_samples['chi2']
#print(-true_samples['loglike'])
logp = -true_samples['loglike']
weights = true_samples.weights



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
    from gpry.convergence import CorrectCounter,DontConverge
    corr_counter_conv = CorrectCounter(model.prior, {})
    def print_and_plot(model, gpr, gp_acquisition, convergence, options, progress,
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
        print("INSIDE CALLBACK", mpi_rank, is_converged, convergence)
        if not is_converged and convergence:
            print("\n\n !!!! CorrectCounter has converged! !!!! \n\n")
            history["total_evals_for_convergence"] = gpr.n_total_evals
            history["accepted_evals_for_convergence"] = gpr.n_accepted_evals
            total_evals_for_convergence_total[n_r] = gpr.n_total_evals
            accepted_evals_for_convergence_total[n_r] = gpr.n_accepted_evals
            is_converged = True
            if plot_final_contours:
                #updated_info2, sampler2 = mcmc(model, gpr, options = {"mcmc": {"Rminus1_stop": rminusone, "max_tries": 10000}})
                updated_info2, sampler2 =  mc_sample_from_gp(gpr, model.prior.bounds(confidence_for_unbounded=0.99995), model.prior.params, sampler="polychord")
                # print("BEFORE GETSAMPLES")
                if is_main_process:
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
                  kl = np.sum(weights2*(logp2-logq2))/np.sum(weights2)
                  del s2, sampler2
                  gdplot = gdplt.get_subplot_plotter(width_inch=5)
                  gdplot.triangle_plot([true_samples, gdsamples2], parnames,
                                       filled=[False, True], legend_labels=['MCMC', 'GPry'])
                  print("BEFORE ADDING TRAINING")
                  getdist_add_training(gdplot, parnames, gpr)
                  print("AFTER ADDING TRAINING")
                  if info_text_in_plot:
                      n_d = model.prior.d()
                      info_text = ("$n_{tot}=%i$\n $n_{accepted}=%i$\n $d_{KL}=%.2e$\n $d_{CC}=%.2e$"
                          %(gpr.n_total_evals, gpr.n_accepted_evals, kl, corr_counter_conv.values[-1]))
                      ax = gdplot.get_axes(ax=(0, n_d-1))
                      gdplot.add_text_left(info_text, x=0.2, y=0.5, ax=(0, n_d-1)) #, transform=ax.transAxes
                      ax.axis('off')
                  print("FINISHING \n")
                  plt.tight_layout()
                  plt.savefig("images/triangle_try_{}_final.pdf".format(n_r))
                  plt.close()
                  print("PLOT CLOSED")

        #accepted_evals = mpi_comm.bcast(accepted_evals if is_main_process else None)
        counter = mpi_comm.bcast(counter if is_main_process else None)
        if len(accepted_evals) > counter:
            if gpr.n_accepted_evals < accepted_evals[counter]: # and not (n_conv >= convergence.ncorrect):
                return
        else:
            return
        # Add correct counter value
        Correct_counter_total[counter, n_r] = corr_counter_conv.values[-1]
        # Calculate KL divergence
        print("RUNNING MCMC!",mpi_rank)
        updated_info2, sampler2 =  mc_sample_from_gp(gpr, model.prior.bounds(confidence_for_unbounded=0.99995), model.prior.params, sampler="polychord")
        #updated_info2, sampler2 = mcmc(model, gpr, options = {"mcmc": {"Rminus1_stop": rminusone, "max_tries": 10000}})
        #mcmc(model, gpr, options = {"mcmc": {"Rminus1_stop": rminusone, "max_tries": 10000}})
        # print("BEFORE GETSAMPLES")

        print("GET SAMPLES")
        if is_main_process:
          s2 = sampler2.products()["sample"]
          gdsamples2 = MCSamplesFromCobaya(updated_info2, s2)
          s2.detemper()
          s2mean, s2cov = s2.mean(), s2.cov()
          s2cov = s2cov[:Npar,:Npar]
          s2mean = s2mean[:Npar]
          print(s1mean,s2mean, s1cov.shape, s2cov.shape)
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
          print("MASK MASK MASK")
          print(logp2, logq2, logp2-logq2)
          print(np.mean(logp2-logq2))
          kl = np.sum(weights2*(logp2-logq2))/np.sum(weights2)
          for itcounter in range(5):
            print("--",itcounter,"--")
            print(x_values[itcounter])
            print(y_values[itcounter])
            print(model.logposterior(x_values[itcounter]))
            print(logp[itcounter])
            print(logq[itcounter])
          print("MASK MASK MASK")
          history['KL_full_true_wrt_gp'].append(kl)
          history['total_evals_kl'].append(gpr.n_total_evals)
          history['accepted_evals_kl'].append(gpr.n_accepted_evals)
          KL_gauss_true_wrt_gp_total[counter, n_r] = kl_norm(s1mean,s1cov, s2mean, s2cov)
          KL_gauss_gp_wrt_true_total[counter, n_r] = kl_norm(s2mean,s2cov, s1mean, s1cov)
          KL_full_true_wrt_gp_total[counter, n_r] = kl
          del s2mean, s2cov, s2, sampler2
          counter += 1
          print("DONE COUNTER")
          if plot_intermediate_contours: # and convergence.ncorrect != n_conv:
              gdplot = gdplt.get_subplot_plotter(width_inch=5)
              gdplot.triangle_plot([true_samples, gdsamples2], parnames,
                                   filled=[False, True], legend_labels=['MCMC', 'GPry'])
              print("BEFORE ADD INTERMEDIATE")
              getdist_add_training(gdplot, parnames, gpr)
              print("AFTER ADD INTERMEDIATE")
              if info_text_in_plot:
                  n_d = model.prior.d()
                  info_text = ("$n_{tot}=%i$\n $n_{accepted}=%i$\n $d_{KL}=%.2e$\n $d_{CC}=%.2e$"
                      %(gpr.n_total_evals, gpr.n_accepted_evals, kl, corr_counter_conv.values[-1]))
                  ax = gdplot.get_axes(ax=(0, n_d-1))
                  gdplot.add_text_left(info_text, x=0.2, y=0.5, ax=(0, n_d-1)) #, transform=ax.transAxes
                  ax.axis('off')
              plt.tight_layout()
              plt.savefig("images/triangle_try_{}_{}.pdf".format(n_r,gpr.n_accepted_evals))
              plt.close()
              print("AFTER CLOSE INTERMEDIATE")
              del gdplot, gdsamples2
          data ={'hist':history,'n_accepted':gpr.n_accepted_evals,'n_tot':gpr.n_total_evals}
          with open("images/history_try_{}.pkl".format(n_r), "wb") as f:
              pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
          print("EXITING CALLBACK",mpi_rank)

    print("COUNTER RESET",mpi_rank,n_r)
    counter = 0
    gpry.run.run(model, convergence_criterion="DontConverge", verbose=verbose,callback = print_and_plot, options={'max_accepted':n_accepted_evals, 'max_points':10000}, callback_is_MPI_aware=True)
    print("RUN DONE",mpi_rank,n_r)
print("\n\nALL DONE\n\n")
if is_main_process:
  data_2 = {
      'KL_gauss_true_wrt_gp':KL_gauss_true_wrt_gp_total,
      'KL_gauss_gp_wrt_true':KL_gauss_gp_wrt_true_total,
      'KL_full_true_wrt_gp':KL_full_true_wrt_gp_total,
      'Correct_counter':Correct_counter_total,
      'accepted_evals':accepted_evals,
      'total_evals_for_convergence':total_evals_for_convergence_total,
      'accepted_evals_for_convergence':accepted_evals_for_convergence_total}
  with open("images/history_total.pkl", "wb") as f:
      pickle.dump(data_2, f, pickle.HIGHEST_PROTOCOL)

  run_params = {
      'n_repeats': n_repeats,
      'min_points_for_kl': min_points_for_kl,
      'evaluate_convergence_every_n': evaluate_convergence_every_n,
      'n_accepted_evals': n_accepted_evals,
      'accepted_evals': accepted_evals
  }
  with open("images/run_params.pkl", "wb") as f:
      pickle.dump(run_params, f, pickle.HIGHEST_PROTOCOL)

