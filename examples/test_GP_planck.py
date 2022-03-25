"""
This script runs the GP on the Planck likelihoods.

In particular, tests the accuracy of the covariance recovered (since for well-recovered
covariance matrix, all criteria would be equivalent).
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, random_correlation
from numpy.random import default_rng
import warnings
import time
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# GPry things needed for building the model
from cobaya.yaml import yaml_load_file
from gpry.mpi import is_main_process, mpi_comm
from cobaya.model import get_model
from gpry.plots import getdist_add_training
from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt
from cobaya.run import run
from gpry.run import mcmc, mc_sample_from_gp
import gpry.run
from gpry.tools import kl_norm
from cobaya.cosmo_input import create_input
from gpry.run import _check_checkpoint, _read_checkpoint

# Number of times that each combination of d and zeta is run with different
# gaussians
rminusone = 0.05

# Ratio of prior size to (2x)std of mode
prior_size_in_std = 3

# Print always full dataframes
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

verbose = 5
ch_name = "chains/Planck"





iplot = 6
cname = "check8"
ch_name2 = "chains/GP12"

doplot = False

stype = "polychord"






# Theory code

theory_code = "classy"
preset = "planck_2018_" + theory_code.lower()
info = create_input(preset=preset)
del info['theory']['classy']['extra_args']['non linear']
del info['theory']['classy']['extra_args']['hmcode_min_k_max']
info['params']['theta_s_1e2']['prior'] = {'min':1.038,'max':1.044}
info['params']['n_s']['prior'] = {'min':0.94,'max':0.99}
info['params']['omega_b']['prior'] = {'min':0.021,'max':0.024}
info['params']['omega_cdm']['prior'] = {'min':0.115,'max':0.125}
info['params']['logA']['prior'] = {'min':2.9,'max':3.1}
model = get_model(info)
Npar = len(model.parameterization.sampled_params())

#pt = np.array([float(x) for x in ['3.01373202138339379985e+00', '9.25886299142247737315e-01', '1.04860417640966319119e+00', '2.27334913258291464178e-02', '1.80664895063774588735e-01', '3.71644360456949082727e-02', '1.00096770980313976018e+00', '9.99765212710281847563e-01', '9.97546522304238347800e-01', '8.17320890932995212097e+01', '2.61870306499058347338e-01', '7.51136719911507899639e+00', '1.23553597552720972885e+00', '6.80230824418587953772e+00', '1.33611078875484334816e+01', '1.95353823168258777798e+01', '7.28284720205743383303e+01', '2.22423453280500410756e+02', '3.41941471911878096535e+01', '5.90292261961908195644e+01', '4.48229810488915063615e+01', '5.03008718540670640706e-03', '1.01143921165742188550e-02', '4.34626939439516857266e-01', '1.76127366168236515476e-01', '5.72209243956257829211e-01', '1.97137829247690077317e+00']])
#print("!!!",model.logpost(pt))
#print("INFO --> ",info)
#print(model.parameterization.sampled_params(),model.parameterization.sampled_params_info())
###############################
### RUN THE COMPARISON MCMC ###
###############################

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
  updated_info1, sampler1 = run(info_run,output=ch_name,resume=True)
  s1 = sampler1.products()["sample"]
  if hasattr(s1,"detemper"):
    s1.detemper()
  true_samples = MCSamplesFromCobaya(updated_info1,s1)
s1cov = true_samples.getCov()
s1mean = true_samples.getMeans()
parnames = [x.name for x in true_samples.getParamNames().names]
parnames = parnames[:Npar]
s1cov = s1cov[:Npar,:Npar]
s1mean = s1mean[:Npar]
x_values = true_samples.samples
logp = -0.5*true_samples['chi2']-true_samples['minuslogprior']
weights = true_samples.weights

##################
### RUN THE GP ###
##################
if is_main_process:
  print("PART 2 :: THE 'GP'")

def callback(model, gpr, gp_acquisition, convergence, options,
                  old_gpr, new_X, new_y, y_pred):
  print("HYPER CALLBACK ",gpr.kernel_.theta, (gpr.kernel_.theta-gpr.kernel_.bounds[:,0])/(gpr.kernel_.bounds[:,1]-gpr.kernel_.bounds[:,0]),"<<-HYPER CALLBACK")

#NOTE :: I changed SVM sigma to 14 at some point, but I don't think it matters
if np.all(_check_checkpoint(cname)):
  _, gpr, _, _, _ = _read_checkpoint(cname)
else:
  _, gpr, _, _, _ = gpry.run.run(model, convergence_criterion="CorrectCounter", verbose=verbose,options={'max_accepted':2000, 'max_points':10000},callback=callback,checkpoint=cname)



#####################
### POSSIBLY PLOT ###
#####################
if is_main_process and doplot:
  smask = np.argsort(gpr.y_train)
  x = gpr.X_train[smask][::-1]
  y = gpr.y_train[smask][::-1]
  x0 = x[0]
  print(x0,y[0])
  #print(s1cov)
  stds = np.sqrt(np.diag(s1cov))
  num_err = 15
  sig_devs = 10
  vari = np.linspace(-sig_devs,sig_devs,num=num_err)
  plt.figure()
  minpred = -np.inf
  for d in range(gpr.d):
    errs = np.empty(num_err)
    for ifac,fac in enumerate(vari):
      pt = np.array([x0.copy()])
      pt[0][d]+=fac*stds[d]
      logp = model.logpost(pt[0])
      pred = gpr.predict(pt)[0]
      print("d,fac,gp,log,diff = ",d,fac,pred,logp,pred/logp-1.)
      err = (pred/logp-1. if np.isfinite(logp) else 0.)
      if -pred<-minpred:
        minpred = pred
        position = (d,ifac,fac)
      errs[ifac] = err
    plt.plot(vari, errs, color=plt.get_cmap("gist_rainbow")(float(d)/gpr.d),label=parnames[d])
  plt.legend(bbox_to_anchor=(1.4,1.4))
  print("MIN LKL = {} AT {}".format(minpred,position))
  plt.savefig("diff_plot{}.pdf".format(iplot),bbox_inches="tight")
if doplot:
  quit()


#####################
### RUN GP MCMC   ###
#####################

if is_main_process:
  print("PART 1 :: THE 'GP MCMC'")
  try:
    fo = open(ch_name2+".1.txt")
    found_converged = True
    fo.close()
  except FileNotFoundError as fnfe:
    print("File not found :: ",fnfe)
    found_converged = False
  except Exception as e:
    raise Exception("There was a problem reading something, here's the origianl error message :: ",e) from e
found_converged = mpi_comm.bcast(found_converged if is_main_process else None)
if found_converged and is_main_process:
  from getdist.mcsamples import loadMCSamples
  gp_samples = loadMCSamples(ch_name2)
else:
  from getdist.mcsamples import MCSamplesFromCobaya
  if stype == "mcmc":
    #def cback(sampler):
    #  newest_samples = sampler.collection[sampler.last_point_callback:]
    #  print(np.min(newest_samples.loglikes))
    updated_info2, sampler2 = mc_sample_from_gp(model, gpr, sampler="mcmc", convergence=None, output=ch_name2, 
    add_options={'mcmc':{'max_tries':1000000,"Rminus1_stop": rminusone,'covmat':s1cov,'covmat_params':parnames}})
  elif stype == "polychord":
    updated_info2, sampler2 = mc_sample_from_gp(model, gpr, sampler="polychord", convergence=None, output=ch_name2)
  if is_main_process:
    #print(sampler2.products(),sampler2.products().keys())
    s2 = sampler2.products()["sample"]
    if hasattr(s2,"detemper"):
      s2.detemper()
    gp_samples = MCSamplesFromCobaya(updated_info2,s2)
######################
### DERIVE SUMMARY ###
######################
if is_main_process:
  print("THE POST ANALYSIS!!")
  #s2cov = gp_samples.getCov()
  #s2mean = gp_samples.getMeans()
  #print(s2mean, s1mean)
  gdplot = gdplt.get_subplot_plotter(width_inch=5)
  gdplot.triangle_plot([true_samples,gp_samples], params = parnames[0:5], filled=[True,False], legend_labels = ['TRUE', 'GP'],contour_colors=["red","blue"])
  plt.savefig("planck{}.pdf".format(iplot))
  plt.close()
  s2cov = gp_samples.getCov()
  s2mean = gp_samples.getMeans()
  parnames = [x.name for x in gp_samples.getParamNames().names]
  parnames = parnames[:Npar]
  s2cov = s2cov[:Npar,:Npar]
  s2mean = s2mean[:Npar]
  hist = {'KL_gauss_wrt_true' : kl_norm(s1mean,s1cov, s2mean, s2cov),'KL_gauss_wrt_gp':kl_norm(s2mean,s2cov, s1mean, s1cov)}
  y_values = []
  print("BEFORE PREDICT")
  for i in range(0,len(x_values), 256):
    y_values = np.concatenate([y_values,gpr.predict(x_values[i:i+256,:27])])
  logq = np.array(y_values)
  print(logq.shape)
  print(logp.shape)
  print("AFTER PREDICT")
  hist['KL_truth'] = np.sum(weights*(logp-logq))/np.sum(weights)
  hist['step'] = gpr.n_accepted_evals
  hist['n_tot'] = gpr.n_total_evals
  with open("planck{}.pkl".format(iplot), "wb") as f:
      pickle.dump(hist, f, pickle.HIGHEST_PROTOCOL)
  #plt.show()
  quit()
quit()
