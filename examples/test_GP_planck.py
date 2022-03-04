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
import time
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# GPry things needed for building the model
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

# Theory code

theory_code = "classy"
preset = "planck_2018_" + theory_code.lower()
info = create_input(preset=preset)
del info['theory']['classy']['extra_args']['non linear']
del info['theory']['classy']['extra_args']['hmcode_min_k_max']
info['params']['theta_s_1e2']['prior'] = {'min':1.0,'max':1.1}
model = get_model(info)
print("INFO --> ",info)
print(model.parameterization.sampled_params(),model.parameterization.sampled_params_info())
###############################
### RUN THE COMPARISON MCMC ###
###############################
print("PART 1 :: THE 'TRUTH'")
time1 = time.time()
info_run = info.copy()
info_run['sampler'] = {"mcmc": {"Rminus1_stop": rminusone, "max_tries": 10000}}
updated_info1, sampler1 = run(info_run,output="chains/Planck",resume=True)
time2 = time.time()
s1 = sampler1.products()["sample"]
if hasattr(s1,"detemper"):
  s1.detemper()
s1mean, s1cov = s1.mean(), s1.cov()
parnames = s1.sampled_params
x_values = s1.data[s1.sampled_params]
logp = -s1['minuslogpost']
weights = s1['weight']

##################
### RUN THE GP ###
##################
print("PART 2 :: THE 'GP'")
time3 = time.time()



def callback(model, gpr, gp_acquisition, convergence, options,
                  old_gpr, new_X, new_y, y_pred):
  print("HYPER CALLBACK ",gpr.kernel_.theta, (gpr.kernel_.theta-gpr.kernel_.bounds[:,0])/(gpr.kernel_.bounds[:,1]-gpr.kernel_.bounds[:,0]),"<<-HYPER CALLBACK")


#NOTE :: I changed SVM sigma to 14 at some point, but I don't think it matters
cname = "check3_fewerfits2"
if np.all(_check_checkpoint(cname)):
  _, gpr, _, _, _ = _read_checkpoint(cname)
else:
  _, gpr, _, _, _ = gpry.run.run(model, convergence_criterion="CorrectCounter", verbose=verbose,options={'max_accepted':2000, 'max_points':10000},callback=callback,checkpoint=cname)


#####################
### POSSIBLY PLOT ###
#####################
doplot = False
if is_main_process and doplot:
  smask = np.argsort(gpr.y_train)
  x = gpr.X_train[smask][::-1]
  y = gpr.y_train[smask][::-1]
  x0 = x[0]
  print(x0,y[0])
  #print(s1cov)
  stds = np.sqrt(np.diag(s1cov))
  num_err = 15
  vari = np.linspace(-2,2,num=num_err)
  plt.figure()
  for d in range(gpr.d):
    errs = np.empty(num_err)
    for ifac,fac in enumerate(vari):
      pt = np.array([x0.copy()])
      pt[0][d]+=fac*stds[d]
      logp = model.logpost(pt[0])
      print("d,fac,gp,log,diff = ",d,fac,gpr.predict(pt)[0],logp,gpr.predict(pt)[0]/logp-1.)
      err = (gpr.predict(pt)[0]/logp-1. if np.isfinite(logp) else 0.)
      errs[ifac] = err
    plt.plot(vari, errs, color=plt.get_cmap("gist_rainbow")(float(d)/gpr.d),label=parnames[d])
  plt.legend(bbox_to_anchor=(1.4,1.4))
  plt.savefig("diff_plot.pdf",bbox_inches="tight")
if doplot:
  quit()


#####################
### RUN GP MCMC   ###
#####################
time4 = time.time()
updated_info2, sampler2 = mc_sample_from_gp(model, gpr, sampler="mcmc", convergence=None, output="chains/GP5", add_options={'mcmc':{'max_tries':1000000,"Rminus1_stop": rminusone,
'learn_proposal_Rminus1_max':rminusone,
'covmat':s1cov,'covmat_params':parnames
}}, restart=False)
#{'learn_proposal_Rminus1_max':1e25,'learn_proposal_Rminus1_max_early':1e25,'learn_proposal_Rminus1_min':1e-25,
time5 = time.time()



######################
### DERIVE SUMMARY ###
######################
if is_main_process:
  s2 = sampler2.products()["sample"]
  if hasattr(s2,"detemper"):
    s2.detemper()
  s2mean, s2cov = s2.mean(), s2.cov()
  hist = {'KL_gauss_wrt_true' : kl_norm(s1mean,s1cov, s2mean, s2cov),'KL_gauss_wrt_gp':kl_norm(s2mean,s2cov, s1mean, s1cov)}
  y_values = []
  print("BEFORE PREDICT")
  for i in range(0,len(x_values), 256):
    y_values = np.concatenate([y_values,gpr.predict(x_values[i:i+256])])
  logq = np.array(y_values)
  print(logq.shape)
  print(logp.shape)
  print("AFTER PREDICT")
  hist['KL_truth'] = np.sum(weights*(logp-logq))/np.sum(weights)
  hist['step'] = gpr.n_accepted_evals
  hist['n_tot'] = gpr.n_total_evals
  with open("planck.pkl", "wb") as f:
      pickle.dump(hist, f, pickle.HIGHEST_PROTOCOL)
  np.savetxt("planck_timing.dat",[time2-time1,time4-time3,time5-time4])

  gdsamples1 = MCSamplesFromCobaya(updated_info1, s1)
  gdsamples2 = MCSamplesFromCobaya(updated_info2, s2)
  del s1,s2, sampler1, sampler2, logq, logp, weights, hist
  gdplot = gdplt.get_subplot_plotter(width_inch=5)
  gdplot.triangle_plot([gdsamples1,gdsamples2], params = parnames[:10], filled=[True,False], legend_labels = ['TRUE', 'GP'])
  plt.savefig("planck.pdf")
  #plt.show()
  plt.close()
quit()
