"""
Example code for a simple GP Characterization of a likelihood.
"""

import numpy as np
from scipy.stats import multivariate_normal

mean = [3, 2]
cov = [[0.5, 0.4], [0.4, 1.5]]
rv = multivariate_normal(mean, cov)

# The names of the parameters will be read off the args of the function
def lkl(x, y):
    return rv.logpdf(np.array([x, y]).T)


bounds = [[-10, 10], [-10, 10]]

#############################################################

# Where we will save the results:
checkpoint = "output/simple"

from gpry.run import Runner

runner = Runner(lkl, bounds, checkpoint=checkpoint, load_checkpoint="overwrite")

#############################################################

# Plotting the likelihood
import os
import gpry.mpi as mpi
import matplotlib.pyplot as plt
if mpi.is_main_process:
    from matplotlib.colors import LogNorm
    Nmesh = 200
    x = np.linspace(bounds[0][0], bounds[0][1], Nmesh)
    y = np.linspace(bounds[1][0], bounds[1][1], Nmesh)
    X, Y = np.meshgrid(x, y)
    xy = np.stack((X, Y), axis=-1).reshape(-1, 2)
    Z = -lkl(xy[:, 0], xy[:, 1]).reshape((Nmesh, Nmesh))
    # Plot ground truth
    fig = plt.figure()
    im = plt.pcolor(X, Y, Z, norm=LogNorm())
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
    cbar = fig.colorbar(
        im, cax=cbar_ax, orientation='vertical',
        label=r"$-\log\mathcal{L}$"
    )
    plt.savefig(os.path.join(checkpoint, "simple_ground_truth.png"))
    plt.show(block=False)
    plt.close()

#############################################################

# Run the main GPry loop
runner.run()

# Recover the original function (only for uniform priors, otherwise the posterior)

point = (1, 2)
print(lkl(*point))
print(runner.logp(point) + runner.log_prior_volume)

#############################################################

# Run the MCMC and extract samples
runner.generate_mc_sample()
samples_gp = runner.last_mc_samples()

#############################################################

# Plotting the result (automatically saved)
runner.plot_mc()
import matplotlib.pyplot as plt
plt.show(block=False)

# Other plots
runner.plot_progress()
runner.plot_distance_distribution()

#############################################################

# Validating against the ground truth
# The parameter names need to be the same as for the log-likelihood function
names = "x", "y"
from getdist import MCSamples
samples_truth = MCSamples(samples=rv.rvs(size=10000), names=names)

import os
runner.plot_mc(
    add_samples={"Ground truth": samples_truth},
    output=os.path.join(checkpoint, "images/Comparison_triangle.png"),
)
plt.show(block=True)
