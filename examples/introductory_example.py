"""
Introductory example to using GPry
"""

## Step 1: Setting up a likelihood function

import numpy as np
from scipy.stats import multivariate_normal

mean = [3, 2]
cov = [[0.5, 0.4], [0.4, 1.5]]
rv = multivariate_normal(mean, cov)

def logLkl(x_1, x_2):
    return rv.logpdf(np.array([x_1, x_2]).T)

bounds = [[-10, 10], [-10, 10]]

## Step 2: Creating the Runner object

from gpry.run import Runner
checkpoint = "output/simple"
runner = Runner(logLkl, bounds, checkpoint=checkpoint, load_checkpoint="overwrite")

## Step 3: Running the active learning loop
runner.run()

## Step 4: Running a Monte Carlo sampler on the surrogate model
runner.generate_mc_sample()

## Bonus: Plotting the results with GetDist
runner.plot_mc()

### Bonus: Getting some extra insights
runner.plot_progress()

## Bonus Bonus: Validation
from getdist import MCSamples
samples_truth = MCSamples(samples=rv.rvs(size=10000), names=["x_1", "x_2"])

runner.plot_mc(add_samples={"Fiducial": samples_truth})