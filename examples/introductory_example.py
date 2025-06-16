"""
Introductory example to using GPry
"""

## Step 1: Setting up a likelihood function

import os
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

## Step 4: Monte Carlo samples from the surrogate model

# Retrieve samples generated for convergence check:

mc_samples_dict = runner.last_mc_samples(as_pandas=True)
print(mc_samples_dict)

# To generate new ones:

runner.generate_mc_sample(
    # Example args for denser samples
    # sampler={"nested": {"nlive": "25d", "num_repeats": "10d"}}
)

# To plot the last MC samples:

runner.plot_mc()

### Bonus: Getting some extra insights

runner.plot_progress()

## Bonus Bonus: Validation

truth_samples = rv.rvs(size=10000)
runner.set_fiducial_MC(truth_samples)
runner.plot_mc(output=os.path.join(runner.plots_path, "comparison_triangle.png"))
