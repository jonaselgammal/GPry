# Introductory example to using GPry
# ----------------------------------
#
# This file will guide you through the basic steps for sampling a posterior with GPry

# ### Step 1: Setting up a likelihood function
#
# Let's set up a simple 2d Gaussian likelihood as an example:
#
# $$y(x) \sim \mathcal{N}(x|\mu,\Sigma)
# \qquad
# \text{with}
# \qquad
# \mu=\begin{pmatrix}3 \\ 2\end{pmatrix},\ \Sigma=\begin{pmatrix}0.5 & 0.4 \\ 0.4 & 1.5\end{pmatrix}$$
#
# We want to sample from the posterior within a uniform prior square $[-10, 10]$.
#
# We need to define a **log-likelihood** function, which is the modelling target for GPry
# and the prior bounds:

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

mean = [3, 2]
cov = [[0.5, 0.4], [0.4, 1.5]]
rv = multivariate_normal(mean, cov)

def logLkl(x_1, x_2):
    return rv.logpdf(np.array([x_1, x_2]).T)

bounds = [[-10, 10], [-10, 10]]

# ### Step 2: Creating the Runner object
#
# The `run.Runner` object manages model specification and the active sampling loop of GPry
# up to convergence. A didactic intro to this process can be found in section
# "How does GPry work" (https://gpry.readthedocs.io/en/latest/how_does_gpry_work.html).
# The `Runner` object also implements some post-processing and tests.
#
# To initialize it, we pass it the log-likelihood function as first argument, and the
# prior bounds as the second argument (or via the `bounds` keyword). More complicate
# prior specification can be passed by defining and passing as first argument a Cobaya
# model (https://cobaya.readthedocs.io/en/latest/models.html) (see
# https://gpry.readthedocs.io/en/latest/module_cobaya.html).
#
# Optionally, we can also pass a path where to save checkpoints via the `checkpoint`
# argument. If passed, in order to prevent loss of data, you **must** specify a checkpoint
# policy, either `load_checkpoint="resume"` or `load_checkpoint="overwrite"`). If set to
# `"resume"` the runner object will try to load the checkpoint and resume the active
# sampling loop from there; if set to `"overwrite"` it will start from scratch and delete
# checkpoint files that may already exist.
#
# In this example we will leave to their default values all training parameters: the
# choice of GP, acquisition function, convergence criterion, options of the active
# sampling loop...

from gpry import Runner
checkpoint = "output/intro"
runner = Runner(logLkl, bounds, checkpoint=checkpoint, load_checkpoint="overwrite")

# ### Step 3: Running the active learning loop
#
# Since all training parameters are chosen automatically, all we have to do is to call the `run` method of the `Runner` object:

runner.run()

# This runs the active sampling loop until convergence is reached. It also saves the
# checkpoint files after every iteration and creates progress plots which are saved in
# ``[checkpoint]/images/`` (or ``./images/`` if a checkpoint was not defined).
#
# Once converged, you can access the surrogate model and use it as a function for any
# purpose. (NB: Internally GPry models the **log-posterior**, not the log-likelihood.)
#
# To get the surrogate log-posterior or log-likelihood you can call respectively
# `Runner.logp` or `Runner.logL`, passing each a single `(nsamples, ndims)` array with the
# locations where you want to evaluate the surrogate.
#
# Let us compare the GPry surrogate model and the true likelihood at `(1, 2)`. Both
# evaluations should produce similar numbers.

point = (1, 2)
print(f"True log-likelihood at {point}:      {logLkl(*point)}")
print(f"Surrogate log-likelihood at {point}: {runner.logL(point)[0]}")

# ### Step 4: Monte Carlo samples from the surrogate posterior
#
# As part of a final test before convergence, GPry will have run a Monte Carlo sampler on
# the surrogate model. If everything went well, you can use that sample as you would with
# one obtained with a traditional MC sampler: to extract marginalized quantities, create a
# corner plot of it, etc.
#
# If a checkpoint has been defined, samples are stored in that same folder, inside a
# `chains` sub folder, in one or mode `.txt` files. In those files, the order of the
# columns is `weight log-posterior param_1 param_2 ...`.
#
# If you would like to repeat this process to get a finer sample, you can generate a new
# one by calling the `Runner.generate_mc_sample` method with some options for the sampler:

runner.generate_mc_sample(sampler={"nested": {"nlive": "50d"}})

# The last MC sample can be retrieved with the `Runner.last_mc_samples` method:

print(runner.last_mc_samples(as_pandas=True))

# Now that we have MC samples, we can process and plot them the same way that we would do
# with any other MC samples.
#
# The easiest way to get a corner plot is to call the :meth:`run.Runner.plot_mc` method,
# which will generate a `GetDist` corner plot (it includes the training set unless passed
# ``add_training=False``).

p = runner.plot_mc(ext="svg")
plt.show(block=False)

# ### Bonus: Getting some extra insights
#
# You can do further plots about the progress of the active-learning loop calling the
# `plot_progress()` method.
#
# If you call this method without any arguments it produces the following plots:
#
# * a histogram of the time spent at different parts of the code.
# * if the run has converged, a corner plot of the final MC sample showing the training
# set (the same one you get when calling :meth:`run.Runner.plot_mc`).
# * a convergence history ("trace") plot showing, as a function of posterior evaluations,
# the value(s) of all convergence criteria, the distribution of posterior values, and the
# distribution of samples per model parameter.

a = runner.plot_progress()
plt.show(block=False)

# ### Bonus Bonus: Validation
#
# **NB: This part is optional and only relevant for validating the contours that GPry
# produces. In a realistic scenario you would obviously not run a full MCMC on the
# likelihood  and would need to follow the validation guidelines at "Strategy and
# troubleshooting" (https://gpry.readthedocs.io/en/latest/strategy.html).**
#
# To compare our contours to the true Gaussian we draw 10000 samples from it, and set them
# as *fiducial samples* in the ``Runner``:

truth_samples = rv.rvs(size=10000)
runner.set_fiducial_mc(truth_samples)

runner.plot_mc()
plt.show(block=True)

# As you can see the two agree almost perfectly! And we achieved this with just a few
# evaluations of the posterior distribution!
