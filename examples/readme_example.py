import numpy as np
import scipy.stats as st

means = [[0, 2], [-1.0, -0.5], [1.5, 0.5]]

covs = [
    [[0.06, -0.007], [-0.007, 0.2]],
    [[0.2, -0.04], [-0.04, 0.15]],
    [[0.5, -0.08], [-0.08, 0.3]],
]

norms = [st.multivariate_normal(m, c) for m, c in zip(means, covs)]


def log_likelihood(x, y):
    return np.log(sum(norm.pdf([x, y]) for norm in norms) / len(means))


bounds = [[-5, 5], [-5, 5]]

from gpry import Runner

runner = Runner(
    log_likelihood,
    bounds,
    checkpoint="output/",
    plots={"corner": True},
    load_checkpoint="overwrite",
    # make default!!!
    gp_acquisition={"NORA": {"mc_every": 1}},
    #    options={"n_points_per_acq": 3},
)


nsamples_each = 10000
samples = np.concatenate([norm.rvs(nsamples_each) for norm in norms], axis=0)
runner.set_fiducial_MC(samples)


runner.run()

# To create git (requires imagemagick)
# convert -delay 35 corner_it_0*.png -delay 200 Surrogate_triangle.png -loop 0 animation.gif
