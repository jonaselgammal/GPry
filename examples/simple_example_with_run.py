from scipy.stats import multivariate_normal, random_correlation
from gpry.convergence import KL_from_draw
n_d = 3
input_params = [f"x_{d}" for d in range(n_d)]

std = np.random.uniform(size=n_d)
eigs = np.random.uniform(size=n_d)
eigs = eigs / np.sum(eigs) * n_d
corr = random_correlation.rvs(eigs)
cov = np.multiply(np.outer(std, std), corr)
mean = np.zeros_like(std)

rv = multivariate_normal(mean, cov)

def f(**kwargs):
    X = [kwargs[p] for p in input_params]
    return np.log(rv.pdf(X))

# Define the likelihood and the prior of the model
info = {"likelihood": {"f": {"external": f,
                             "input_params": input_params}}}
# {"external": Lkl,
#                              "requires": {"n_d": n_d}}
param_dict = {}
for d, p_d in enumerate(input_params):
    param_dict[p_d] = {"prior": {"min": mean[d]-5*std[d], "max": mean[d]+5*std[d]}}
info["params"] = param_dict

model = get_model(info)
opt = {"n_points_per_acq" : 2}
conv = KL_from_draw_approx(model.prior, {})
m, gp, acq, conv, opt = run(model, convergence_criterion=conv, callback="test")
