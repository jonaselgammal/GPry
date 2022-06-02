"""
This module provides some plotting routines for plotting the marginalized
posterior distribution and performance of the algorithm.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from gpry.tools import cl_of_nstd, nstd_of_cl, volume_sphere


def getdist_add_training(getdist_plot, model, gpr, colormap="viridis", marker="."):
    """
    Adds the training points to a GetDist triangle plot, coloured according to
    their log-posterior value.

    Parameters
    ----------

    getdist_plot : `GetDist triangle plot <https://getdist.readthedocs.io/en/latest/plots.html?highlight=triangle_plot#getdist.plots.GetDistPlotter.triangle_plot>`_
        Contains the marginalized contours and potentially other things.

    model : Cobaya model
        The model that was used to run the GP on

    gpr : GaussianProcessRegressor
        The trained GP Regressor containing the samples.

    colormap : matplotlib colormap, optional (default="viridis")

    marker : matplotlib marker, optional (default=".")

    Returns
    -------

    The GetDist triangle plot with the added training points.
    """
    # Gather axes and bounds
    sampled_params = list(model.parameterization.sampled_params())
    ax_dict = {}
    bounds = [None] * len(sampled_params)
    for i, pi in enumerate(sampled_params):
        for j, pj in enumerate(sampled_params):
            ax = getdist_plot.get_axes_for_params(pi, pj, ordered=True)
            if not ax:
                continue
            ax_dict[(i, j)] = ax
            bounds[i] = ax.get_xlim()
            bounds[j] = ax.get_ylim()
    # Now reduce the set of points to the ones within ranges
    # (needed to get good limits for the colorbar of the log-posterior)
    Xs = np.copy(gpr.X_train)
    ys = np.copy(gpr.y_train)
    for i, (mini, maxi) in enumerate(bounds):
        i_within = np.argwhere(np.logical_and(mini < Xs[:, i], Xs[:, i] < maxi))
        Xs = np.atleast_2d(np.squeeze(Xs[i_within]))
        ys = np.atleast_1d(np.squeeze(ys[i_within]))
    if not len(Xs):  # no points within plotting ranges
        return
    # Create colormap with appropriate limits
    matplotlib.cm.get_cmap(colormap)
    norm = matplotlib.colors.Normalize(vmin=min(ys), vmax=max(ys))
    # Add points
    for (i, j), ax in ax_dict.items():
        points = Xs[:, [i, j]]
        ax.scatter(*points.T, marker=marker, c=norm(ys))

    # TODO: actually add colorbar (see GetDist add_colorbar method)
    return getdist_plot

def plot_convergence(convergence_criterion, evaluations="total", marker=""):
    """
    Plots the value of the convergence criterion as function of the number of
    (accepted) training points.

    Parameters
    ----------

    convergence_criterion : The instance of the convergence criterion which has
        been called in the BO loop

    evaluations : "total" or "accepted"
        Whether to plot the total number of posterior evaluations or only the
        accepted steps.

    marker : matplotlib marker, optional (default="")

    Returns
    -------

    The plot convergence criterion vs. number of training points
    """
    values, n_posterior_evals, n_accepted_evals = convergence_criterion.get_history()
    fig, ax = plt.subplots()
    if evaluations == "total":
        ax.plot(n_posterior_evals, values, marker=marker)
    elif evaluations == "accepted":
        ax.plot(n_accepted_evals, values, marker=marker)
    else:
        raise ValueError("'evaluations' must be either 'total' or 'accepted'.")
    if hasattr(convergence_criterion, "limit"):
        ax.axhline(convergence_criterion.limit, ls="--", lw="0.5", c="0.5")
    ax.set_xlabel(f"# {evaluations} posterior evaluations")
    ax.set_ylabel("Value of convergence criterion")
    ax.set_yscale("log")
    ax.grid()
    return fig, ax


def plot_distance_distribution(points, mean, covmat, density=False):
    """
    Plots a histogram of the distribution of points with respect to the number of standard
    deviations. Bluer stacks represent newer points.

    Confidence level boundaries (dimension-dependent) are shown too.

    If ``density=True`` (default: ``False``), bin height is normalised to the
    (hyper)volume of the (hyper)spherical shell corresponding to each standard deviation.
    """
    dim = np.atleast_2d(points).shape[1]
    mean = np.atleast_1d(mean)
    covmat = np.atleast_2d(covmat)
    assert (mean.shape == (dim,) and covmat.shape == (dim, dim)), \
        (f"Mean and/or covmat have wrong dimensionality: dim={dim}, "
         f"mean.shape={mean.shape} and covmat.shape={covmat.shape}.")
    # Transform to normalised gaussian
    std_diag = np.diag(np.sqrt(np.diag(covmat)))
    invstd_diag = np.linalg.inv(std_diag)
    corrmat = invstd_diag.dot(covmat).dot(invstd_diag)
    Lscalefree = np.linalg.cholesky(corrmat)
    L = np.linalg.inv(std_diag).dot(Lscalefree)
    points_transf = L.dot((points - mean).T).T
    radial_distances = np.sqrt(np.sum(points_transf**2, axis=1))
    bins = list(range(0, int(np.ceil(np.max(radial_distances))) + 1))
    if density:
        volumes = [volume_sphere(bins[i], dim) - volume_sphere(bins[i - 1], dim)
                   for i in range(1, len(bins))]
        weights = [1 / volumes[int(np.floor(r))] for r in radial_distances]
    else:
        weights = np.ones(len(radial_distances))
    fig, ax = plt.subplots()
    cmap = cm.get_cmap('Spectral')
    colors = [cmap(i / len(points)) for i in range(len(points))]
    plt.hist(np.atleast_2d(radial_distances), bins=bins, weights=np.atleast_2d(weights),
             color=colors, stacked=True)
    cls = [cl_of_nstd(1, s) for s in [1, 2, 3, 4]]  # using 1d cl's as reference
    linestyles = ["-", "--", "-.", ":"]
    for cl, ls in zip(cls, linestyles):
        std_of_cl = nstd_of_cl(dim, cl)
        if std_of_cl < max(radial_distances):
            plt.axvline(std_of_cl, c="0.75", ls=ls, zorder=-99,
                        label=f"{cl:.4f}% prob mass")
    num_or_dens = "Density" if density else "Number"
    plt.title(f"{num_or_dens} of points per standard deviation (bluer=newer)")
    plt.ylabel(f"{num_or_dens} of points")
    plt.xlabel("Number of standard deviations")
    plt.legend()
    return fig, ax
