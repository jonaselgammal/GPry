import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from gpry.gpr import GaussianProcessRegressor
from gpry.tools import (
    credibility_of_nstd,
    nstd_of_1d_nstd,
    volume_sphere,
    gaussian_distance,
)


def getdist_add_training(getdist_plot, model, gpr, colormap="viridis",
                         marker=".", marker_inf="x"):
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
        Color map from which to get the color scale to represent the GP model value for
        the training points.

    marker : matplotlib marker, optional (default=".")
        Marker to be used for the training points.

    marker_inf : matplotlib marker, optional (default=".")
        Marker to be used for the non-finite training points.

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
    Xs_finite = np.copy(gpr.X_train)
    ys_finite = np.copy(gpr.y_train)
    Xs_infinite = np.copy(gpr.X_train_infinite)
    for i, (mini, maxi) in enumerate(bounds):
        i_within_finite = np.argwhere(
            np.logical_and(mini < Xs_finite[:, i], Xs_finite[:, i] < maxi))
        Xs_finite = np.atleast_2d(np.squeeze(Xs_finite[i_within_finite]))
        ys_finite = np.atleast_1d(np.squeeze(ys_finite[i_within_finite]))
        i_within_infinite = np.argwhere(
            np.logical_and(mini < Xs_infinite[:, i], Xs_infinite[:, i] < maxi))
        Xs_infinite = np.atleast_2d(np.squeeze(Xs_infinite[i_within_infinite]))
    if not len(Xs_finite) and not len(Xs_infinite):  # no points within plotting ranges
        return
    # Create colormap with appropriate limits
    matplotlib.cm.get_cmap(colormap)
    norm = matplotlib.colors.Normalize(vmin=min(ys_finite), vmax=max(ys_finite))
    # Add points
    for (i, j), ax in ax_dict.items():
        if len(Xs_finite):
            points_finite = Xs_finite[:, [i, j]]
            ax.scatter(*points_finite.T, marker=marker, c=norm(ys_finite), alpha=0.3)
        if len(Xs_infinite):
            points_infinite = Xs_infinite[:, [i, j]]
            ax.scatter(*points_infinite.T, marker=marker_inf, c="k", alpha=0.3)
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
        Marker used for the plot. Will be passed to ``matplotlib.pyplot.plot``.

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
    ax.set_xlabel(f"{evaluations} number of posterior evaluations")
    ax.set_ylabel("Value of convergence criterion")
    ax.set_yscale("log")
    ax.grid()
    return fig, ax


def plot_distance_distribution(
    points, mean, covmat, density=False, show_added=True, ax=None
):
    """
    Plots a histogram of the distribution of points with respect to the number of standard
    deviations. Confidence level boundaries (Gaussian approximantion, dimension-dependent)
    are shown too.

    Parameters
    ----------
    points: array-like, with shape ``(N_points, N_dimensions)``, or GPR instance
        Points to be used for the histogram.
    mean: array-like, ``(N_dimensions)``.
        Mean of the distribution.
    covmat: array-like, ``(N_dimensions, N_dimensions)``.
        Covariance matrix of the distribution.
    density: bool (default: False)
        If ``True``, bin height is normalised to the (hyper)volume of the (hyper)spherical
        shell corresponding to each standard deviation.
    show_added: bool (default True)
        Colours the stacks depending on how early or late the corresponding points were
        added (bluer stacks represent newer points).
    ax: matplotlib axes
        If provided, they will be used for the plot.

    Returns
    -------
    Tuple of current figure and axes ``(fig, ax)``.
    """
    if isinstance(points, GaussianProcessRegressor):
        points = points.X_train
    dim = np.atleast_2d(points).shape[1]
    radial_distances = gaussian_distance(points, mean, covmat)
    bins = list(range(0, int(np.ceil(np.max(radial_distances))) + 1))
    num_or_dens = "Density" if density else "Number"
    if density:
        volumes = [
            volume_sphere(bins[i], dim) - volume_sphere(bins[i - 1], dim)
            for i in range(1, len(bins))
        ]
        weights = [1 / volumes[int(np.floor(r))] for r in radial_distances]
    else:
        weights = np.ones(len(radial_distances))
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    title_str = f"{num_or_dens} of points per standard deviation"
    if show_added:
        title_str += " (bluer=newer)"
        cmap = cm.get_cmap("Spectral")
        colors = [cmap(i / len(points)) for i in range(len(points))]
        ax.hist(
            np.atleast_2d(radial_distances),
            bins=bins,
            weights=np.atleast_2d(weights),
            color=colors,
            stacked=True,
        )
    else:
        ax.hist(radial_distances, bins=bins, weights=weights)
    ax.set_title(title_str)
    cls = [credibility_of_nstd(s, 1) for s in [1, 2, 3, 4]]  # using 1d cl's as reference
    nstds = [1, 2, 3, 4]
    linestyles = ["-", "--", "-.", ":"]
    for nstd, ls in zip(nstds, linestyles):
        std_of_cl = nstd_of_1d_nstd(nstd, dim)
        if std_of_cl < max(radial_distances):
            ax.axvline(
                std_of_cl, c="0.75", ls=ls, zorder=-99,
                label=f"${100 * credibility_of_nstd(std_of_cl, dim):.2f}\%$ prob mass"
            )
    ax.set_ylabel(f"{num_or_dens} of points")
    ax.set_xlabel("Number of standard deviations")
    ax.legend()
    return (fig, ax)


def _plot_2d_model_acquisition(gpr, acquisition, last_points=None, res=200):
    """
    Contour plots for model prediction and acquisition function value of a 2d model.

    If ``last_points`` passed, they are highlighted.
    """
    if gpr.d != 2:
        warnings.warn("This plots are only possible in 2d.")
        return
    # TODO: option to restrict bounds to the min square containing traning samples,
    #       with some padding
    bounds = gpr.bounds
    x = np.linspace(bounds[0][0], bounds[0][1], res)
    y = np.linspace(bounds[1][0], bounds[1][1], res)
    X, Y = np.meshgrid(x, y)
    xx = np.ascontiguousarray(np.vstack([X.reshape(X.size), Y.reshape(Y.size)]).T)
    model_mean = gpr.predict(xx)
    # TODO: maybe change this one below if __call__ method added to GP_acquisition
    acq_value = acquisition(xx, gpr, eval_gradient=False)
    # maybe show the next max of acquisition
    acq_max = xx[np.argmax(acq_value)]
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    cmap = [cm.magma, cm.viridis]
    label = ["Model mean (log-posterior)", "Acquisition function value"]
    for i, Z in enumerate([model_mean, acq_value]):
        ax[i].set_title(label[i])
        # Boost the upper limit to avoid truncation errors.
        Z = np.clip(Z, min(Z[np.isfinite(Z)]), max(Z[np.isfinite(Z)]))
        levels = np.arange(min(Z) * 0.99, max(Z) * 1.01, (max(Z) - min(Z)) / 500)
        Z = Z.reshape(*X.shape)
        norm = cm.colors.Normalize(vmax=Z.max(), vmin=Z.min())
        # # Background of the same color as the bottom of the colormap, to avoid "gaps"
        # plt.gca().set_facecolor(cmap[i].colors[0])
        ax[i].contourf(X, Y, Z, levels, cmap=cm.get_cmap(cmap[i], 256), norm=norm)
        points = ax[i].scatter(
            *gpr.X_train.T, edgecolors="deepskyblue", marker=r"$\bigcirc$")
        # Plot position of next best sample
        point_max = ax[i].scatter(*acq_max, marker="x", color="k")
        if last_points is not None:
            points_last = ax[i].scatter(
                *last_points.T, edgecolors="violet", marker=r"$\bigcirc$")
        # Bounds
        ax[i].set_xlim(bounds[0][0], bounds[0][1])
        ax[i].set_ylim(bounds[1][0], bounds[1][1])
        # Remove ticks, for ilustrative purposes only
        # ax[i].set_xticks([], minor=[])
        # ax[i].set_yticks([], minor=[])
    legend_labels = {points: "Training points"}
    if last_points is not None:
        legend_labels[points_last] = "Points added in last iteration."
    legend_labels[point_max] = "Next optimal location"
    fig.legend(list(legend_labels), list(legend_labels.values()),
               loc="lower center", ncol=99)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15)

def _plot_2d_model_acquisition_finite(gpr, acquisition, last_points=None, res=200):
    """
    Contour plots for model prediction and acquisition function value of a 2d model.

    If ``last_points`` passed, they are highlighted.
    """
    if gpr.d != 2:
        warnings.warn("This plots are only possible in 2d.")
        return
    # TODO: option to restrict bounds to the min square containing traning samples,
    #       with some padding
    bounds = gpr.bounds
    x = np.linspace(bounds[0][0], bounds[0][1], res)
    y = np.linspace(bounds[1][0], bounds[1][1], res)
    X, Y = np.meshgrid(x, y)
    xx = np.ascontiguousarray(np.vstack([X.reshape(X.size), Y.reshape(Y.size)]).T)
    model_mean = gpr.predict(xx)
    # TODO: maybe change this one below if __call__ method added to GP_acquisition
    acq_value = acquisition(xx, gpr, eval_gradient=False)
    # maybe show the next max of acquisition
    acq_max = xx[np.argmax(acq_value)]
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    cmap = [cm.magma, cm.viridis]
    label = ["Model mean (log-posterior)", "Acquisition function value"]
    for i, Z in enumerate([model_mean, acq_value]):
        ax[i].set_title(label[i])
        # Boost the upper limit to avoid truncation errors.
        Z_finite = Z[np.isfinite(Z)]
        #Z_clipped = np.clip(Z_finite, min(Z[np.isfinite(Z)]), max(Z[np.isfinite(Z)]))
        Z_sort = np.sort(Z_finite)[::-1]
        top_x_perc = np.sort(Z_finite)[::-1][:int(len(Z_finite)*0.5)]
        relevant_range = max(top_x_perc)-min(top_x_perc)
        levels = np.linspace(max(Z_finite)-1.99*relevant_range, max(Z_finite) + 0.01*relevant_range,  500)
        Z[np.isfinite(Z)] = np.clip(Z_finite, min(levels), max(levels))
        Z = Z.reshape(*X.shape)
        norm = cm.colors.Normalize(vmax=max(levels), vmin=min(levels))
        ax[i].set_facecolor('grey')
        # # Background of the same color as the bottom of the colormap, to avoid "gaps"
        # plt.gca().set_facecolor(cmap[i].colors[0])
        ax[i].contourf(X, Y, Z, levels, cmap=cm.get_cmap(cmap[i], 256), norm=norm)
        points = ax[i].scatter(
            *gpr.X_train.T, edgecolors="deepskyblue", marker=r"$\bigcirc$")
        # Plot position of next best sample
        point_max = ax[i].scatter(*acq_max, marker="x", color="k")
        if last_points is not None:
            points_last = ax[i].scatter(
                *last_points.T, edgecolors="violet", marker=r"$\bigcirc$")
        # Bounds
        ax[i].set_xlim(bounds[0][0], bounds[0][1])
        ax[i].set_ylim(bounds[1][0], bounds[1][1])
        # Remove ticks, for ilustrative purposes only
        # ax[i].set_xticks([], minor=[])
        # ax[i].set_yticks([], minor=[])
    legend_labels = {points: "Training points"}
    if last_points is not None:
        legend_labels[points_last] = "Points added in last iteration."
    legend_labels[point_max] = "Next optimal location"
    fig.legend(list(legend_labels), list(legend_labels.values()),
               loc="lower center", ncol=99)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15)

def _plot_2d_model_acquisition_std(gpr, acquisition, last_points=None, res=200):
    """
    Contour plots for model prediction and acquisition function value of a 2d model.

    If ``last_points`` passed, they are highlighted.
    """
    if gpr.d != 2:
        warnings.warn("This plots are only possible in 2d.")
        return
    # TODO: option to restrict bounds to the min square containing traning samples,
    #       with some padding
    bounds = gpr.bounds
    x = np.linspace(bounds[0][0], bounds[0][1], res)
    y = np.linspace(bounds[1][0], bounds[1][1], res)
    X, Y = np.meshgrid(x, y)
    xx = np.ascontiguousarray(np.vstack([X.reshape(X.size), Y.reshape(Y.size)]).T)
    model_mean,model_std = gpr.predict(xx,return_std=True)
    # TODO: maybe change this one below if __call__ method added to GP_acquisition
    acq_value = acquisition(xx, gpr, eval_gradient=False)
    # maybe show the next max of acquisition
    acq_max = xx[np.argmax(acq_value)]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    cmap = [cm.magma, cm.viridis,cm.magma]
    label = ["Model mean (log-posterior)", "Acquisition function value","Model std dev."]
    for i, Z in enumerate([model_mean, acq_value]):
        ax[i].set_title(label[i])
        # Boost the upper limit to avoid truncation errors.
        Z_finite = Z[np.isfinite(Z)]
        #Z_clipped = np.clip(Z_finite, min(Z[np.isfinite(Z)]), max(Z[np.isfinite(Z)]))
        Z_sort = np.sort(Z_finite)[::-1]
        top_x_perc = np.sort(Z_finite)[::-1][:int(len(Z_finite)*0.5)]
        relevant_range = max(top_x_perc)-min(top_x_perc)
        levels = np.linspace(max(Z_finite)-1.99*relevant_range, max(Z_finite) + 0.01*relevant_range,  500)
        Z[np.isfinite(Z)] = np.clip(Z_finite, min(levels), max(levels))
        Z = Z.reshape(*X.shape)
        norm = cm.colors.Normalize(vmax=max(levels), vmin=min(levels))
        ax[i].set_facecolor('grey')
        # # Background of the same color as the bottom of the colormap, to avoid "gaps"
        # plt.gca().set_facecolor(cmap[i].colors[0])
        ax[i].contourf(X, Y, Z, levels, cmap=cm.get_cmap(cmap[i], 256), norm=norm)
        points = ax[i].scatter(
            *gpr.X_train.T, edgecolors="deepskyblue", marker=r"$\bigcirc$")
        # Plot position of next best sample
        point_max = ax[i].scatter(*acq_max, marker="x", color="k")
        if last_points is not None:
            points_last = ax[i].scatter(
                *last_points.T, edgecolors="violet", marker=r"$\bigcirc$")
        # Bounds
        ax[i].set_xlim(bounds[0][0], bounds[0][1])
        ax[i].set_ylim(bounds[1][0], bounds[1][1])
        # Remove ticks, for ilustrative purposes only
        # ax[i].set_xticks([], minor=[])
        # ax[i].set_yticks([], minor=[])
    ax[2].set_title(label[2])
    Z = model_std
    Z_finite = Z[np.isfinite(model_mean)]
    Z[~np.isfinite(model_mean)] = -np.inf
    minz = min(Z_finite)
    zrange = max(Z_finite)-minz
    levels = np.linspace(minz, minz + (zrange if zrange>0 else 0.00001),  500)
    #Z[np.isfinite(model_mean)] = np.clip(Z_finite, min(levels), max(levels))
    Z = Z.reshape(*X.shape)
    norm = cm.colors.Normalize(vmax=max(levels), vmin=min(levels))
    ax[2].set_facecolor('grey')
    ax[2].contourf(X, Y, Z, levels, cmap=cm.get_cmap(cmap[2], 256), norm=norm)
    points = ax[2].scatter(
        *gpr.X_train.T, edgecolors="deepskyblue", marker=r"$\bigcirc$")
    # Plot position of next best sample
    point_max = ax[2].scatter(*acq_max, marker="x", color="k")
    if last_points is not None:
        points_last = ax[2].scatter(
            *last_points.T, edgecolors="violet", marker=r"$\bigcirc$")
    # Bounds
    ax[2].set_xlim(bounds[0][0], bounds[0][1])
    ax[2].set_ylim(bounds[1][0], bounds[1][1])
    legend_labels = {points: "Training points"}
    if last_points is not None:
        legend_labels[points_last] = "Points added in last iteration."
    legend_labels[point_max] = "Next optimal location"
    fig.legend(list(legend_labels), list(legend_labels.values()),
               loc="lower center", ncol=99)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15)
