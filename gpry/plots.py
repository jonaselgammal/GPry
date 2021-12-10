"""
Utilities to do and modify plots.
"""
import numpy as np
import matplotlib


def getdist_add_training(getdist_plot, model, gpr, colormap="viridis", marker="."):
    """
    Adds the training points to a GetDist triangle plot, coloured according to their
    log-posterior value.
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
        Xs = np.squeeze(Xs[i_within])
        ys = np.squeeze(ys[i_within])
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
