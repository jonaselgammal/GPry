"""
This module provides some plotting routines for plotting the marginalized
posterior distribution and performance of the algorithm.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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
    return getdist_plot

def plot_convergence(convergence_criterion, evaluations="total", marker="_"):
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

    marker : matplotlib marker, optional (default=".")

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
    ax.set_xlabel(f"# {evaluations} posterior evaluations")
    ax.set_ylabel("Value of convergence criterion")
    ax.set_yscale("log")
    ax.grid()
    print(values)
    print(n_posterior_evals)
    print(n_accepted_evals)
    return fig
