"""
Module gathering general plots.

(Other plots are in methods of some classes, e.g. Progress contains the timings plot.)
"""

import warnings
from typing import Sequence, Mapping
from numbers import Number

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

from gpry.gpr import GaussianProcessRegressor
from gpry.mc import process_gdsamples
from gpry.tools import (
    credibility_of_nstd,
    nstd_of_1d_nstd,
    volume_sphere,
    gaussian_distance,
    delta_logp_of_1d_nstd,
    generic_params_names,
)

# Use latex labels when available
plt.rcParams["text.usetex"] = True
_plot_dist_fontsize = 7

# Param name and label for log-posterior in corner plots
_name_logp, _label_logp = "logpost", r"\log(p)"


def simple_latex_sci_notation(string):
    """
    If ``string`` contains a ``%g`` or ``%e`` number representation, substitutes the ``e``
    for a power of 10.

    It does *not* add dollars around the string.

    NB: it assumes that the string passed contains a single number, and nothing else.
    """
    if "e" not in string:
        return string
    sigfigs, exp = string.split("e")
    sign = "" if exp.startswith("+") else "-"
    return f"{sigfigs}\\cdot 10^{{{sign}{exp[1:].lstrip('0')}}}"


def param_samples_for_slices(X, i, bounds, n=200):
    """
    From an array of points `X = [X^i] = [[X^1_1, X^1_2,...], [X^2_1,...], ...]`, it
    generates a list of points per sample, where the `i` coordinate is sliced within the
    region defined by `bounds`, and the rest of them are kept fixed.
    """
    X = np.atleast_2d(X)
    Xs_i = np.linspace(bounds[0], bounds[1], n)
    X_slices = np.empty(shape=(X.shape[0], n, X.shape[1]), dtype=float)
    for j, X_j in enumerate(X):
        X_slices[j, :, :] = n * [X_j]
    X_slices[:, :, i] = Xs_i
    return X_slices


def prepare_slices_func(func, X_fiducial, bounds, indices=None, n=50):
    """
    Prepare slices of the given function,

    Parameters
    ----------
    func : callable
        Function for which to prepare slices. It needs to take arguments in the way
        ``X_fiducial`` is passed: ``func(*X_fiducial)`` if ``X_fiducual`` is a list, or
        ``func(**X_fiducial)`` if X_fiducial is a dictionary.

    X_fiducial : array-like, shape = (n_dimensions), or dict
        Fiducial point for the slices: slice ``i`` corresponds to fixing all parameters
        but that with index ``i``, which is evaluaded on a grid within its bounds. It can
        be a dictionary with arguments of ``func`` as keys.

    bounds : array-like, shape = (n_dimensions, 2), or dict
        Bounds for the slices per parameter.

    indices : list, optional
        A list of integers (if ``X_fiducial`` is a list) or parameter names (if
        ``X_fiducial`` is a dict), denoting the parameters for which the slices will be
        prepared (all of them, if left unspecified).

    n : int
        Number of samples per slice (default: 50). Careful if the posterior is slow!

    Returns
    -------
    indices, params, Xs, ys : list(int), list(str) len=len(indices), array-like
                     shape=(len(indices), n, dim)), array-like shape=(len(indices, n))
    """
    # Parse and check input
    if isinstance(X_fiducial, Mapping):
        is_kwarg = True
        dim = len(X_fiducial)
        X_fiducial = np.array(list(X_fiducial.values()))
        params = list(X_fiducial)
        if indices is None:
            indices = params
        try:  # Assumes indices is a list of str
            indices = [params.index(p) for p in indices]
        except ValueError as excpt:
            raise ValueError(
                "`indices` is not a list of parameter names, or contains names not in "
                "`X_fiducial`."
            ) from excpt
        try:  # Assumes bonds is a Mapping
            bounds = [bounds[p] for p in params]
        except (TypeError, IndexError) as excpt:
            raise ValueError(
                "`bounds` is not a dict, or bounds could not be founds for all "
                "parameters."
            ) from excpt
    else:
        is_kwarg = False
        X_fiducial = np.atleast_1d(X_fiducial)
        dim = len(X_fiducial)
        params = generic_params_names(dim)
        if indices is None:
            indices = list(range(dim))
        # Assumes indices is a list of int
        if not all((isinstance(i, int) and i < dim) for i in indices):
            raise ValueError(
                "`indices` is not a list of integer indices, or contains indices larger "
                "than the length of `X_fiducial`."
            )
        # Assumes bonds is a (2d) list
        if len(bounds) < len(params):
            raise ValueError("`bounds` is not a list of bounds of the right lenght.")
    if not isinstance(n, int) or n < 2:
        raise ValueError("`n` must be a positive integer > 2.")
    # Prepare and evaluate slices
    Xs, ys = np.empty(shape=(len(indices), n, dim)), np.empty(shape=(len(indices), n))
    for j, index in enumerate(indices):
        Xs[j] = param_samples_for_slices([X_fiducial], index, bounds[index], n=n)[0]
        progress_bar_desc = f"Slicing param {j + 1} of {len(indices)}"
        for k, x in tqdm(enumerate(Xs[j]), total=n, desc=progress_bar_desc):
            try:
                if is_kwarg:
                    x_arg = dict(zip(params, x))
                    ys[j][k] = func(**x_arg)
                else:
                    x_arg = x
                    ys[j][k] = func(*x_arg)
            except TypeError as excpt:
                raise TypeError(
                    f"Could not call the target function with arguments {x_arg}. Maybe "
                    "`X_fiducial` contained keys that are not argument names of the "
                    f"function? Err msg: {excpt}"
                ) from excpt
            except Exception as excpt:  # pylint: disable=broad-exception-caught
                warnings.warn(
                    f"The function failed when called with arguments {x_arg}. Using NaN."
                    f"Err masg: {excpt}"
                )
                ys[j][k] = np.nan
    return indices, [params[i] for i in indices], Xs, ys


def plot_slices_func(
        func, X_fiducial, bounds, indices=None, n=50, fig_kwargs=None, labels=None,
):
    """
    Plot slices of the given function,

    Parameters
    ----------

    func : callable
        Function for which to prepare slices. It needs to take arguments in the way
        ``X_fiducial`` is passed: ``func(*X_fiducial)`` if ``X_fiducual`` is a list, or
        ``func(**X_fiducial)`` if X_fiducial is a dictionary.

    X_fiducial : array-like, shape = (n_dimensions), or dict
        Fiducial point for the slices: slice ``i`` corresponds to fixing all parameters
        but that with index ``i``, which is evaluaded on a grid within its bounds. It can
        be a dictionary with arguments of ``func`` as keys.

    bounds : array-like, shape = (n_dimensions, 2), or dict
        Bounds for the slices per parameter.

    indices : list, optional
        A list of integers (if ``X_fiducial`` is a list) or parameter names (if
        ``X_fiducial`` is a dict), denoting the parameters for which the slices will be
        prepared (all of them, if left unspecified).

    n : int
        Number of samples per slice (default: 50). Careful if the posterior is slow!

    fig_kwargs : dict, optional
        Dict of kw arguments to pass to the `subplots` constructor. Only ``layout``,
        ``dpi`` considered safe.

    labels : lst(str), optional
        Strings (possibly Latex) to use for axes labels. Length cases: None or len=0:
        plain parameter names for x labels and no y label; len=1: used as y label, plain
        names for x labels; len=len(indices): used as x labels; len=len(indices)+1: used
        as x labels and y label, in that order.

    Returns
    -------
    fig, axarr: figure and array of axes used for the plot.
    """
    indices, params, Xs, ys = prepare_slices_func(
        func, X_fiducial, bounds, indices=indices, n=n
    )
    if not isinstance(labels, Sequence) or len(labels) == 0:
        x_labels = params
        y_label = None
    elif len(labels) == 1:
        x_labels = params
        y_label = labels[0]
    elif len(labels) == len(params):
        x_labels = labels
        y_label = None
    elif len(labels) == len(params) + 1:
        x_labels = labels[:-1]
        y_label = labels[-1]
    else:
        raise ValueError("Value for `labels` not recognised, or length not valid.")
    fig_kwargs_defaults = dict(
        nrows=1,
        ncols=len(indices),
        layout="constrained",
        figsize=(4 * len(indices), 2),
        dpi=200,
    )
    fig_kwargs_defaults.update(fig_kwargs or {})
    fig, axes = plt.subplots(**fig_kwargs_defaults)
    if not isinstance(axes, Sequence):
        axes = [axes]
    color = "tab:blue"
    for j, (i, p) in enumerate(zip(indices, params)):
        axes[j].axvline(X_fiducial[i], c="0.75", ls="--")
        axes[j].plot(Xs[j, :, i], ys[j], c=color)
        axes[j].scatter(Xs[j, :, i], ys[j], marker=".", s=10, c=color)
        axes[j].set_xlabel(x_labels[j])
        axes[j].set_ylabel(y_label)
    return fig, axes


def plot_slices(model, gpr, acquisition, X=None, reference=None):
    """
    Plots slices along parameter coordinates for a series `X` of given points (the GPR
    training set if not specified). For each coordinate, there is a slice per point,
    leaving all coordinates of that point fixed except for the one being sliced.

    Lines are coloured according to the value of the mean GP at points X.

    # TODO: make acq func optional
    """
    params = list(model.parameterization.sampled_params())
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(params),
        sharex="col",
        layout="constrained",
        figsize=(4 * len(params), 4),
        dpi=200,
    )
    # Define X to plot
    if X is None:
        X = gpr.X_train.copy()
        y = gpr.y_train.copy()
    else:
        y = gpr.predict(X)
    min_y, max_y = min(y), max(y)
    norm_y = lambda y: (y - min_y) / (max_y - min_y)
    prior_bounds = model.prior.bounds(confidence_for_unbounded=0.999)
    Xs_for_plots = dict(
        (p, param_samples_for_slices(X, i, prior_bounds[i], n=200))
        for i, p in enumerate(params)
    )
    if reference is not None:
        reference = _prepare_reference(reference, model)
    cmap = matplotlib.colormaps["viridis"]
    for i, p in enumerate(params):
        for j, Xs_j in enumerate(Xs_for_plots[p]):
            cmap_norm = cmap(norm_y(y[j]))
            alpha = 1
            # TODO: could cut by half # of GP evals by reusing for acq func
            axes[0, i].plot(Xs_j[:, i], gpr.predict(Xs_j), c=cmap_norm, alpha=alpha)
            axes[0, i].scatter(X[j][i], y[j], color=cmap_norm, alpha=alpha)
            axes[0, i].set_ylabel(r"$\log(p)$")
            acq_values = acquisition(Xs_j, gpr)
            axes[1, i].plot(Xs_j[:, i], acq_values, c=cmap_norm, alpha=alpha)
            axes[1, i].set_ylabel(r"$\alpha(\mu,\sigma)$")
            label = model.parameterization.labels()[p]
            if label != p:
                label = "$" + label + "$"
            axes[1, i].set_xlabel(label)
            bounds = (reference or {}).get(p)
        if bounds is not None:
            for ax in axes[:, i]:
                if len(bounds) == 5:
                    ax.axvspan(
                        bounds[0], bounds[4], facecolor="tab:blue", alpha=0.2, zorder=-99
                    )
                    ax.axvspan(
                        bounds[1], bounds[3], facecolor="tab:blue", alpha=0.2, zorder=-99
                    )
                ax.axvline(bounds[2], c="tab:blue", alpha=0.3, ls="--")


def plot_slices_reference(model, gpr, X, truth=True, reference=None):
    """
    Plots slices of the gpr model and true log-posterior (if ``truth=True``) along
    parameter coordinates for a given point ``X``, leaving all coordinates of that point
    fixed except for the one being sliced.
    """
    params = list(model.parameterization.sampled_params())
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(params),
        sharex="col",
        layout="constrained",
        figsize=(4 * len(params), 2),
        dpi=200,
    )
    prior_bounds = model.prior.bounds(confidence_for_unbounded=0.999)
    if X is None:
        if reference is None:
            raise ValueError("Needs at least a reference point or a reference sample.")
        # TODO: if reference given as a sample, take best point from it.
    X_array = np.array([X[p] for p in model.parameterization.sampled_params()])
    # y_gpr_centre = gpr.predict(np.atleast_2d(X_array))[0]
    if truth:
        y_truth_centre = model.logpost(X_array)
    Xs_for_plots, ys_gpr_for_plot, sigmas_gpr_for_plot, ys_truth_for_plot = {}, {}, {}, {}
    for i, p in enumerate(params):
        Xs_for_plots[p] = param_samples_for_slices([X_array], i, prior_bounds[i], n=200)[
            0
        ]
        ys_gpr_for_plot[p], sigmas_gpr_for_plot[p] = gpr.predict(
            Xs_for_plots[p], return_std=True
        )
        if truth:
            ys_truth_for_plot[p] = np.array([model.logpost(x) for x in Xs_for_plots[p]])
    # Training set referenced to X and div by X, for distance-based transparency
    X_train_diff = (gpr.X_train - X_array) / X_array
    if reference is not None:
        reference = _prepare_reference(reference, model)
    for i, p in enumerate(params):
        axes[i].fill_between(
            Xs_for_plots[p][:, i],
            ys_gpr_for_plot[p] - 2 * sigmas_gpr_for_plot[p],
            ys_gpr_for_plot[p] + 2 * sigmas_gpr_for_plot[p],
            color="0.5",
            alpha=0.25,
            edgecolor="none",
        )
        axes[i].fill_between(
            Xs_for_plots[p][:, i],
            ys_gpr_for_plot[p] - 1 * sigmas_gpr_for_plot[p],
            ys_gpr_for_plot[p] + 1 * sigmas_gpr_for_plot[p],
            color="0.5",
            alpha=0.25,
            edgecolor="none",
        )
        if truth:
            axes[i].plot(Xs_for_plots[p][:, i], ys_truth_for_plot[p], ls="--")
            axes[i].scatter(X[p], y_truth_centre, marker="*")
        axes[i].set_ylabel(r"$\log(p)$")
        label = model.parameterization.labels()[p]
        if label != p:
            label = "$" + label + "$"
        axes[i].set_xlabel(label)
        # If there is an infinities classifier, use if for the lower bound on y:
        diff_min_logp = getattr(gpr, "diff_threshold", None)
        if diff_min_logp is not None:
            try:
                max_y = max(ys_gpr_for_plot[p])
                upper_y = max_y
                if truth:
                    upper_y = max(max_y, *ys_truth_for_plot[p])
                axes[i].set_ylim(
                    max_y - 1.05 * diff_min_logp, upper_y + 0.05 * diff_min_logp
                )
            except ValueError as e:
                print(
                    f"ERROR when setting y-lims for '{p}': max(y) was "
                    f"{max(ys_gpr_for_plot[p])}, diff_threshold was {diff_min_logp}, "
                    f"lower_bound was {max(ys_gpr_for_plot[p]) - 1.05 * diff_min_logp}, "
                    f"upper bound was {upper_y}, MSG was {e}"
                )
        # Add training set
        dists = np.sqrt(np.sum(np.power(np.delete(X_train_diff, i, axis=-1), 2), axis=-1))
        dists_relative = dists / max(dists)
        axes[i].scatter(
            gpr.X_train[:, i],
            gpr.y_train,
            marker=".",
            alpha=1 - dists_relative,
            zorder=-9,
        )
        bounds = (reference or {}).get(p)
        if bounds is not None:
            if len(bounds) == 5:
                axes[i].axvspan(
                    bounds[0], bounds[4], facecolor="tab:blue", alpha=0.2, zorder=-99
                )
                axes[i].axvspan(
                    bounds[1], bounds[3], facecolor="tab:blue", alpha=0.2, zorder=-99
                )
                axes[i].axvline(bounds[2], c="tab:blue", alpha=0.3, ls="--")


def plot_corner_getdist(
        mc_samples,
        params=None,
        filled=None,
        training=None,
        training_highlight_last=False,
        markers=None,
        output=None,
        output_dpi=200,
        subplot_size=2,
):
    """
    Creates a corner plot the given MC samples, and optionally shows evaluation locations.

    If called repeatedly, it may leak memory, unfortunately. To avoid this, execute it
    like this:

    .. code:: python

        # Temporarily switch to Agg backend
        prev_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        try:
            plot_corner_getdist(...)
        except:
            ...
        finally:
            # Switch back to prev backend
            mpl.use(prev_backend)

    Parameters
    ----------

    mc_samples: dict(str, (cobaya.SampleCollection, getdist.MCSamples, str))
        Dict of MC samples, with their plot label as key, and the sample as value, either
        as GetDist or Cobaya samples, or as a path where there are samples saved.

    params : list(str), optional
        List of parameter names to be plotted, by default all of the ones in the first
        MC sample, included derived ones like probability densities.

    filled : dict(str, bool)
        Dictionary with labels as keys specifying the `filled` property of the contours.
        Contours are filled by default when unspecified (including key missing for a
        passed sample).

    training : GaussianProcessRegressor, dict(str or tuple, GaussianProcessRegressor), optional
        If a GPR is passed, it plots the training samples (including the discarded ones)
        on top of the contours. Samples outside the axes ranges are not plotted.
        The parameters of the GPR need to be assumed, since the GPR does not save names:
        if a GPR is passed, the sampled parameters of the first MC sample will be used; if
        a single-key dict is passed with a str as a key, it will used the parameter names
        from the MC sample with that label; if the key is a tuple of strings, they will be
        used as parameters

    subplot_size : float, default = 2
        Size of each subplot in the corner plot.

    output : str, optional (default=None)
        Path, including name and extension, of the saved figure.
        Not saved if left unspecified.

    output_dpi : int (default: 200)
        The resolution of the generated plot in DPI.

    Returns
    -------
    getdist.plots.GetDistPlotter object containing the figure.
    """
    if not isinstance(mc_samples, Mapping):
        raise TypeError(
            "The first argument must be a list of MC samples with the sample legend "
            "labels as keys."
        )
    gdsamples_dict = process_gdsamples(mc_samples)
    # Prepare training samples early -- fail asap.
    training_params = None
    if training is not None:
        if isinstance(training, Mapping):
            training_key = list(training)[0]
            training = list(training.values())[0]
            if isinstance(training_key, tuple):
                training_params = training_key
            else:  # assumed key corresponding to passed mcsamples
                if training_key not in mc_samples:
                    raise ValueError(
                        "`training` passed as dict, but key not found in mc_samples."
                    )
                training_params = \
                    gdsamples_dict[training_key].getParamNames().getRunningNames()
        elif isinstance(training, GaussianProcessRegressor):
            # Use first MC passed
            training_params = \
                gdsamples_dict[list(gdsamples_dict)[0]].getParamNames().getRunningNames()
        else:
            raise TypeError("'training' is not a GaussianProcessRegressor instance.")
    import getdist.plots as gdplt  # pylint: disable=import-outside-toplevel
    gdplot = gdplt.get_subplot_plotter(subplot_size=subplot_size, auto_close=True)
    gdplot.settings.line_styles = 'tab10'
    gdplot.settings.solid_colors = 'tab10'
    triang_args = [list(gdsamples_dict.values())]
    if params is not None:
        triang_args.append(params)
    triang_kwargs = {
        "legend_labels": list(gdsamples_dict),
        "filled": [(filled or {}).get(k, True) for k in gdsamples_dict],
        "markers": markers,
    }
    try:
        gdplot.triangle_plot(*triang_args, **triang_kwargs)
    except Exception as excpt:
        raise ValueError(
            f"Could not do corner plot. GetDist err. msg.: {excpt}"
        ) from excpt
    if training is not None and training.d > 1:
        getdist_add_training(
            gdplot, training_params, training, highlight_last=training_highlight_last
        )
    if output is not None:
        plt.savefig(output, dpi=output_dpi)
    return gdplot


def getdist_add_training(
    getdist_plot,
    params,
    gpr,
    colormap="viridis",
    marker=".",
    marker_inf="x",
    highlight_last=False,
):
    """
    Adds the training points to a GetDist triangle plot, coloured according to
    their log-posterior value.

    Parameters
    ----------
    getdist_plot : `GetDist triangle plot <https://getdist.readthedocs.io/en/latest/plots.html?highlight=triangle_plot#getdist.plots.GetDistPlotter.triangle_plot>`_
        Contains the marginalized contours and potentially other things.

    params : list(str)
        The assumed parameter names for the GPR samples. Need to be a subset of the ones
        plotted by the GetDistPlotter.

    gpr : GaussianProcessRegressor
        The trained GP Regressor containing the samples.

    colormap : matplotlib colormap, optional (default="viridis")
        Color map from which to get the color scale to represent the GP model value for
        the training points.

    marker : matplotlib marker, optional (default=".")
        Marker to be used for the training points.

    marker_inf : matplotlib marker, optional (default=".")
        Marker to be used for the non-finite training points.

    highlight_last: bool (default=False)
        Draw a red circle around the points added in the last iteration

    Returns
    -------
    The GetDist triangle plot with the added training points.
    """
    # Gather axes and bounds
    d = len(params)
    ax_dict = {}
    bounds = [None] * len(params)
    for i, pi in enumerate(params):
        for j, pj in enumerate(params):
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
            np.logical_and(mini < Xs_finite[:, i], Xs_finite[:, i] < maxi)
        )
        Xs_finite = np.atleast_2d(np.squeeze(Xs_finite[i_within_finite]))
        ys_finite = np.atleast_1d(np.squeeze(ys_finite[i_within_finite]))
        i_within_infinite = np.argwhere(
            np.logical_and(mini < Xs_infinite[:, i], Xs_infinite[:, i] < maxi)
        )
        Xs_infinite = np.atleast_2d(np.squeeze(Xs_infinite[i_within_infinite]))
        if highlight_last:
            Xs_last = gpr.last_appended[0]
            i_within_last = np.argwhere(
                np.logical_and(mini < Xs_last[:, i], Xs_last[:, i] < maxi)
            )
            Xs_last = np.atleast_2d(np.squeeze(Xs_last[i_within_last]))
    if len(Xs_finite) == 0 and len(Xs_infinite) == 0:  # no points within plotting ranges
        return getdist_plot
    # Create colormap with appropriate limits
    cmap = matplotlib.colormaps[colormap]
    if len(Xs_finite):
        Ncolors = 256
        color_bounds = np.linspace(min(ys_finite), max(ys_finite), Ncolors)
        norm = matplotlib.colors.BoundaryNorm(color_bounds, Ncolors)
    # Add points
    for (i, j), ax in ax_dict.items():
        # 1st -inf points, so the are displayed in the background of the finite ones.
        # and we give them low zorder anyway, so that they lay behind the contours
        if len(Xs_infinite) > 0:
            points_infinite = Xs_infinite[:, [i, j]]
            ax.scatter(
                *points_infinite.T, marker=marker_inf, s=20, c="k", alpha=0.3, zorder=-99
            )
        if len(Xs_finite) > 0:
            points_finite = Xs_finite[:, [i, j]]
            ax.scatter(
                *points_finite.T, marker=marker, c=norm(ys_finite), alpha=0.3, cmap=cmap
            )
        if highlight_last and len(Xs_last) > 0:
            points_last = Xs_last[:, [i, j]]
            ax.scatter(
                *points_last.T,
                marker="o",
                c=len(points_last) * [[0, 0, 0, 0]],
                edgecolor="r",
                lw=0.5,
            )
    # Colorbar
    if len(Xs_finite) > 0 and not np.isclose(min(ys_finite), max(ys_finite)):
        getdist_plot.fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            label=r"$\log(p)$",
            ax=getdist_plot.fig.add_axes(
                [1 - 0.2 / d, 1 - 0.85 / d, 0.5 / d, 0.5 / d],
                frame_on=False,
                xticks=[],
                yticks=[],
            ),
            ticks=np.linspace(min(ys_finite), max(ys_finite), 5),
            location="left",
        )
    return getdist_plot


def plot_convergence(
    convergence_criterion,
    evaluations="total",
    marker="",
    axes=None,
    ax_labels=True,
    legend_loc="upper right",
):
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

    axes : matplotlib axes, optional
        Axes to be used, if passed.

    ax_labels : bool, optional (default: True)
        Add axes labels.

    legend_loc : str (default: "upper right")
        Location of the legend.

    Returns
    -------
    The plot convergence criterion vs. number of training points
    """
    if not isinstance(convergence_criterion, Sequence):
        convergence_criterion = [convergence_criterion]
    if axes is None:
        fig, axes = plt.subplots()
    else:
        fig = axes.get_figure()
    for i, cc in enumerate(convergence_criterion):
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
        values, n_posterior_evals, n_accepted_evals = cc.get_history()
        name = cc.__class__.__name__
        n_evals = np.array(
            {"total": n_posterior_evals, "accepted": n_accepted_evals}[evaluations],
            dtype=int,
        )
        try:
            axes.plot(n_evals, values, marker=marker, color=color, label=name)
        except KeyError as excpt:
            raise ValueError(
                "'evaluations' must be either 'total' or 'accepted'."
            ) from excpt
        if hasattr(cc, "limit"):
            axes.axhline(cc.limit, ls="--", lw=1, c=color)
    if ax_labels:
        axes.set_xlabel(f"{evaluations} number of posterior evaluations")
        axes.set_ylabel("Value of convergence criterion")
    axes.set_yscale("log")
    axes.grid(axis="y")
    axes.legend(loc=legend_loc, prop={"size": _plot_dist_fontsize})
    return fig, axes


def _prepare_reference(
    reference,
    model,
):
    """
    Turns `reference` into a dict with parameters as keys and a list of 5 numbers as
    values: two lower bounds, a central value, and two upper bounds, e.g. percentiles
    5, 25, 50, 75, 95.

    If getdist.MCSamples passed, bounds are by default 68% and 95%, and the central value
    is the mean.
    """
    # Ensure it is a dict
    try:
        from getdist import MCSamples  # pylint: disable=import-outside-toplevel

        if isinstance(reference, MCSamples):
            means = reference.getMeans()
            margstats = reference.getMargeStats()
            bounds = {}
            for p in model.parameterization.sampled_params():
                # NB: numerOfName doest not use renames; needs to find "original" name
                p_in_ref = reference.paramNames.parWithName(p).name
                i_p = reference.paramNames.numberOfName(p_in_ref)
                # by default lims/contours are [68, 95, 99]
                try:
                    lims = margstats.parWithName(p).limits
                except AttributeError as excpt:
                    raise ValueError(
                        f"Could not find parameter {p} in reference sample, which "
                        f"includes {reference.getParamNames().list()})"
                    ) from excpt
                bounds[p] = [
                    lims[1].lower,
                    lims[0].lower,
                    means[i_p],
                    lims[0].upper,
                    lims[1].upper,
                ]
            reference = bounds
    except ModuleNotFoundError:  # getdist not installed
        return None
    if not isinstance(reference, Mapping):
        # Assume parameters in order; check right number of them
        if len(reference) != model.prior.d():
            raise ValueError(
                "reference must be a list containing bounds per parameter for all of them"
                ", or a dict with parameters as keys and these same values."
            )
        reference = dict(zip(model.parameterization.sampled_params(), reference))
    # Ensure it contains all parameters and 5 numbers (or None's) per parameter
    for p in model.parameterization.sampled_params():
        if p not in reference:
            reference[p] = [None] * 5
        values = reference[p]
        if isinstance(values, Number):
            values = [values]
        if len(values) == 1:
            reference[p] = [None, None] + list(values) + [None, None]
        elif len(values) != 5:
            raise ValueError(
                "the elements of reference must be either a single central value, or a "
                "list of 5 elements: [lower_bound_2, lower_bound_1, central_value, "
                "upper_bound_2, upper_bound_1]."
            )
    return reference


def plot_trace(
    model,
    gpr,
    convergence_criterion,
    progress,
    colormap="viridis",
    reference=None,
):
    """
    Plots the evolution of the run along true model evaluations, showing evolution of the
    convergence criterion and the values of the log-posterior and the individual
    parameters.

    Can take a reference sample or reference bounds (dict with parameters as keys and 5
    sorted bounds as values, or alternatively just a central value).
    """
    X = gpr.X_train_all
    y = gpr.y_train_all
    if gpr.infinities_classifier is not None:
        y_finite = gpr.infinities_classifier.y_finite
    else:
        y_finite = np.full(shape=len(y), fill_value=True)
    if reference is not None:
        reference = _prepare_reference(reference, model)
    fig, axes = plt.subplots(
        nrows=2 + model.prior.d(),
        ncols=1,
        sharex=True,
        layout="constrained",
        figsize=(min(4, 0.3 * len(X)), 1.5 * (2 + X.shape[1])),
        dpi=400,
    )
    i_eval = list(range(1, 1 + len(X)))
    # TOP: convergence plot
    try:
        plot_convergence(
            convergence_criterion,
            evaluations="total",
            marker="",
            axes=axes[0],
            ax_labels=False,
            legend_loc="lower left",
        )
    except ValueError:  # no criterion computed yet
        pass
    axes[0].set_ylabel("Conv. crit.")
    # 2nd: posterior plot
    kwargs_accepted = {
        "marker": ".",
        "linewidths": 0.1,
        "edgecolor": "0.1",
        "cmap": colormap,
    }
    axes[1].scatter(i_eval, y, c=np.where(y_finite, y, np.inf), **kwargs_accepted)
    # Gaussian contours
    dashdotdotted = (0, (3, 5, 1, 5, 1, 5))
    nsigmas_styles = {1: "-", 2: "--", 5: "-.", 10: ":", 20: dashdotdotted}
    y_min_plot, y_max_plot = axes[1].get_ylim()
    y_max = np.max(y)
    for ns, nsls in nsigmas_styles.items():
        y_ns = y_max - delta_logp_of_1d_nstd(ns, model.prior.d())
        if y_ns > y_min_plot:
            axes[1].axhline(
                y_ns,
                ls=nsls,
                c="0.3",
                lw=0.75,
                zorder=-1,
                label=f"{ns}-$\\sigma$ (Gauss. approx.)",
            )
    axes[1].set_ylabel(r"$\log(p)$")
    axes[1].grid(axis="y")
    axes[1].legend(loc="lower left", prop={"size": _plot_dist_fontsize})
    # Kernel scales
    output_scale, length_scales = gpr.scales
    scales_kwargs = {
        "verticalalignment": "center",
        "horizontalalignment": "right",
        "fontsize": _plot_dist_fontsize,
        "bbox": {
            "facecolor": "white",
            "alpha": 0.5,
        },
    }
    axes[1].text(
        0.965,
        0.12,
        f"Output scale: ${simple_latex_sci_notation(f'{output_scale:.2g}')}$",
        transform=axes[1].transAxes,
        **scales_kwargs,
    )
    # NEXT: parameters plots
    for i, p in enumerate(model.parameterization.sampled_params()):
        label = model.parameterization.labels()[p]
        ax = axes[i + 2]
        if gpr.infinities_classifier is not None and sum(y_finite) < len(X):
            ax.scatter(
                i_eval,
                X[:, i],
                marker="x",
                c=np.where(y_finite, None, 0.5),
                cmap="gray",
                vmin=0,
                vmax=1,
                s=20,
            )
        ax.scatter(
            i_eval,
            X[:, i],
            c=np.where(y_finite, y, np.inf),
            **kwargs_accepted,
        )
        bounds = (reference or {}).get(p)
        if bounds is not None:
            if len(bounds) == 5:
                ax.axhspan(
                    bounds[0], bounds[4], facecolor="tab:blue", alpha=0.2, zorder=-99
                )
                ax.axhspan(
                    bounds[1], bounds[3], facecolor="tab:blue", alpha=0.2, zorder=-99
                )
            ax.axhline(bounds[2], c="tab:blue", alpha=0.3, ls="--")
        ax.set_ylabel("$" + label + "$" if label != p else p)
        ax.grid(axis="y")
        # Length scales
        ax.text(
            0.965,
            0.12,
            f"Length scale: ${simple_latex_sci_notation(f'{length_scales[i]:.2g}')}$",
            transform=ax.transAxes,
            **scales_kwargs,
        )
    # Format common x-axis
    axes[0].set_xlim(0, len(X) + 0.5)
    axes[-1].set_xlabel("Number of posterior evaluations")
    n_train = progress.data["n_total"][1]
    for ax in axes:
        ax.axvspan(0, n_train + 0.5, facecolor="0.85", zorder=-999)
        for n_iteration in progress.data["n_total"][1:]:
            ax.axvline(n_iteration + 0.5, ls="--", c="0.75", lw=0.75, zorder=-9)
    # TODO: make sure the x ticks are int


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
        cmap = plt.get_cmap("Spectral")
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
    # cls = [credibility_of_nstd(s, 1) for s in [1, 2, 3, 4]]  # use 1d cl's as reference
    nstds = [1, 2, 3, 4]
    linestyles = ["-", "--", "-.", ":"]
    for nstd, ls in zip(nstds, linestyles):
        std_of_cl = nstd_of_1d_nstd(nstd, dim)
        if std_of_cl < max(radial_distances):
            ax.axvline(
                std_of_cl,
                c="0.75",
                ls=ls,
                zorder=-99,
                label=f"${100 * credibility_of_nstd(std_of_cl, dim):.2f}\\%$ prob mass",
            )
    ax.set_ylabel(f"{num_or_dens} of points")
    ax.set_xlabel("Number of standard deviations")
    ax.legend(loc="upper right")
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
    cmap = [plt.get_cmap("magma"), plt.get_cmap("viridis")]
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
        ax[i].contourf(X, Y, Z, levels, cmap=plt.get_cmap(cmap[i], 256), norm=norm)
        points = ax[i].scatter(
            *gpr.X_train.T, edgecolors="deepskyblue", marker=r"$\bigcirc$"
        )
        # Plot position of next best sample
        point_max = ax[i].scatter(*acq_max, marker="x", color="k")
        if last_points is not None:
            points_last = ax[i].scatter(
                *last_points.T, edgecolors="violet", marker=r"$\bigcirc$"
            )
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
    fig.legend(
        list(legend_labels), list(legend_labels.values()), loc="lower center", ncol=99
    )
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
    cmap = [plt.get_cmap("magma"), plt.get_cmap("viridis")]
    label = ["Model mean (log-posterior)", "Acquisition function value"]
    for i, Z in enumerate([model_mean, acq_value]):
        ax[i].set_title(label[i])
        # Boost the upper limit to avoid truncation errors.
        Z_finite = Z[np.isfinite(Z)]
        # Z_clipped = np.clip(Z_finite, min(Z[np.isfinite(Z)]), max(Z[np.isfinite(Z)]))
        # Z_sort = np.sort(Z_finite)[::-1]
        top_x_perc = np.sort(Z_finite)[::-1][: int(len(Z_finite) * 0.5)]
        relevant_range = max(top_x_perc) - min(top_x_perc)
        levels = np.linspace(
            max(Z_finite) - 1.99 * relevant_range,
            max(Z_finite) + 0.01 * relevant_range,
            500,
        )
        Z[np.isfinite(Z)] = np.clip(Z_finite, min(levels), max(levels))
        Z = Z.reshape(*X.shape)
        norm = cm.colors.Normalize(vmax=max(levels), vmin=min(levels))
        ax[i].set_facecolor("grey")
        # # Background of the same color as the bottom of the colormap, to avoid "gaps"
        # plt.gca().set_facecolor(cmap[i].colors[0])
        ax[i].contourf(X, Y, Z, levels, cmap=plt.get_cmap(cmap[i], 256), norm=norm)
        points = ax[i].scatter(
            *gpr.X_train.T, edgecolors="deepskyblue", marker=r"$\bigcirc$"
        )
        # Plot position of next best sample
        point_max = ax[i].scatter(*acq_max, marker="x", color="k")
        if last_points is not None:
            points_last = ax[i].scatter(
                *last_points.T, edgecolors="violet", marker=r"$\bigcirc$"
            )
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
    fig.legend(
        list(legend_labels), list(legend_labels.values()), loc="lower center", ncol=99
    )
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
    model_mean, model_std = gpr.predict(xx, return_std=True)
    # TODO: maybe change this one below if __call__ method added to GP_acquisition
    acq_value = acquisition(xx, gpr, eval_gradient=False)
    # maybe show the next max of acquisition
    acq_max = xx[np.argmax(acq_value)]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    cmap = [plt.get_cmap("magma"), plt.get_cmap("viridis"), plt.get_cmap("magma")]
    label = ["Model mean (log-posterior)", "Acquisition function value", "Model std dev."]
    for i, Z in enumerate([model_mean, acq_value]):
        ax[i].set_title(label[i])
        # Boost the upper limit to avoid truncation errors.
        Z_finite = Z[np.isfinite(Z)]
        # Z_clipped = np.clip(Z_finite, min(Z[np.isfinite(Z)]), max(Z[np.isfinite(Z)]))
        # Z_sort = np.sort(Z_finite)[::-1]
        top_x_perc = np.sort(Z_finite)[::-1][: int(len(Z_finite) * 0.5)]
        relevant_range = max(top_x_perc) - min(top_x_perc)
        levels = np.linspace(
            max(Z_finite) - 1.99 * relevant_range,
            max(Z_finite) + 0.01 * relevant_range,
            500,
        )
        Z[np.isfinite(Z)] = np.clip(Z_finite, min(levels), max(levels))
        Z = Z.reshape(*X.shape)
        norm = cm.colors.Normalize(vmax=max(levels), vmin=min(levels))
        ax[i].set_facecolor("grey")
        # # Background of the same color as the bottom of the colormap, to avoid "gaps"
        # plt.gca().set_facecolor(cmap[i].colors[0])
        ax[i].contourf(X, Y, Z, levels, cmap=cm.get_cmap(cmap[i], 256), norm=norm)
        points = ax[i].scatter(
            *gpr.X_train.T, edgecolors="deepskyblue", marker=r"$\bigcirc$"
        )
        # Plot position of next best sample
        point_max = ax[i].scatter(*acq_max, marker="x", color="k")
        if last_points is not None:
            points_last = ax[i].scatter(
                *last_points.T, edgecolors="violet", marker=r"$\bigcirc$"
            )
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
    zrange = max(Z_finite) - minz
    levels = np.linspace(minz, minz + (zrange if zrange > 0 else 0.00001), 500)
    # Z[np.isfinite(model_mean)] = np.clip(Z_finite, min(levels), max(levels))
    Z = Z.reshape(*X.shape)
    norm = cm.colors.Normalize(vmax=max(levels), vmin=min(levels))
    ax[2].set_facecolor("grey")
    ax[2].contourf(X, Y, Z, levels, cmap=plt.get_cmap(cmap[2], 256), norm=norm)
    points = ax[2].scatter(*gpr.X_train.T, edgecolors="deepskyblue", marker=r"$\bigcirc$")
    # Plot position of next best sample
    point_max = ax[2].scatter(*acq_max, marker="x", color="k")
    if last_points is not None:
        points_last = ax[2].scatter(
            *last_points.T, edgecolors="violet", marker=r"$\bigcirc$"
        )
    # Bounds
    ax[2].set_xlim(bounds[0][0], bounds[0][1])
    ax[2].set_ylim(bounds[1][0], bounds[1][1])
    legend_labels = {points: "Training points"}
    if last_points is not None:
        legend_labels[points_last] = "Points added in last iteration."
    legend_labels[point_max] = "Next optimal location"
    fig.legend(
        list(legend_labels), list(legend_labels.values()), loc="lower center", ncol=99
    )
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15)
