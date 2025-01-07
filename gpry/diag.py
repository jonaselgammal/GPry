"""
Diagnosis-focused callback function. Runs a MC sampler at every call, and prints several
relevant quantities.

This function is MPI-aware, and takes advantage of parallelisation for MC sampling.
"""

import os
import warnings

import pandas as pd
import numpy as np

from gpry import mpi

pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 500)
pd.set_option("expand_frame_repr", False)


do_check_inf_classifier = True
do_plot_mc = True
do_plot_slices = True


def diagnosis(runner):
    if not mpi.is_main_process:
        return
    if do_check_inf_classifier and runner.gpr.infinities_classifier:
        print("**************************************************")
        print("Traning set (full):")
        points = pd.DataFrame(
            {f"x_{i + 1}": X for i, X in enumerate(runner.gpr.X_train_all.T)}
        )
        points = pd.DataFrame(dict(zip(
            runner.model.parameterization.sampled_params(), runner.gpr.X_train_all.T
        )))
        points["y_GP"] = runner.gpr.y_train_all
        points["GP"] = [point in runner.gpr.X_train for point in runner.gpr.X_train_all]
        y_finite = runner.gpr.infinities_classifier.y_finite
        print(points)
        print(
            f"TRAINING POINTS: {len(points)} TOTAL of which {sum(points['GP'])} FINITE"
        )
        # TESTS and other data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            consistent_is_finite = y_finite == runner.gpr.is_finite(points["y_GP"])
        consistent_predict = (
            y_finite == runner.gpr.predict_is_finite(runner.gpr.X_train_all)
        )
        min_finite_y = min(points[points["GP"]]["y_GP"])
        consistent_threshold = min_finite_y > runner.gpr.infinities_classifier.abs_threshold_finite
        print(
            f"THRESHOLD: {runner.gpr.infinities_classifier.abs_threshold_finite}. "
            f"Min finite y is {min_finite_y}"
        )
        is_inf_self_consistent = (
            all(consistent_is_finite) & all(consistent_predict) & consistent_threshold
        )
        print(
            "TEST: is the infinities classifier self consistent?", is_inf_self_consistent
        )
        if not is_inf_self_consistent:
            print("    SUBTEST: method is_finite consistent:", all(consistent_is_finite))
            if not all(consistent_is_finite):
                bad_i = [i for i, val in enumerate(consistent_is_finite) if not val]
                print("        Bad points:", bad_i)
                print("        y values:", points["y_GP"][bad_i].to_numpy(dtype=float))
                print("        is_finite:", runner.gpr.is_finite(points["y_GP"][bad_i]))
            print("    SUBTEST: method predict consistent:", all(consistent_predict))
            if not all(consistent_predict):
                bad_i = [i for i, val in enumerate(consistent_predict) if not val]
                print("        Bad points:", bad_i)
                print("        y values:", points["y_GP"][bad_i].to_numpy(dtype=float))
                print(
                    "        predict:",
                    runner.gpr.predict_is_finite(runner.gpr.X_train_all[bad_i])
                )
            print("    SUBTEST: threshold consistent:", consistent_threshold)
        all_finite_from_full_in_GP = all(
            [point in runner.gpr.X_train
             for point in runner.gpr.X_train_all[runner.gpr.is_finite()]
             ]
        )
        same_length_finite_and_GP = sum(points["GP"]) == len(runner.gpr.y_train)
        is_train_consistent = all_finite_from_full_in_GP & same_length_finite_and_GP
        print("TEST: are the full and GP training sets consistent?", is_train_consistent)
        if not is_train_consistent:
            print("    SUBTEST: are finite points missing from GP?", all_finite_from_full_in_GP)
            print("    SUBTEST: are there more points in GP and finite?", same_length_finite_and_GP)

    # PLOTS ##########################################################################
    # Create the plots path, just in case it want not there yet
    from gpry.io import create_path
    import matplotlib.pyplot as plt
    create_path(runner.plots_path)
    # Get the "reference" MC sample from runner.reference, if set.
    reference = getattr(runner, "reference", None)
    fiducial = getattr(runner, "fiducial", None)

    # Plot points distribution and convergence criterion
    from gpry.plots import \
        plot_points_distribution, plot_slices, plot_slices_reference, \
        getdist_add_training, _plot_2d_model_acquisition_std
    # Do not plot if sample reweighted, to save time
    if not runner.acquisition.is_last_MC_reweighted:
        try:
            plot_points_distribution(
                runner.model, runner.gpr, runner.convergence,
                runner.progress,
                reference=reference)
        except ValueError as e:
            print(f"Could not plot points distributions (yet). Err msg: {e}")
        else:
            plt.savefig(os.path.join(runner.plots_path, "points_dist.svg"))
        plt.close()

    # Plot mean GP and acq func slices
    if do_plot_slices:
        plot_slices(
            runner.model, runner.gpr, runner.acquisition, reference=reference
        )
        plt.savefig(os.path.join(
            runner.plots_path,
            f"slices_iteration_{runner.current_iteration:03d}.png")
        )
        plt.close()
        try:
            plot_slices_reference(
                runner.model, runner.gpr, fiducial, truth=True, reference=reference,
            )
            plt.savefig(os.path.join(
                runner.plots_path,
                f"comparison_slices_iteration_{runner.current_iteration:03d}.png")
            )
        except:
            raise
            pass
        plt.close()

    # Plot current MC sample (if available)
    from gpry.gp_acquisition import NORA
    if (
        do_plot_mc and isinstance(runner.acquisition, NORA) and
        not runner.acquisition.is_last_MC_reweighted
    ):
        from getdist import plots
        from getdist.mcsamples import MCSamplesError
        from getdist.densities import DensitiesError
        mcsamples = runner.acquisition.last_MC_sample_getdist(runner.model)
        to_plot = [mcsamples]
        to_plot_params = list(runner.model.parameterization.sampled_params())
        name_logp, label_logp = "logpost", r"\log(p)"
        if mcsamples.loglikes is not None:
            to_plot_params += [name_logp]
        if reference is not None:
            if reference.loglikes is not None:
                try:
                    # PROBLEM: loglikes will be -chi2 if obtained with PolyChord and
                    # not corrected, but we cannot assume anything here: it may have
                    # been corrected already
                    reference.addDerived(
                        -reference.loglikes, name_logp, label=label_logp,
                         range=(-np.inf, max(-reference.loglikes)),
                    )
                except ValueError:  # already added
                    pass
            else:
                # Limitation: if logp not present in reference MCSamples (rare), can
                # not plot for GP either (1st sample needs to contain all params)
                if to_plot_params[-1] == name_logp:
                    to_plot_params = to_plot_params[:-1]
        filled = [True]
        legend_labels = [f"Last NORA sample ({len(runner.gpr.X_train_all)} evals.)"]
        if reference is not None:
            to_plot = [reference] + to_plot
            filled = [False] + filled
            legend_labels += ["Reference sample"]
        g = plots.get_subplot_plotter(subplot_size=2)
        try:
            g.triangle_plot(
                to_plot, to_plot_params, filled=filled, legend_labels=legend_labels
            )
            try:
                getdist_add_training(g, runner.model, runner.gpr, highlight_last=True)
            except Exception as e:
                raise
                print(f"FAILED ADDING TRAINING POINTS! Error msg: {e}")
            plt.savefig(
                os.path.join(
                    runner.plots_path,
                    f"NORA_iteration_{runner.current_iteration:03d}.png"))
            plt.close()
        except (ValueError, IndexError, AttributeError, np.linalg.LinAlgError, MCSamplesError, DensitiesError) as e:
            print(f"COULD NOT DO TRIANGLE PLOT! Reason: {e}")

        if runner.model.prior.d() == 2:
            try:
                _plot_2d_model_acquisition_std(
                    runner.gpr, runner.acquisition, last_points=None, res=200
                )
                plt.savefig(
                    os.path.join(
                        runner.plots_path,
                        f"contours_iteration_{runner.current_iteration:03d}.png"))
                plt.close()
            except ValueError as e:
                print(f"COULD NOT PLOT CONTOURS! Reason: {e}")

    print("**************************************************")
