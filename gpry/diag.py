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


def diagnosis(runner):
    if mpi.is_main_process:
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
        consistent_threshold = min_finite_y > runner.gpr.abs_threshold_finite
        print(
            f"THRESHOLD: {runner.gpr.abs_threshold_finite}. "
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
        # Get the "reference" MC sample from runner.reference, if set.
        reference = getattr(runner, "reference", None)
        # Points distribution and convergence criterion
        from gpry.plots import plot_points_distribution
        try:
            plot_points_distribution(
                runner.model, runner.gpr, runner.convergence,
                runner.progress,
                reference=reference)
        except ValueError as e:
            print(f"Could not plot points distributions (yet). Err msg: {e}")
        else:
            import matplotlib.pyplot as plt
            plt.savefig(
                os.path.join(
                    runner.plots_path,
                    f"points_dist_iteration_{runner.current_iteration:03d}.png"))
        # Current MC sample (if available)
        from gpry.gp_acquisition import NORA
        if isinstance(runner.acquisition, NORA):
            from getdist import MCSamples, plots
            from getdist.mcsamples import MCSamplesError
            mcsamples = MCSamples(
                samples=runner.acquisition.X_mc,
                weights=runner.acquisition.w_mc,
                names=runner.model.parameterization.sampled_params(),
                ranges=dict(zip(runner.model.parameterization.sampled_params(),
                                runner.model.prior.bounds())),
            )
            to_plot = [mcsamples]
            filled = True
            if reference is not None:
                to_plot = [reference] + to_plot
                filled = [False, True]
            g = plots.get_subplot_plotter()
            try:
                g.triangle_plot(to_plot, filled=filled)
                try:
                    from gpry.plots import getdist_add_training
                    getdist_add_training(g, runner.model, runner.gpr, highlight_last=True)
                except Exception as e:
                    raise
                    print(f"FAILED ADDING TRAINING POINTS! Error msg: {e}")
                import matplotlib.pyplot as plt
                plt.savefig(
                    os.path.join(
                        runner.plots_path,
                        f"NORA_iteration_{runner.current_iteration:03d}.png"))
            except (ValueError, IndexError, AttributeError, np.linalg.LinAlgError, MCSamplesError) as e:
                raise
                print(f"COULD NOT PLOT! Reason: {e}")
        print("**************************************************")
