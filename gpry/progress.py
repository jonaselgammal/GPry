import time
import numpy as np
import pandas as pd
from gpry.mpi import is_main_process, multiple_processes, mpi_comm, sync_processes


class Progress:
    """
    Pandas DataFrame to store progress, timing, numbers of evaluations, etc.
    A progress instance is created by the :class:`run.Runner` object and internally
    populated when calling the :meth:`run.Runner.run` function.
    """

    _colnames = {
        "n_total": "number of training points at the start of the iteration",
        "n_finite": ("number of finite-posterior training points "
                       "at the start of the iteration"),
        "time_acquire": "time needed to acquire candidates for truth evaluation",
        "evals_acquire": ("number of evaluations of the GP needed to acquire candidates "
                          "for truth evaluation"),
        "time_truth": "time needed to evaluate the true model at the candidate points",
        "evals_truth": "number of evaluations of the true model",
        "time_fit": "time of refitting of the GP model after adding new training points",
        "evals_fit": ("number of evaluations of the GP during refitting after adding new"
                      "training points"),
        "time_convergence": "time needed to compute the convergence criterion",
        "evals_convergence": ("number of evaluations of the GP needed to compute the "
                              "convergence criterion"),
        "convergence_crit_value": "value of the convergence criterion"}
    _dtypes = {col: (int if col.split("_")[0].lower() in ["n", "evals"]
                     else float)
               for col in _colnames}

    def __init__(self):
        """Initialises Progress table."""
        self.data = pd.DataFrame(columns=list(self._colnames))

    def __repr__(self):
        return self.data.__repr__()

    def help_column_names(self):
        """Prints names and description of columns."""
        print(self._colnames)

    def add_iteration(self):
        """
        Adds the next row to the table. New values will be added to this row.
        """
        self.data = pd.concat(
            [self.data, pd.DataFrame({c: [np.nan] for c in self.data.columns})],
            axis=0, ignore_index=True)

    def add_current_n_truth(self, n_truth, n_truth_finite):
        """
        Adds the number of total and finite evaluations of the true model
        at the beginning of the iteration.
        """
        self.data.iloc[-1, self.data.columns.get_loc("n_total")] = n_truth
        self.data.iloc[-1, self.data.columns.get_loc("n_finite")] = n_truth_finite

    def add_acquisition(self, time, evals):
        """Adds timing and #evals during acquisitions."""
        self.data.iloc[-1, self.data.columns.get_loc("time_acquire")] = time
        self.data.iloc[-1, self.data.columns.get_loc("evals_acquire")] = evals

    def add_truth(self, time, evals):
        """Adds timing and #evals during truth evaluations."""
        self.data.iloc[-1, self.data.columns.get_loc("time_truth")] = time
        self.data.iloc[-1, self.data.columns.get_loc("evals_truth")] = evals

    def add_fit(self, time, evals):
        """Adds timing and #evals during GP fitting."""
        self.data.iloc[-1, self.data.columns.get_loc("time_fit")] = time
        self.data.iloc[-1, self.data.columns.get_loc("evals_fit")] = evals

    def add_convergence(self, time, evals, crit_value):
        """
        Adds timing and #evals during convergence computation, together with the new
        criterion value.
        """
        self.data.iloc[-1, self.data.columns.get_loc("time_convergence")] = time
        self.data.iloc[-1, self.data.columns.get_loc("evals_convergence")] = evals
        self.data.iloc[-1, \
                       self.data.columns.get_loc("convergence_crit_value")] = crit_value

    def mpi_sync(self):
        """
        When running in parallel, synchronises all individual instances by taking the
        maximum times and numbers of GP evaluations where each process run an independent
        step.

        The number of truth evaluations in the present iteration is the individual process
        one, instead of the total number of new evaluations, in order to be consistent
        with the reported evaluation time.
        """
        if not multiple_processes:
            return
        # For the number of evaluations, not sure summing them is very helpful.
        # Maybe keep all of them so that the can be plotted per item in slightly different
        # colours for each process?
        self.bcast_last_max("time_acquire")
        self.bcast_sum("evals_acquire")
        self.bcast_last_max("time_truth")
        self.bcast_sum("evals_truth")
        self.bcast_last_max("time_fit")
        self.bcast_sum("evals_fit")
        self.bcast_last_max("time_convergence")
        self.bcast_sum("evals_convergence")
        self.bcast_last_max("convergence_crit_value")  # prob not needed
        sync_processes()

    def bcast_last_max(self, column):
        """
        Sets the last row value of a column to the max of all MPI processes.

        If only one defined (the rest are nan's), takes it.
        """
        self._bcast_operation(column, "max")

    def bcast_sum(self, column):
        """
        Sets the last row value of a column to the sum over all MPI processes.

        If only one defined (the rest are nan's), takes it.
        """
        self._bcast_operation(column, "sum")

    def _bcast_operation(self, column, operation):
        f = {"max": max, "sum": sum}[operation.lower()]
        all_values = np.array(mpi_comm.gather(
            self.data.iloc[-1, self.data.columns.get_loc(column)]))
        max_value = None
        if is_main_process:
            all_finite_values = all_values[np.isfinite(all_values)]
            max_value = f(all_finite_values) if len(all_finite_values) else np.nan
        self.data.iloc[-1, self.data.columns.get_loc(column)] = mpi_comm.bcast(max_value)

    def _x_ticks_for_bar_plot(self, fig, ax):
        fig.canvas.draw()
        xticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        n_xticks = len(xticks)
        xticks = xticks[::max(1, int(n_xticks/10.))]
        labels = labels[::max(1, int(n_xticks/10.))]
        ax.set_xticks(xticks, labels=labels)

    def plot_timing(self, truth=True, show=False, save="progress_timing.png"):
        """
        Plots as stacked bars the timing of each part of each iteration.

        In multiprocess runs, max of the time taken per step.

        Pass ``truth=False`` (default: True) to exclude the computation time of the true
        posterior at training points, for e.g. overhead-only plots.
        """
        import matplotlib.pyplot as plt
        plt.set_loglevel('WARNING')  # avoids a useless message
        fig, ax = plt.subplots()
        # cast x values into list, to prevent finer x ticks
        iters = [str(i) for i in self.data.index.to_numpy(int)]
        bottom = np.zeros(len(self.data.index))
        for col, label in {
                "time_acquire": "Acquisition",
                "time_truth": "Truth",
                "time_fit": "GP fit",
                "time_convergence": "Convergence crit."}.items():
            if not truth and col == "time_truth":
                continue
            dtype = self._dtypes[col]
            ax.bar(iters, self.data[col].astype(dtype), label=label, bottom=bottom)
            bottom += self.data[col].to_numpy(dtype=dtype)
        plt.xlabel("Iteration")
        plt.draw()
        self._x_ticks_for_bar_plot(fig, ax)
        multiprocess_str = " (max over processes)" if multiple_processes else ""
        plt.ylabel("Time (s)" + multiprocess_str)
        plt.legend()
        if save:
            plt.savefig(save)
        if show:
            plt.show(block=True)
        plt.close()

    def plot_evals(self, show=False, save="progress_evals.png"):
        """
        Plots as stacked bars the number of evaluations of each part of each iteration.

        In multiprocess runs, sum of the number of evaluations of all processes.

        Pass ``truth=False`` (default: True) to exclude the number of evaluations of the
        true posterior at training points, for e.g. overhead-only plots.
        """
        import matplotlib.pyplot as plt
        plt.set_loglevel('WARNING')  # avoids a useless message
        fig, ax = plt.subplots()
        # cast x values into list, to prevent finer x ticks
        iters = [str(i) for i in self.data.index.to_numpy(int)]
        bottom = np.zeros(len(self.data.index))
        for col, label in {
                "evals_acquire": "Acquisition",
                "evals_fit": "GP fit",
                "evals_convergence": "Convergence crit."}.items():
            dtype = self._dtypes[col]
            ax.bar(iters, self.data[col].astype(dtype), label=label, bottom=bottom)
            bottom += self.data[col].to_numpy(dtype=dtype)
        ax.set_xlabel("Iteration")
        self._x_ticks_for_bar_plot(fig, ax)
        multiprocess_str = " (summed over processes)" if multiple_processes else ""
        plt.ylabel("Number of evaluations" + multiprocess_str)
        plt.legend()
        if save:
            plt.savefig(save)
        if show:
            plt.show(block=True)
        plt.close()


class Timer:
    """Class for timing code within ``with`` block."""

    def __enter__(self):
        """Saves initial wallclock time."""
        self.start = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        """Saves final wallclock time and difference."""
        self.end = time.time()
        self.time = self.end - self.start


class TimerCounter(Timer):
    """
    Class for timing code within ``with`` block, and count number of evaluations of a
    given GP model.
    """

    def __init__(self, *gps):
        """Takes the GP's whose evaluations will be counted."""
        self.gps = gps  # save references for use at exit

    def __enter__(self):
        """Saves initial wallclock time and number of evaluations."""
        super().__enter__()
        self.init_eval = np.array([gp.n_eval for gp in self.gps], dtype=int)
        self.init_eval_loglike = np.array(
            [gp.n_eval_loglike for gp in self.gps], dtype=int)
        return self

    def __exit__(self, *args, **kwargs):
        """Saves final wallclock time and number of evaluations, and their differences."""
        super().__exit__()
        self.final_eval = np.array([gp.n_eval for gp in self.gps], dtype=int)
        self.evals = sum(self.final_eval - self.init_eval)
        self.final_eval_loglike = np.array(
            [gp.n_eval_loglike for gp in self.gps], dtype=int)
        self.evals_loglike = sum(self.final_eval_loglike - self.init_eval_loglike)
