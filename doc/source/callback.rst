==============================================================
Intermediate plotting, diagnostics etc.: The Callback function
==============================================================

In some cases it may be useful to interact with the active sampling loop which is
created internally when calling the :meth:`run.Runner.run` function. This loop is run after
having drawn an initial set of samples and each iteration consists of
four steps:

#. Acquire :math:`m=` *number of Kriging believer steps* sampling locations by optimizing
   the acquisition function.
#. Evaluate the true posterior at the proposed locations.
#. Add the new samples to the training set of the GP and optimize it's hyperparamters.
#. Evaluate the convergence criterion. If it is below the set threshold or the
   computational budget (max. number of posterior evaluations) has been exhausted stop the
   loop.

The ``callback`` function allows you to interact with this loop while it is running. It
is called between the third and fourth step and any method which takes a
:class:`run.Runner` instance is a valid callback function.

In the following there will be a few examples for ``callback`` functions which serve
different purposes which you could copy and modify to your specific application.

Printing extra information
==========================

Although the level of verbosity can be controlled with the ``verbose`` variable of the
:class:`Runner <run.Runner>` instance there might be cases in which you would want a
custom output. This is a simple example exposing some of the information that's available
within the :class:`Runner <run.Runner>` class::

    def callback(runner):
        print("Current iteration:", runner.current_iteration)
        print("True model:", runner.model)
        print("GPR:", runner.gpr)
        print("Previous GPR:", runner.old_gpr)
        print("Acquisition instance:", runner.acquisition)
        print("Convergence instance:", runner.convergence)
        print("Progress instance:", runner.progress)
        print("Last appended points (regardless of finiteness):")
        print(runner.gpr.last_appended)
        print("Last appended points to the GPR (finite):")
        print(runner.gpr.last_appended_finite)

.. note::

    Often information (such as convergence history, values of hyperparameters etc.) is
    stored in the sub-modules of the :class:`Runner <run.Runner>` instance. To access
    those please refer to the documentation of the modules.

Intermediate MCMCs
==================

This callback function runs an MCMC sampler at intermediate steps in the bayesian
optimization loop (in this case every 10 steps starting from 20 min. posterior samples
and going until 100) and saves the chains and a triangle plot of the run in the same
directory as where the checkpoint is saved::

    i = 0
    def callback(runner, steps=np.arange(20, 100, 10)):
        n_total = runner.gpr.n_total
        checkpoint_location = runner.checkpoint

        if n_total >= steps[i]:
            surr_info, sampler = runner.generate_mc_sample(
                output=os.path.join(checkpoint_location, n_total))
            runner.plot_mc(surr_info, sampler,
                           output=os.path.join(checkpoint_location, f"{n_total}.pdf"))
            i += 1

Of couse you could save the chains and plots anywhere you wish or where it would be
more convenient.

MPI-aware callback function
===========================

The callback is always called from the main process unless the option
``callback_is_MPI_aware`` in the :class:`Runner <run.Runner>` is set to ``True``. The
following snippet shows a very simple example of how to make a simple MPI-aware callback
function::

    from gpry.mpi import mpi_comm, mpi_size, mpi_rank, is_main_process
    callback_is_MPI_aware = True
    def callback(runner):
        print(f"I am process {mpi_rank} of {mpi_size}")
        if is_main_process:
            print("This is only printed by the main process")
       # share something from the main process to the rest
       if is_main_process:
            something = value
       something = mpi.bcast(something if is_main_process else None)
