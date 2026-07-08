The callback function
=====================

In some cases it may be useful to interact with the active sampling loop while it is running. Remember that this loop goes through the following steps:

1. Acquire one or more optimal sampling locations
2. Evaluate the true posterior at the proposed locations.
3. Refit the surrogate model (and maybe its hyperparameters) with the new training points.
4. Evaluate the convergence criterion and stop the loop if passed.

You can define a **callback** function that will be called between the third and fourth step and pass it to the runner using the ``callback`` keyword. The only argument passed to it is the current :class:`~gpry.run.Runner` instance, which contains all the information about the running process.

Here is a simple example of a callback function printing some extra information:

.. code:: python

   def my_callback(runner):
       # Run this function only every 4 iterations
       if runner.current_iteration % 4:
           return
       print("Current iteration:", runner.current_iteration)
       print("Surrogate model:", runner.surrogate)
       print("Previous surrogate model:", runner.old_surrogate)
       print("Acquisition instance:", runner.acquisition)
       print("Convergence instance:", runner.convergence)
       print("Last appended points (regardless of finiteness):")
       print(runner.surrogate.X_last_appended)
       print("Last appended points to the GPR (finite):")
       print(runner.surrogate.X_last_appended_regress)
       print("Tabulated progress information:")
       print(runner.progress)

   run = Runner(
       ...
       callback=my_callback,
   )


MPI-aware callback function
---------------------------

If running multiple MPI processes, the callback function is by default only called by the main process. If you would like to incorporate MPI parallelization into your function, pass ``callback_is_MPI_aware=True`` to the :class:`~gpry.run.Runner`.

The following snippet shows a very simple example of an MPI-aware callback function:

.. code:: python

   import gpry.mpi as mpi

   def my_callback(runner):
       print(f"I am process {mpi.RANK} of {mpi.SIZE}")
       if mpi.is_main_process:
           print("This is only printed by the main process")
       # Share something from the main process to the rest
       if mpi.is_main_process:
           something = value  # some object to broadcast
       something = mpi.bcast(something if mpi.is_main_process else None)

   run = Runner(
       ...
       callback=my_callback,
       callback_is_MPI_aware=True,
   )
