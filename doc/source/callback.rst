==============================================================
Intermediate plotting, diagnostics etc.: The Callback function
==============================================================

In some cases it may be useful to interact with the active sampling loop which is
created internally when calling the :meth:`runner.run` function. This loop is run after
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

In the following there will be a few examples for callback functions which serve
different purposes which you could copy and modify to your specific application.

Be aware that the callback is always called from the main process unless the option
``callback_is_MPI_aware`` in the :class:`run.Runner` is set to ``True``. Then it is
called from all processes.

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

Of couse you could save the chains and plots anywhere you wish or where it would be
more convenient.
