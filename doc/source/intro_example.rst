Introductory example to using GPry
==================================

.. note::
   The code for this example is available at :download:`../../examples/introductory_example.ipynb` and :download:`../../examples/introductory_example.py`

Step 1: Setting up a likelihood function
----------------------------------------

Let's start with a very simple example where we want to characterize a 2d-Gaussian Likelihood:

.. math::
    y(x) \sim \mathcal{N}(x|\boldsymbol{\mu},\Sigma)

with

.. math::

   \boldsymbol{\mu}=\pmatrix{3\\ 2},\ \Sigma=\pmatrix{0.5 & 0.4 \\ 0.4 & 1.5}.

We need to define a **log-likelihood** function, which is the modelling target for GPry:

.. code:: python

    import numpy as np
    from scipy.stats import multivariate_normal

    mean = [3, 2]
    cov = [[0.5, 0.4], [0.4, 1.5]]
    rv = multivariate_normal(mean, cov)

    def logLkl(x, y):
        return rv.logpdf(np.array([x, y]).T)

with a uniform prior square in :math:`[-10, 10]`

.. code:: python

    bounds = [[-10, 10], [-10, 10]]


Step 2: Creating the Runner object
----------------------------------

The :class:`~run.Runner` manages model specification and the active sampling loop of GPry up to convergence, as well as allows for some post-processing and tests.

To initialise it, we pass it the log-likelihood function as first argument, and the prior bounds via the ``bounds`` keyword. More complicated prior specifications can be used by defining and passing as first argument a `Cobaya model <https://cobaya.readthedocs.io/en/latest/models.html>`_ (see :ref:`running_cobaya`).

Optionally, we will also pass a path to save checkpoints via the ``checkpoint`` argument. If passed, in order to prevent loss of data, you **must** decide a checkpoint policy (either ``"resume"`` or ``"overwrite"``). If set to ``"resume"`` the runner object will try to load the checkpoint and resume the active sampling loop from there; if set to ``"overwrite"`` it will start from scratch and overwrite checkpoint files which already exist.

.. code:: python

    from gpry.run import Runner
    checkpoint = "output/simple"
    runner = Runner(logLkl, bounds, checkpoint=checkpoint, load_checkpoint="overwrite")

In this example we will leave all training parameters (the choice of GP,
acquisition function, convergence criterion and options of the active sampling loop) as default.


Step 3: Running the active learning loop
----------------------------------------

Since all training parameters are chosen automatically all we have to do is to call the
:meth:`~run.Runner.run` method of the :class:`~Runner` object:

.. code:: python

    runner.run()

This will run the active sampling loop until convergence is reached. It also saves
the checkpoint files after every iteration of the bayesian optimization loop and creates
progress plots which are saved in ``[checkpoint]/images/`` (or ``./images/`` if checkpoint is
None).

Once converged, you can access the surrogate model and use it as a function for any purpose.

.. note::
   Internally GPry models the **log-posterior**, not the log-likelihood.

To get the surrogate log-posterior or log-likelihood you can call respectively :meth:`~run.Runner.logp` or :meth:`~run.Runner.logL`, passing each a single ``(nsamples, ndims)`` array with the locations where you want to evaluate the surrogate.

Let us compare GPry and the likelihood at `(1, 2)`:

.. code:: python

   point = (1, 2)
   print(f"Log-lkl at (1,2): {logLkl(*point)}")
   print(f"surrogate at (1,2): {runner.logL(point)[0]}")

Both evaluations should produce similar numbers.


Step 4: Monte Carlo samples from the final surrogate model
----------------------------------------------------------

The :class:`~run.Runner` object can also run an MC sampler on the GP in order to extract marginalised quantities. To do that, we use the :meth:`~run.Runner.generate_mc_sample` method of the :class:`~run.Runner`.

By default, GPry would already have run an MC sampler at the end of the main loop, for diagnosis purposes. You can get the result using :meth:`~run.Runner.last_mc_samples`, which returns a dictionary containing the samples' parameter values, weight (``None`` if all samples carry equal weight), and values for the surrogate log-posterior, true log- prior, and surrogate log-likelihood (i.e., the surrogate log-posterior minus the analytic log-prior):

.. code:: python

   mc_samples_dict = runner.last_mc_samples(as_pandas=True)
   print(mc_samples_dict)

.. code::

          w    logpost  logprior   loglike       x_1       x_2
   0    1.0 -11.598237 -5.991465 -5.606773  4.896665  4.535424
   1    1.0 -11.286758 -5.991465 -5.295293  1.117008 -0.148755
   2    1.0 -11.262597 -5.991465 -5.271132  4.806402  4.460790
   3    1.0 -10.672167 -5.991465 -4.680702  4.313618  1.246258
   4    1.0 -10.670824 -5.991465 -4.679360  2.068655 -1.042577
   ..   ...        ...       ...       ...       ...       ...
   237  1.0  -7.570368 -5.991465 -1.578904  3.024982  2.123483
   238  1.0  -7.570094 -5.991465 -1.578629  3.067862  2.078437
   239  1.0  -7.569896 -5.991465 -1.578432  3.056808  1.987092
   240  1.0  -7.565576 -5.991465 -1.574112  2.979181  1.981701
   241  1.0  -7.565178 -5.991465 -1.573713  2.996010  1.998560

   [242 rows x 6 columns]

Samples are also stored by default in the same folder as the checkpoint, inside a ``chains`` sub folder. The order of the columns in that file are ``weight log-posterior param_1 param_2 ...``.

Subsequent calls to the :meth:`~run.Runner.generate_mc_sample` method can be used to re-generate MC samples from the surrogate posterior, e.g. if a finer representation is needed.  More details can be found in the :doc:`mc_samples` section of the documentation.


Plotting the results
--------------------

Now that we have MC samples you can process and plot them the same way that you would do with any other MC samples.

The easiest way to get a corner plot though is to call the :meth:`~run.Runner.plot_mc` method of the :class:`~run.Runner` object which will generate a corner plot.

It includes the training set unless passed ``add_training=False``.

.. code:: python

   runner.plot_mc()

.. image:: images/simple_surrogate_triangle.svg
   :width: 450
   :align: center

Bonus: Getting some extra insights
----------------------------------

You can do further plots about the progress of the active-learning loop using:

.. code:: python

   runner.plot_progress()

If you call this method without any arguments it results in the following plots:

* a histogram of the distribution time spent at different parts of the code (`timing.png`)
* the distribution of the training samples (`trace.png`)
* A plot showing the value(s) of all convergence criteria as function of the number of
  posterior evaluations (`convergence.png`). The upper part of this plot shows the
  convergence criterion, the second from the top the distribution of posterior values over
  time, and the rest of them the distribution of samples per model parameter. The blue bands
  in these parameter plots represent the 1-d marginalised posterior obtained with the
  MC sampler, and won't appear if :meth:`~run.Runner.plot_progress` is called before generating
  an MC sample. If the training points were not centred around the blue band, the run has not
  converged correctly. In this case, see :ref:`strategy-troubleshooting` for tips on fixing this issue.


.. image:: images/simple_timing.svg
   :width: 370

.. image:: images/simple_convergence.svg
   :width: 370

.. image:: images/simple_trace.svg
   :width: 500
   :align: center


Validation
----------

.. note::
    This part is optional and only relevant for validating the contours that GPry produces. In a realistic scenario you would obviously not run a full MCMC on the likelihood and will need to follow the validation guidelines at :ref:`strategy-troubleshooting`.

As explained :ref:`here <help_reference>`, we can easily compare our results to samples from the original gaussian by setting them as *fiducial samples*:

.. code:: python

   truth_samples = rv.rvs(size=10000)
   runner.set_fiducial_MC(truth_samples)

   runner.plot_mc()

.. image:: images/simple_comparison_triangle.svg
  :width: 450
  :align: center

As you can see the two agree almost perfectly! And we achieved this with just a few evaluations of the posterior distribution!
