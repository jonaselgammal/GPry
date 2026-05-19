Cobaya interface for GPRy
=========================

This class implements an interface for the Bayesian inference code `Cobaya <https://cobaya.readthedocs.io>`_, by inheriting from the ``cobaya.sampler.Sampler`` class.

To use GPry as a Cobaya sampler, once GPry has been installed with pip, mention it in the sampler block simply as ``gpry``:

.. code:: yaml

   likelihood:
     gaussian_mixture:
       means: [0.2, 0]
       covs: [[0.1, 0.05], [0.05,0.2]]

   params:
     a:
       prior: [-0.5, 3]
       latex: \alpha
     b:
       prior:
         dist: norm
         loc: 0
         scale: 1
       ref: 0
       latex: \beta

   sampler:
     gpry:
       # option: value

Inside the ``gpry`` block you can specify options for the different GPry modules similarly to how you would pass them to a :class:`~run.Runner` object at initialization.


.. code:: yaml

   # Options regarding the Bayesian optimization loop.
   # NB: 'd' after a number means the dimensionality of the sampling space,
   #     and a number following 'd' means the power of the dimensionality factor.

   # General options for the main loop
   options:
     # Number of finite initial truth evaluations before starting the learning loop
     n_initial: 3d
     # Maximum number of truth evaluations at initialization. If it is reached before
     # `n_initial` finite points have been found, the run will fail. To avoid that, try
     # decreasing the prior volume.
     max_initial: 30d1.5
     # Maximum number of truth evaluations before the run stops. This is useful for e.g.
     # restricting the maximum computation resources.
     max_total: 70d1.5
     # Maximum number of sampling points accepted into the GP training set before the run
     # stops. If this limit is frequently saturated, try decreasing the prior volume.
     max_finite:  # default (undefined) = max_total
     # Number of points which are acquired with Kriging believer for every acquisition step.
     # Gets adjusted (with 20% tol.) to match a multiple of the num. of parallel processes.
     n_points_per_acq: d
     # Number of iterations between full GP hyperparameters fits, including several
     # restarts of the optimizer. If 'np.inf' or a large number, never refit with restarts.
     fit_full_every: 2
     # Similar to `fit_full_every`, but with a single optimizer run from the last optimum.
     # Overridden by `fit_full_every` if it applies. Pass np.inf or a large number to never
     # refit from last optimum (hyperparameters kept constant in that iteration).
     fit_simple_every: 1  # every iteration

   # The surrogate GP regressor used for interpolating the posterior
   surrogate:
     regressor:
       # Spatial correlation kernel and params, e.g. RBF, Matern, {Matern: {nu: 2.5}}, ...
       kernel: RBF
       # Priors for the output and length scale, in normalised logp units
       output_scale_prior: [1e-2, 1e3]
       length_scale_prior: [1e-3, 1e1]
       # Noise level in logp units; increase for numerically noisy likelihoods
       noise_level: 1e-1
       # Whether to fix the noise or to fit for a white-noise additive kernel term
       noise_fixed: False
       # Hyperparameter fitting: optimizer (from scipy) and number of restarts for full fits
       optimizer: fmin_l_bfgs_b
       # Number of restarts of the hyperparameter optimizer
       n_restarts_optimizer: 20d
     # Treatment of infinities and large negative values; False for no classifier
     infinities_classifier:
       svm:
         # Difference in standard deviations ('s') for considering a value as -infinity
         threshold: 20s
       trust_region:
         # Difference in standard deviations ('s') for considering a value as -infinity
         threshold: 30s
         # Factor by which to multiply the sides of the hyperrectangular trust region
         factor: 2  # none for 1
     # Factor used to clip the GPR from above, to avoid overshoots (undefined to disable)
     clip_factor: 1.1  # none: no clipping
     # Verbosity (set only if different from the overall verbosity)
     verbose:

   # The acquisition class, function and their options
   gp_acquisition:
     # Acquisition function and its arguments (if passed as dict)
     acq_func:
       LogExp:
         zeta_scaling: 0.85  # larger z_scaling for more exploratory acquisition
     # Verbosity (set only if different from the overall verbosity)
     verbose:
     # Acquisition engine: NORA or BatchOptimizer
     engine: NORA
     # Options for the engine
     options_NORA:
       # nested sampler used for acquisition
       sampler:  # undefined: in order, of available: polychord > ultranest > nessai
       mc_every: 2  # number of iterations between full NS runs
       nlive_per_training: 3  # number of live points per training sample
       nlive_max: 25d  # cap for the number of live points
       num_repeats: 5d  # number of steps of slice chains (polychord only)
       precision_criterion_target: 0.01  # precision criterion for the NS
       nprior_per_nlive: 10  # number of prior points in the initial sample, times nlive
       max_ncalls:  # maximum number of calls to the GPR model during NS (none: infinite)
     options_BatchOptimizer:
       proposer:  # default (undefined): a mixture of uniform and centroids
       acq_optimizer: fmin_l_bfgs_b  # scipy optimizer to use
       n_restarts_optimizer: 5d  # number of restarts during hyperparameter fitting
       n_repeats_propose: 10  # number of starting points drawn from the proposer

   # Proposer used for drawing the initial training samples before running
   # the acquisition loop. One of [reference, prior, uniform].
   # Can be specified as dict with args, e.g. {reference: {max_tries: 1000}}
   initial_proposer: reference

   # Convergence criterion.
   # (add or replace by DontConverge to run until evaluation budget exhausted)
   # `policy` can be [n]ecessary (default), [s]ufficient, both ([ns]), or [m]onitoring
   convergence_criterion:
     CorrectCounter: {policy: s}
     GaussianKL: {policy: n, limit: 5e-02}  # ignored if NORA not used
     TrainAlignment: {policy: n}  # ignored if NORA not used
   # TODO: if we control a bit better the number of points for crrectcoutner, maybe we can make correctcounter and gaussiankl each sufficient, not necessary

   # Sampler used to generate intermediate and final samples from the surrogate model
   mc_options: nested  # pass as a dict for options

   # Produce progress plots (inside the gpry_output dir).
   # One can specify options detailing which plots will be made, and in which format, e.g.:
   # {timing: True, convergence: True, trace: False, slices: False, format: svg}
   # (Adds significant overhead for fast likelihoods.)
   plots: False

   # Fiducial point and/or MC samples
   fiducial_point: # dict of {parameters: values}; can use keys 'logpost' and 'loglike'.
   fiducial_mc: # path to MC samples in Cobaya-compatible format

   # Function run each iteration after adapting the recently acquired points and
   # the computation of the convergence criterion. See docs for implementation.
   callback:

   # Whether the callback function handles MPI-parallelization internally
   # Otherwise run only by the rank-0 process
   callback_is_MPI_aware:

   # Change to increase or reduce verbosity. If None, it is handled by Cobaya.
   # '3' produces general progress output (default for Cobaya if None),
   # and '4' debug-level output
   verbose:

You can also get this info by running ``python -m cobaya doc gpry`` in a shell.

.. note::

   The combination of GPry + Cobaya is MPI-parallelizable!

A call to the Cobaya ``run`` function or ``cobaya-run`` shell command will run the GPry's learning loop and, after convergence or exhaustion of the evaluation budget, will run an MC sample of the surrogate model, and return/write it in the same way as other Cobaya-interfaced samplers:

- As a :ref:`Cobaya SampleCollection <cobaya:output_format>` using the ``samples`` or ``products`` method of the Cobaya sampler wrapper (if in a script or a notebook).

- Written into a ``[prefix].1.txt`` file if calling ``cobaya-run`` from the command line.

.. note::

   As with other Cobaya samplers, you can resume a run with the ``-r/--resume`` argument (command line) or with ``resume=True`` (script).

.. note::

   If a run has not converged before the evaluation budget has been exhausted, a warning will be printed at the end of the output, and an MC sample will be run and saved anyway.

The last state/checkpoint of the GPry relevant objects of the run can be found, if writing to the hard drive, in a ``[prefix]_gpry_output`` sub-folder, or via the ``Runner`` instance stored in a ``gpry_runner`` attribute of the Cobaya ``sampler`` object, which in this case is the :class:`GPry Cobaya wrapper <CobayaWrapper>`:

.. code:: python

   input_dict = {...}

   from cobaya.run import run

   upd_input, gpry_wrapper = run(input_dict)

   # The GPry runner is at:
   gpry_wrapper.gpry_runner


Doing some plots
^^^^^^^^^^^^^^^^

A full set of plots can be generated and saved automatically by passing a ``plot: True`` option to the Cobaya gpry wrapper. The value of ``plot`` can also be a dictionary of the arguments passed to the :meth:`run.Runner.plot_progress` (click for documentation) method, to control which plots are made, and in which format.

Individual plots can be generated by hand using the methods of the stored ``gpry_runner`` attribute (see above).

Alternative: running GPRy with a Cobaya Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to control the calls to GPRy but still use a Cobaya-defined model pipeline, you can pass a :doc:`Cobaya model <cobaya:models>` as the first argument of the :class:`~run.Runner`, and run GPry normally.

In this case, GPry will use the prior specified in the Cobaya model, so there is no need to pass parameter ``bounds`` to the :class:`~run.Runner`. Parameter names and labels will also be automatically used for tables and plots.


Module documentation
--------------------

.. automodule:: gpry.cobaya_interface
   :members:
   :undoc-members:
   :show-inheritance:
