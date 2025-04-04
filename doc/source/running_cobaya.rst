.. _running_cobaya:

===================
Running with Cobaya
===================

GPry has two ways to interface with the general and cosmological Bayesian inference code `Cobaya <https://cobaya.readthedocs.io>`_.

.. note::

   Regardless of the interfacing method, the combination of GPry + Cobaya is MPI-parallelizable!


a) Passing a Cobaya ``model`` to the GPry ``Runner``
----------------------------------------------------

You can pass a :doc:`Cobaya model <cobaya:models>` as the first argument of the :class:`~run.Runner`, and run GPry normally.

In this case, GPry will use the prior specified in the Cobaya model, so there is no need to pass parameter ``bounds`` to the :class:`~run.Runner`. Parameter names and labels will also be automatically used for tables and plots.


b) Running Cobaya with GPry as a sampler
----------------------------------------

Conversely, you can integrate GPry into a Cobaya-based inference pipeline, including for example calls to a minimizer, or tests with alternative samplers. As as GPry is installed, this is as simple as naming GPry as the ``sampler`` of choice in an input ``.yaml`` file or an input dictionary:

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

   output: chains/example

A call to the Cobaya ``run`` function or ``cobaya-run`` shell command will run the GPry's learning loop and, after convergence or exhaustion of the evaluation budget, will run an MC sample of the surrogate model, and return/write it in the same way as other Cobaya-interfaced samplers:

- As a :ref:`Cobaya SampleCollection <cobaya:output_format>` using the ``samples`` or ``products`` method of the Cobaya sampler wrapper (if in a script or a notebook).

- Written into a ``[prefix].1.txt`` [TODO: CHECK SUFFIX] file if calling ``cobaya-run`` from the command line.

.. note::

   As with other Cobaya samplers, you can resume a run with the ``-r/--resume`` argument (command line) or with ``resume=True`` (script).

.. note::

   If a run has not converged before the evaluation budget has been exhausted, a warning will be printed at the end of the output, and an MC sample will be run and saved anyway.

The last state/checkpoint of the GPry relevant objects of the run can be found, if writing to the hard drive, in a ``[prefix]_gpry_output`` sub-folder, or via the ``Runner`` instance stored in a ``gpry_runner`` attribute of the Cobaya ``sampler`` object, which in this case is the GPry Cobaya wrapper, :ref:`documented below <cobaya_wrapper>`:

.. code:: python

   input_dict = {...}

   from cobaya.run import run

   upd_input, gpry_wrapper = run(input_dict)

   # The GPry runner is at:
   gpry_wrapper.gpry_runner


Doing some plots
^^^^^^^^^^^^^^^^

A full set of plots can be generated and saved automatically by passing a ``plot: True`` option to the Cobaya gpry wrapper (see :ref:`cobaya_custom`), or an any point later calling the :meth:`~gpry.cobaya.CobayaWrapper.do_plots` method of the wrapper.

The ``plot`` option can be assigned also individual arguments of :meth:`run.Runner.plot_progress` (click for documentation), to control which plots are made, and in which format.

Individual plots can be generated by hand using the methods of the stored ``gpry_runner`` attribute (see above).


Refining/re-running the MC sample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you would like to re-run the MC sample of the surrogate model with a different sampler or with different settings, you can call the :meth:`~gpry.cobaya.CobayaWrapper.do_surrogate_sample` method of the wrapper by hand. The new samples will be written to the hard drive or returned by the :meth:`~gpry.cobaya.CobayaWrapper.samples` or :meth:`~gpry.cobaya.CobayaWrapper.products` method of the wrapper.

You can pass that method a different ``sampler`` to the one specified at initialisation, and/or different options for it using the ``add_options`` argument.


.. _cobaya_custom:

Customising the GPry run from Cobaya
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same as for other Cobaya samplers, ``gpry`` can be customised setting its options as a dictionary. These options are the same, with small formatting differences, as for the :class:`~run.Runner`. An overview with default values can be seen below:

.. note::

   Undefined values are left to their default for the :class:`~run.Runner`. In general, to disable a feature, set it to ``False`` instead of not ``None``.

.. literalinclude:: ../../gpry/CobayaWrapper.yaml
   :language: yaml


.. _cobaya_wrapper:

The Cobaya wrapper
------------------

.. autoclass:: gpry.cobaya.CobayaWrapper
   :members:
