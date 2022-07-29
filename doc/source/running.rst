======================================
Running the Bayesian optimization loop
======================================

The best way to run the Bayesian optimization loop is to use the :class:`run.Runner`
class from which to create a ``Runner`` object. This has to be fed with at least the
``Cobaya`` model we want to run on and optionally takes instances of the GaussianProcessRegressor,
GPAcquisition and ConvergenceCriterion as well as an options dict for the run. You can
then run the model using the :meth:`run.Runner.run` method. After the model has run you
can then extract marginalised quantities by calling :meth:`run.Runner.generate_mc_sample`.

Furthermore the runner object allows for checkpointing, callbacks and can create a
triangle plot of the final MC sample. For more information on how to use the runner object
look into the examples.

For more information on the exact usage have a look at the documentation of :mod:`run`
