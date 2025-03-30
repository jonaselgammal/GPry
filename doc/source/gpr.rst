Gaussian Process Regressor
==========================

The GPR model
-------------

The infinities classifier
-------------------------

Though in principle there is no lower limit to the log-posterior values that GPry can handle, there are reasons in practice for dealing with very small posterior values in a different way:

- Large negative log-posteriors, especially those that are literally or effectively minus infinity, can create instabilities in the GP interpolation, even when regularised.
- It is common that returning these values is the way likelihood implementations signify that somewhere along the computational pipeline a particular step failed, so it would not make sense to include it in the interpolation.
- Especially for likelihoods of noisy data, very-low-likelihood values have numerical (deterministic) noise, which does not make sense to model with the GPR.

Since GPry is an inference code, aiming at modelling probability density functions around their modes, it makes sense to censor such values, and if possible to predict them before evaluation of the true likelihood to prevent wasting time exploring a very-low-probability region and making the GPR model heavier.

To do that, if the ``acount_for_inf`` option of the GPR is set to ``SVM`` (default), GPry uses a given /threshold/ (``inf_threshold``) to classify values of the true and simulated log-posterior as either finite or negative infinity, depending on whether the difference between their log-posterior and the maximum found so far is smaller or larger than the threshold. Only points classified as /finite/ will form part of the GPR training set, whereas both point types will be used to retrain an SVM classifier at each iteration. The SVM classifier partitions the parameter space into /finite/ and /minus infinity/ regions. Points in parameter space that fall in the /minus infinity/ region are automatically discarded during the acquisition phase before their true posterior evaluation.

The SVM classifier is stored in the ``infinities_classifier`` attribute of the GPR instance. Because it is defined in the transformed space [TODO: ADD REFERENCE], its methods should not be used directly. Instead, it can be called with un-transformed parameter input via the GPR methods :meth:`~gpr.GaussianProcessRegressor.is_finite`, that takes a log-posterior value and returns a boolean depending of whether it is above the threshold, and :meth:`~gpr.GaussianProcessRegressor.predict_is_finite`, which takes a point in parameter space and returns a boolean depending on whether a log-posterior evaluation is expected to be finite at that location. Both methods accept vector input. The current log-posterior threshold is returned by the property :meth:`~gpr.GaussianProcessRegressor.abs_finite_threshold`.

.. automodule:: gpr
   :members:
   :private-members:
   :show-inheritance:
