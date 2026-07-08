Initialization
==============

In some problems, a good initialization can be key to efficient convergence. These cases include:

- Difficulty localizing the mode, if it is very small with respect to the prior, or if the prior is not significantly sloped towards it.

- Difficulty detecting multiple modes, if they are too separated (see the :doc:`adv_example` for settings that would also help in those cases).

This section is aimed at giving some general pointers for good initialization, relying on a priori information about the posterior. This a priori information can be gathered from physical intuition, simplified versions of the problem, or other less-accurate but faster (amortized) ML approaches which can produce fast estimates of the maxima a posteriori of regions of interest, and which often exist in real-world problems where GPry is an attractive tool because of the low inference costs.


Prior size
----------

.. hint::
   **The most immediately-helpful measure is to crop the prior as close as possible around the position of the expected mode.**

As long as one has some confidence that the center of the mode is captured, and that the distribution is not expected to have secondary modes outside that region (i.e. be multi-modal), this is usually a safe approach.

If the full mode is not contained within the cropped prior, this will show up in the final corner plot as the mode not fully vanishing towards the edge of the prior, indicating that it should be enlarged.


Initial proposals for training points
--------------------------------------

If one really does not know how far along the parameters the mode extends, but does know a region where part of the mode would be located, we can use this region to propose the initial set of training points. To do that, pass ``initial_proposer="reference"`` to the :class:`~gpry.run.Runner`, and pass that region as a list of bounds as ``ref_bounds=[[min,  max], [min, max], ...]``.

This knowledge can be useful even if we only have this sort of knowledge over one or a few parameters. In that case, pass ``None`` in the list of ``ref_bounds`` for the parameters for which we do not have that information, and their initial proposals will be drawn from the prior.

For example, if we expect the region ``[0, 1]`` for the second parameter to be high likelihood value, we would initialize the :class:`~gpry.run.Runner` as:

.. code:: python

   runner = Runner(
       loglike=...,
       bounds=[[-10, 10], [-10, 10]],
       ref_bounds=[None, [0, 1]],
       initial_proposer="reference",
       ...
   )

If that knowledge is better expressed as a gaussian or ellipsoidal region defined by a mean and a covariance matrix, we can use instead the ``meancov`` proposer as

.. code:: python

   runner = Runner(
       loglike=...,
       bounds=[[-10, 10], [-10, 10]],
       initial_proposer={"meancov": {"mean": [mean], "cov": [covmat]}},
       ...
   )
