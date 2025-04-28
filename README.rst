**GPry**: A package for Bayesian inference of expensive likelihoods with Gaussian Processes
-------------------------------------------------------------------------------------------

:Author: Jonas El Gammal, Jesus Torrado, Nils Schoeneberg and Christian Fidler

:Source: `Source code on GitHub <https://github.com/jonaselgammal/GPry>`_

:Documentation: `Documentation on Read the Docs <https://gpry.readthedocs.io>`_

:License: `LGPL <https://www.gnu.org/licenses/lgpl-3.0.en.html>`_ + mandatory bug reporting asap + mandatory `arXiv'ing <https://arxiv.org>`_ of publications using it (see `LICENSE <https://github.com/jonaselgammal/GPry/blob/main/LICENSE>`_ for exceptions). The documentation is licensed under the `GFDL <https://www.gnu.org/licenses/fdl-1.3.en.html>`_.

:Support: For questions drop me an `email <mailto:jonas.e.elgammal@uis.no>`_. For issues/bugs please use `GitHub's Issues <https://github.com/jonaselgammal/GPry/issues>`_.

:Installation: ``pip install gpry`` (for MPI and nested samplers, see `here <https://gpry.readthedocs.io/en/latest/installation.html>`_)

GPry is a drop-in alternative to traditional Monte Carlo samplers (such as MCMC or Nested Sampling), for likelihood-based inference. It is aimed at speeding up posterior exploration and inference of marginal quantities from computationally expensive likelihoods, reducing the cost of inference by a factor of 100 or more.

GPry can be installed with pip (``python -m pip install gpry``), and needs only a callable likelihood and some bounds:

.. code:: python

   def log_likelihood(x, y):
       return [...]

   bounds = [[..., ...], [..., ...]]
          
   from gpry import Runner

   runner = Runner(log_likelihood, bounds, checkpoint="output/")
   runner.run()

.. image:: https://github.com/jonaselgammal/GPry/blob/balrog/doc/source/images/readme_animation.gif?raw=true
   :width: 400px
   :align: center

An `interface to the Cobaya sampler <https://gpry.readthedocs.io/en/latest/running_cobaya.html>`_ is available, for richer model especification, and direct access to some physical likelihood pipelines. 

GPry was developed as part of J. El Gammal's M.Sc. and Ph.D. thesis projects.


How it works
^^^^^^^^^^^^

GPry uses a `Gaussian Process <https://gaussianprocess.org/gpml/>`_ (GP) to create an interpolating model of the log-posterior density function, using as few evaluations as possible. It achieves that using **active learning**: starting from a minimal set of training samples, the next ones are chosen so that they maximise the information gained on the posterior shape. For more details, see section `How GPry works <https://gpry.readthedocs.io/how-it-works>`_ of the documentation, and check out the :ref:`GPry papers <readme_cite>`.

GPry introduces some innovations with respect to previous similar approaches [TODO: citations]:

- It imposes weakly-informative priors on the target function, based on a comparison with an n-dimensional Gaussian, and uses that information e.g. for convergence metrics, balancing exploration vs. exploitation, etc.

- It introduces a parallelizable batch acquisition algorithm (NORA) which increases robustness, reduces overhead and enables the evaluation of the likelihood/posterior in parallel using multiple cores.

- Complementing the GP model, it implements an SVM classifier that learns the shape of uninteresting regions, where proposal are discarded, wherever the value of the likelihood is very-low (for increased efficiency) or undefined (for increased robustness).

At the moment, GPry utilizes a modification of the CPU-based `scikit-learn GP implementation <https://scikit-learn.org/stable/modules/gaussian_process.html>`_.

  
What kinds of likelihoods/posteriors should work with GPry?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Non-stochastic log-probability density functions, smooth up to a small amount of (deterministic) numerical noise (:math:`\Delta\log p \sim 0.1`).

- Large evaluation times, so that the GPry overhead is subdominant with respect to posterior evaluation. How slow depends on the number of dimensions and expected shape of the posterior distribution but as a rule of thumb, if an MCMC takes longer to converge than you're willing to wait you should give it a shot.

- The parameter space needs to be *low-dimensional* (:math:`d<20` as a rule of thumb). In higher dimensions you might still gain considerable improvements in speed if your likelihood is sufficiently slow but the computational overhead of the algorithm increases considerably.

What may not work so well:

- Highly multimodal posteriors, especially if the separation between modes is large.

- Highly non-Gaussian posteriors, that would not be well modelled by orthogonal constant correlation lengths.

**GPry is under active developing, in order to mitigate some of those issues, so look out for new versions!**


It does not work!
^^^^^^^^^^^^^^^^^

Please check out the `Strategy and Troubleshooting <https://gpry.readthedocs.io/strategy>`_ page, or get in touch for `issues <https://github.com/jonaselgammal/GPry/issues>`_ or `more general discussions <https://github.com/jonaselgammal/GPry/discussions>`_.


.. _readme_cite:

What to cite
^^^^^^^^^^^^

If you use GPry, please cite the following papers:

- `arXiv:2211.02045 <https://arxiv.org/abs/2211.02045>`_ for the core algorithm.
- `arXiv:2305.19267 <https://arxiv.org/abs/2305.19267>`_ for the NORA Nested-Sampling acquisition engine.

Papers applying GPry
^^^^^^^^^^^^^^^^^^^^

- `arXiv:2503.21871 <https://arxiv.org/abs/2503.21871>`_: astrophysical resolvable gravitational wave sources with the LISA survey (inference forecasts).
