How does GPry work
==================

GPry creates an interpolating Gaussian Process (GP) model of the log-posterior density function. It does so using the least amount of evaluations possible. The locations in parameter space of these evaluations are chosen sequentially, so that they maximise the amount of information that can be obtained by evaluating the posterior there. This careful selection, together with the prior on the functional shape of the posterior that the GP imposes, helps GPry converge towards the true distribution using usually a factor :math:`\mathcal{O}(10^{-2})` of the evaluations needed by a traditional Monte Carlo sampler (such as MCMC or Nested Sampling).

Here we explain some of the key aspects of the GPry algorithm. Unless otherwise stated, what follows is not exclusive to or pioneered by GPry, but typical in active learning approaches.


Active learning of a Gaussian Process
-------------------------------------

GPry does not need any pre-training: it trains its surrogate model at run time by selecting optimal evaluation locations. The process of selecting these optimal locations based on current information is commonly known as **active learning**. It involves finding the maximum of an **acquisition function** measuring the amount of information about the true model expected to be gained by evaluating it at a given point. Acquisition functions must manage a good balance between **exploration** (getting an overall-good model of the true function) versus **exploitation** (prioritising a better modelling of the true function where its value is the highest). The default acquisition function used by GPry is described in section :doc:`acquisition_functions`. The automatic scaling with dimensionality of the balance between exploration and exploitation of this acquisition function is one of the novel aspects of GPry.

You can see the way active learning works in the following figure: the top plots show the current GP model, and the bottom ones the value of the acquisition function (for this simple example, the GP standard deviation times the exponential of the double of the GP mean); every column is an iteration of the algorithm. Notice how at every step an evaluation of the true function at the previous maximum of the acquisition function has been added:

.. image:: images/active_learning.png
   :width: 950
   :align: center

.. note::

   This aspect of GPry is one of the main difference with **amortised** approaches, such as forward-modelling (or simulation-based) inference: the latter can produce inference at very low cost in exchange for some usually-costly pre-training; instead, the cost of inference in active learning approaches is higher (due both to the need for evaluating the true posterior at least a few times, and the overhead from active sampling), but they do not need pre-training.


Parallelising truth evaluation with kriging-believer
----------------------------------------------------

The active learning approach described above is sequential: one candidate is proposed for evaluation of the true posterior at a time. But being this evaluation often the slowest step, if one has sufficient computational resources available to perform :math:`n` posterior evaluations in parallel it would be desirable to obtain not just the next optimal location, but the next :math:`n` optimal ones.

However, simultaneously optimizing the acquisition function for a set of candidate locations is not a trivial problem: each of the candidates modifies the landscape of the acquisition function for the rest, so that we cannot simply assume that a set of local maxima is a viable solution.

But there is one way to give up some effectiveness (total information gained) of the solution in exchange for the possibility to turn this into a sequential problem: we can find the global maximum, assume an evaluation of the true model there, to which the **mean of the GP** is assigned, create with it an *augmented model*, and repeat this procedure using the augmented model, as many times as desired. This approach is called *kriging-believer* (KB), and though suboptimal, it at least includes the effect of the *exploration* term of the acquisition function, reducing the amount of redundant information with respect to a naive multiple-candidate solution.

Obviously, this procedure only makes sense up to a certain amount of iterations, or we risk assuming completely false information about the model. In GPry, we recommend at most a number of KB steps equals to the dimensionality of the problem (times some factor smaller or equal the number of expected posterior modes, if more than one).

In the following figure, to be compared with the one above, we only evaluate the posterior every two steps, the red stars in being the temporary kriging-believer evaluations that will be assigned their true values in the next iteration. It performs slightly worse, but has the advantage that the true posterior can be evaluated in parallel in batches of two points.

.. image:: images/active_learning_kb.png
   :width: 950
   :align: center


The acquisition engine
----------------------

It is implied above that the acquisition step of active learning involves a direct optimization of the acquisition function. GPry provides an acquisition engine that does precisely that, with some parallelization involved (see :ref:`batchoptimizer`).

GPry also introduces an alternative approach called NORA (Nested sampling Optimization for Ranked Acquistion). In it, the optimization of the acquisition function is swapped by a Nested Sampling exploration of the mean of the GP. The resulting sample is then ranked according to their acquisition function values, and subsequently re-ranked after sequentially augmenting the GP with the point at the top of the list. For more detail, see :ref:`nora`.

This approach has a number of advantages:

- NS is extremely efficiently parallelizable, and the raking of the NS sample too (but less efficiently). This greatly helps with the increase in dimensionality.
- This approach provides a better exploration of the parameter space, since NS probes the tails of the (surrogate) posterior, whereas in a direct optimization approach the problem of proposing good starting points for optimization is not trivial, and diverges worse with dimensionality than NS does.
- Since a sample from the mean GP is produced together with the candidates, better diagnosis and convergence tools are available at every iteration.

This approach to parallelising the acquisition process itself is another of GPry's novel aspects.



Fitting the surrogate model
---------------------------

Updating the surrogate model with the new evaluations entails two distinct operations:

- Conditioning the Gaussian Process Regressor on the new, enlarged set of training samples.

- Choosing the optimal hyperparameters for the kernel given the new information.

The first one entails [TODO] and scales as :math:`N^2`, where :math:`N` is the number of training samples. The second one entails [TODO] and thus scales as :math:`N^3`. Because of this large scaling, and also because we do not expect the addition of new training samples to dramatically change the value of the optimal kernel hyperparameters, we do not perform the second operation (full hyperparameter fit) at every iteration (or we may decide doing a mild version of it, such as only optimizing once from the optimum of the last iteration, instead of re-running the optimizer from different points in hyperparameter space).

.. note::

   At this step of the algorithm we also re-fit the pre-processors for the input and output data, as well as, if used, the SVM aimed at classifying regions of the parameter space as either interesting (if the posterior value is expected to be significantly high) or not (if the posterior value is expected to be very or infinitely low), see section :ref:`svm`.


Convergence check
-----------------

Since we do not have access in general to the target distribution, assuming GPry is converging towards the right target, we base our criteria on stability of the current surrogate model. By default, we use two criteria:

- That we do not get any more *surprises* when evaluating the true posterior at the proposal optimal locations, by comparing the obtained value with the mean GP prediction. See :class:`convergence.CorrectCounter`.

- That the current surrogate model does not diverge significantly from that of the previous iteration. For this last one, we need a Monte Carlo sample of the surrogate posterior, which, if we are using NORA, we already have at hand at every iteration. See :class:`convergence.GaussianKL`.

On top of these criteria, we check that the region where the GP is value highest corresponds to the location of the highest training samples, in case the GP is has temporarily high expectation value for a region with no training support, which is being explored at the moment. See :class:`convergence.TrainAlignment`. We use this as a necessary but not sufficient condition.


The algorithm, putting everything together
------------------------------------------


