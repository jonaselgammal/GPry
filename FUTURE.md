Notes on future developments
----------------------------

## Greediness and relationship with long correlation lengths

Long correlation lengths (~size of the prior) are producing ill-defined GP's: in principle they spoil the acquisition function. In particular the linearies acq function basically turns into the mean with some noise, which can acquire points too close to each other, creating a badly-conditioned kernel matrix. In any case, a GP with badly-defined standard deviation is never desirable.

One could also argue that we would want to encourage exploration somewhat, which would also point towards a low correlation length.

On the other hand, a very small correlation length (e.g. avg distance between training samples) may encourage exploration too much, since we don't know what's beyond the furthest points. In that case, we need to sample with certain density away from the centre, and then our approach is not better than e.g. latin hypercube sampling: if we need to sample a few points per dimention, we need n**dim, which diverges. The only solution to that is being greedy by ignoring low-value parts of the parameter space, and long correlation lengths do precisely that, so we would prefer to keep them.

The contradiction may be solved by splitting the tasks of "defining sigma for acquisition" and "making strong assumptions away from the mode", and give them to different parts of the algorithm.

One proposal is to use the baseline for the "making assumptions" part. Some possibilities:
- fit another GP with longer correlation length allowed and use its mean for the baseline. to make it cheaper, re-fit only once in while
- use as the baseline the smallest function value, instead of the mean of them.

In any case, the correlation length grows naturally for simple functions such as our gaussians, so whatever the solution we would need to add a prior/cut to the log-marginal-likelihood of the hyperparameters when refitting.


## Planck and regions of interest in "very" high dimensions

It looks like in some cases the GP fits well the mode of interest, but it also diverges towards high values e.g. when approaching a prior boundary, which is detected when running a Monte Carlo on the process, that explores the fake mode instead of the real one. It happens relatively often with Full Planck. In this case, the algorithm has succeeded within the region of interest, but failed globally.

A few solutions have been proposed:
- cut the prior boundaries so that they are closer to the mode.
- run with full priors, but distribute together with the resulting GP a definition of a "region of interest"

## Initial point generation in high dimensions
As the mode/prior volume shrinks exponentially as function of the number of dimensions the finite log-posterior area becomes extremely small. In such a case drawing initial samples from the prior is typically unfeasable. While there is currently the option to pass a reference distribution which acts as a rough guess of where the mode lies it would be good to give the option to just pass a single finite point and run a very short (burn-in) MCMC chain from there to have some initial samples from where to start the BO loop.
