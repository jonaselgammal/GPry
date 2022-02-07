# GPry
### A Package for Bayesian inference of expensive likelihoods with Gaussian Processes.

[Documentation on Read the Docs](https://gpry.readthedocs.io/en/latest/)

GPry was developed as a result of my master thesis and and can be seen as an alternative to MCMC.
It is targeted at a specific class of posterior distributions (with the initial goal of speeding up inference in cosmology) with the aim of creating an algorithm which can correctly get marginal quantities for expensive likelihoods.

Although my background is in Cosmology GPry will work with any likelihood which can be called as a python function.

### What kinds of likelihoods/posteriors work with GPry?
The requirements that your likelihood/posterior has to fulfil in order for this algorithm to be efficient and give correct results are as follows:

- The likelihood/posterior should be *smooth* (continuous) and you should know how smooth (how many times differentiable).
- The likelihood/posterior evaluation should be *slow*. What slow means depends on the number of dimensions and expected shape of the posterior distribution but as a rule of thumb, if your MCMC takes longer to converge than you're willing to wait you should definitely give it a shot. 
- The likelihood should be *low-dimensional* (d<20 as a rule of thumb). In higher dimensions you might still gain considerable improvements in speed if your likelihood is sufficiently slow but the computational overhead of the algorithm increases considerably.

### What does GPry do?
Unlike algorithms like MCMC which sample a posterior distribution GPry is designed to interpolate it using Gaussian Process regression and active sampling. This converges to the posterior shape requiring much fewer posterior samples than sampling algorithms because it sets a prior on the functional shape of the posterior.
This doesn't mean that your posterior has to have a certain shape (i.e. it doesn't have to be a perfect gaussian) but rather that we assume that the posterior is a continuous, differentiable function which has a single characteristic length scale along each dimension.
Furthermore GPry implements a number of tricks to mitigate some of the pitfalls associated with interpolating functions with GPs. The most important ones are:
- A novel **acquisition function** for efficient sampling of the parameter space. This procedure is inspired by Bayesian optimization.
- A batch acquisition algorithm which enables evaluating the likelihood/posterior in parallel using multiple cores. This is based on the **Kriging-believer** algorithm. We increase the performance of this using the block-wise inversion lemma to save computation resources.
- In order to prevent sampling regions which fall well outside the 2- &sigma; contours and account for the fact that many theory codes just return 0 far away from the fiducial values instead of computing the actual likelihood (which leads to - &‌infin; in the log-posterior) we shrink the prior using an **SVM classifier** to divide the parameter space into a "finite" and "infinite" region.

### What benefits does GPry offer compared to MCMC, Nested Sampling, ...?
To put it bluntly mostly the number of samples required to converge to the correct posterior shape. The increase in performance is therefore most pronounced in cases where evaluating the likelihood/posterior at a single location is very costly (i.e. when it requires running some expensive theory calculations, large amounts of data need to be processed, ...) .

### Why does GPry require so many less samples to converge?
Unlike most samplers GPry does not select sampling locations statistically but instead uses a deterministic function which is optimized in order to always sample the location which adds most information. Furthermore unlike samplers which essentially build a histogram of the sampling locations (like MCMC) and are oblivious to the posterior values themselves GPry uses all information it can get from the samples. Furthermore there are no such things as rejected steps. Every sample contributes to the GP interpolation.

