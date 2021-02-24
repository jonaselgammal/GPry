# GPry
### A Package for Bayesian inference of expensive likelihoods with Gaussian Processes.

## This package is still under heavy development and not ready for use...

[Documentation on Read the Docs](https://gpry.readthedocs.io/en/latest/)

This package is designed to be used with any kind of Likelihood/Posterior but the main
focus is on cosmological likelihoods for now.

GPry was developed as a result of my master thesis and and can be seen as an alternative to MCMC.
It is targeted at a specific class of likelihood functions which frequently appear in cosmology
(but also in other scientific fields). The requirements that have to be fulfilled in order for this
algorithm to be efficient and give correct results are as follows:

- The likelihood/posterior should be *smooth* (continuous) and you should know how smooth (how many
  times differentiable).
- The likelihood/posterior evaluation should be *slow*. What slow means depends on the number of dimensions
  and expected shape of the posterior distribution though.
- The likelihood should be *low-dimensional* (d<10 as a rule of thumb).
- If d>10 there should be a pronounced *speed hierarchy* between the dimensions such that the likelihood/posterior
  can be split into *slow* dimensions which are evaluated using the GP algorithm and *fast* dimensions which are
  evaluated using [PolyChord](https://arxiv.org/abs/1502.01856). The speed hierarchy usually needs to be at least
  one order of magnitude.

## TODOs for the release:

- Fix the convergence criterion (kl_from_draw does not seem to be working correctly at all)
- Do some robustness tests of the SVM. It seems like *C* needs to be set to a finite value for the SVM to converge
  in a finite time but this should be checked in detail.
- Maybe write a method which dynamically expands the hyperparameter space prior so the user doesn't have to worry about this
- Interface to Cobaya
- Write a module which acts as a "frontend" for easy use.
