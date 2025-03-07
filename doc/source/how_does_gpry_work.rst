How does GPry work
==================

From old README
===============

Unlike algorithms like MCMC which sample a posterior distribution GPry is designed to interpolate it using Gaussian Process regression and active sampling. This converges to the posterior shape requiring much fewer posterior samples than sampling algorithms because it sets a prior on the functional shape of the posterior.
This doesn't mean that your posterior has to have a certain shape but rather that we assume that the posterior is a continuous, differentiable function which has a single characteristic length scale along each dimension (don't take this too literally though. It will still work with many likelihoods which do not have a single characteristic length-scale at all!)
Furthermore GPry implements a number of tricks to mitigate some of the pitfalls associated with interpolating functions with GPs. The most important ones are:
- A novel **acquisition function** for efficient sampling of the parameter space. This procedure is inspired by Bayesian optimization.
- A batch acquisition algorithm which enables evaluating the likelihood/posterior in parallel using multiple cores. This is based on the **Kriging-believer** algorithm. A nice bonus is that it also decreases the time for fitting the GP's hyperparameters.
- In order to prevent sampling regions which fall well outside the 2- &sigma; contours and account for the fact that many theory codes just return 0 far away from the fiducial values instead of computing the actual likelihood (which leads to - &infin; in the log-posterior) we shrink the prior using an **SVM classifier** to divide the parameter space into a "finite" region of interest and an "infinite" (uninteresting) region.
- Instead of simply optimizing the acquisition function we created the **N**ested sampling **O**ptimization for **R**anked **A**cquisition algorithm for parallelizing the acquisition procedure and gaining robustness with regards to the highly multimodal nature of the acquisition function.


What kinds of likelihoods/posteriors work with GPry?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The requirements that your likelihood/posterior has to fulfil in order for this algorithm to be efficient and give correct results are as follows:

- The likelihood/posterior should be *smooth* (continuous) and you should know how smooth (how many times differentiable).
(there can be a little noise, but it must be deterministic!!!!)
  
- The likelihood/posterior evaluation should be *slow*. What slow means depends on the number of dimensions and expected shape of the posterior distribution but as a rule of thumb, if your MCMC takes longer to converge than you're willing to wait you should give it a shot.
- The likelihood should be *low-dimensional* (d<20 as a rule of thumb). In higher dimensions you might still gain considerable improvements in speed if your likelihood is sufficiently slow but the computational overhead of the algorithm increases considerably.

Where's the catch?
^^^^^^^^^^^^^^^^^^

Like every other sampler GPry isn't perfect and has some limitations:
- GPs don't scale well with the number of training samples as training the GP involves inverting a kernel matrix. Unfortunately the computational complexity of this inversion scales with the number of training samples cubed. This means that as the number of training samples required grows, the overhead of the algorithm increases considerably.
- While we tested GPry on mildly multimodal posterior distributions it is not a supported feature and should be used with caution. The algorithm is fairly greedy and can easily miss a mode, especially if the separation between modes is large.
- The algorithm is generally robust towards "weirdly" shaped posterior distributions, however the structure of the kernel still assumes a single characteristic correlation length. This means that in very pathological cases (for instance a very wide distribution with a tiny spike in it) GPry will struggle to capture the mode correctly.
- This code is novel and hasn't profited from years of user feedback and maintenance. Bugs can (and probably will) occur. If you find any please report them to us!

**We are actively working on mitigating some of those issues and we will keep on developing this code so look out for new versions!**
