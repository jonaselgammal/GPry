# GPry
### A Package for Bayesian inference of expensive likelihoods with Gaussian Processes.


| Authors       | Jonas El Gammal, Jesus Torrado, Nils Schoeneberg and Christian Fidler                                                                                                                                                                                                                                                                                      |
|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Source code   | [Source code on GitHub](https://github.com/jonaselgammal/GPry)                                                                                                                                                                                                                                                                                             |
| Documentation | [Documentation on Read the Docs](https://gpry.readthedocs.io/en/latest/)                                                                                                                                                                                                                                                                                   |
| License       | [LGPL](https://www.gnu.org/licenses/lgpl-3.0.en.html) + mandatory bug reporting asap + <br>mandatory [arXiv'ing](https://arxiv.org) of publications using it (see [LICENCE.txt](https://github.com/jonaselgammal/GPry/blob/main/LICENSE) for exceptions).<br>The documentation is licensed under the [GFDL](https://www.gnu.org/licenses/fdl-1.3.en.html). |
| Support       | For questions drop me an [email](mailto:jonas.el.gammal@rwth-aachen.de). For issues/bugs please use GitHub's functions.                                                                                                                                                                                                                                    |
| Installation  | ``pip install gpry``                                                                                                                                                                                                                                                                                                                                       |

GPry was developed as a result of my master thesis and and can be seen as an alternative to established samplers like MCMC and Nested Sampling.
It is targeted at a specific class of posterior distributions (with the initial goal of speeding up inference in cosmology) with the aim of creating an algorithm which can efficiently obtain marginal quantities from (computationally) expensive likelihoods.

Although our background is in cosmology, GPry will work with any likelihood which can be called as a python function. It uses [Cobaya's](https://github.com/CobayaSampler/cobaya) model framework so all of Cobaya's inbuilt likelihoods work too.

### What kinds of likelihoods/posteriors work with GPry?
The requirements that your likelihood/posterior has to fulfil in order for this algorithm to be efficient and give correct results are as follows:

- The likelihood/posterior should be *smooth* (continuous) and you should know how smooth (how many times differentiable).
- The likelihood/posterior evaluation should be *slow*. What slow means depends on the number of dimensions and expected shape of the posterior distribution but as a rule of thumb, if your MCMC takes longer to converge than you're willing to wait you should give it a shot.
- The likelihood should be *low-dimensional* (d<20 as a rule of thumb). In higher dimensions you might still gain considerable improvements in speed if your likelihood is sufficiently slow but the computational overhead of the algorithm increases considerably.

### What does GPry do?
Unlike algorithms like MCMC which sample a posterior distribution GPry is designed to interpolate it using Gaussian Process regression and active sampling. This converges to the posterior shape requiring much fewer posterior samples than sampling algorithms because it sets a prior on the functional shape of the posterior.
This doesn't mean that your posterior has to have a certain shape but rather that we assume that the posterior is a continuous, differentiable function which has a single characteristic length scale along each dimension (don't take this too literally though. It will still work with many likelihoods which do not have a single characteristic length-scale at all!)
Furthermore GPry implements a number of tricks to mitigate some of the pitfalls associated with interpolating functions with GPs. The most important ones are:
- A novel **acquisition function** for efficient sampling of the parameter space. This procedure is inspired by Bayesian optimization.
- A batch acquisition algorithm which enables evaluating the likelihood/posterior in parallel using multiple cores. This is based on the **Kriging-believer** algorithm. A nice bonus is that it also decreases the time for fitting the GP's hyperparameters.
- In order to prevent sampling regions which fall well outside the 2- &sigma; contours and account for the fact that many theory codes just return 0 far away from the fiducial values instead of computing the actual likelihood (which leads to - &infin; in the log-posterior) we shrink the prior using an **SVM classifier** to divide the parameter space into a "finite" region of interest and an "infinite" (uninteresting) region.

### What benefits does GPry offer compared to MCMC, Nested Sampling, ...?
To put it bluntly mostly the number of samples required to converge to the correct posterior shape. The increase in performance is therefore most pronounced in cases where evaluating the likelihood/posterior at a single location is very costly (i.e. when it requires running some expensive theory calculations, large amounts of data need to be processed, ...).

### Why does GPry require so few samples to converge?
Unlike most samplers GPry does not select sampling locations statistically but instead uses a deterministic function which is optimized in order to always sample the location which adds most information. Furthermore, unlike samplers which essentially build a histogram of the sampling locations (like MCMC) and are oblivious to the posterior values themselves, GPry uses all information it can get from the samples by interpolating. There are no such things as rejected steps. Every sample contributes to the GP interpolation.

### Where's the catch?
Like every other sampler GPry isn't perfect and has some limitations:
- GPs don't scale well with the number of training samples as training the GP involves inverting a kernel matrix. Unfortunately the computational complexity of this inversion scales with the number of training samples cubed. This means that as the number of training samples required grows, the overhead of the algorithm increases considerably.
- While we tested GPry on mildly multimodal posterior distributions it is not a supported feature and should be used with caution. The algorithm is fairly greedy and can easily miss a mode, especially if the separation between modes is large.
- The algorithm is generally robust towards "weirdly" shaped posterior distributions, however the structure of the kernel still assumes a single characteristic correlation length. This means that in very pathological cases (for instance a very wide distribution with a tiny spike in it) GPry will struggle to capture the mode correctly.
- This code is novel and hasn't profited from years of user feedback and maintenance. Bugs can (and probably will) occur. If you find any please report them to us!

**We are actively working on mitigating some of those issues and we will keep on developing this code so look out for new versions!**
