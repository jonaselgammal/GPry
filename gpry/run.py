"""
Top level run file which constructs the loop for mapping a posterior
distribution.
"""


def run(prior, likelihood, gp=None, gp_acquisition=None,
        convergence_criterion=None, callback=None,
        time_likelihood=True):
    """
    This function takes care of constructing the Bayesian quadrature/likelihood
    characterization loop. This is the easiest way to make use of the
    gpry algorithm. The minimum requirements for running this are a Cobaya
    prior object and a likelihood. Furthermore the details of the GP and
    and acquisition can be specified by the user.

    Parameters
    ----------

    prior : Cobaya `prior object <https://cobaya.readthedocs.io/en/latest/params_prior.html>`
        Contains all information about the parameters in the likelihood and
        their priors.

    likelihood : function
        The likelihood function. Should contain all non-derived input variables
        defined in the prior. The likelihood multiplied with the prior gives
        the posterior distribution.

    gp: GaussianProcessRegressor, optional (default=None)
        If

    Returns
    -------

    KL_divergence : The value of the KL divergence between the two models.
    If the KL divergence cannot be determined ``None`` is returned.
    """
