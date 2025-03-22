import jax.numpy as jnp
import jax.random

from adaptive_smc.experiments_bis.GLOBAL import *
from adaptive_smc.problems.gaussian import create_sparse_problem


def construct_my_prior_and_target():
    r"""
    The prior is a standard Gaussian distribution.
    The target is a Gaussian distribution 0.1N(0, C) + 0.9N(0, C'),
    where C is scaled identity on the latent space otherwise scaled down to tau**2,
    same for C' with a different scaling factor. Orthogonal directions.
    """

    r"""
    Take the log-likehood function such that the target is correct.
    """

    logpdf1 = create_sparse_problem(dim, latent_dim=latent_dim, mean=jnp.zeros(dim),
                                    scale=1 / (1 / tau ** 2))
    logpdf2 = create_sparse_problem(dim, latent_dim=latent_dim, mean=jnp.zeros(dim),
                                    scale=1 / (1 / tau2 ** 2))

    def loglikelihood_fn(x):

        log_tgt_distrib = jnp.log(0.1 * jnp.exp(logpdf1(x)) + 0.9 * jnp.exp(logpdf2(x)))
        log_prior = jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.zeros(dim),
                                                               cov = jnp.eye(dim))
        ll = log_tgt_distrib - log_prior
        return ll

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.zeros(dim), cov=jnp.eye(dim))

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn


tau = jnp.sqrt(0.3)
tau2 = jnp.sqrt(0.6)
latent_dim = 0  # if set to 0, the target is N(1, tau**2 * I) \beta + (1-\beta) N(0, tau'**2 I)
loglikelihood_fn, base_measure_sampler, logbase_density_fn = construct_my_prior_and_target()
length_of_the_tempering_sequence = 10 + dim
my_tempering_sequence = jnp.linspace(0, 1, length_of_the_tempering_sequence)
