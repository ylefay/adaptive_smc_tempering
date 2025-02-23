import jax.numpy as jnp

from adaptive_smc.experiments.GLOBAL import *
from adaptive_smc.problems.gaussian import create_sparse_problem


def construct_my_prior_and_target(tau):
    """
    The prior is a standard Gaussian distribution.
    The target is a Gaussian distribution N(1, C),
    where C is scaled identity on the latent space otherwise scaled down to tau**2.
    """

    """
    Take the log-likehood function such that the target is N(1, tau**2 * I)
    """

    loglikelihood_fn = create_sparse_problem(dim, latent_dim=dim // 4, mean=jnp.zeros(dim),
                                             scale=1 / (1 / tau ** 2 - 1))

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.zeros(dim), cov=jnp.eye(dim))

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn


tau = jnp.sqrt(0.1)
latent_dim = 0  # if set to 0, the target is N(1, tau**2 * I)
loglikelihood_fn, base_measure_sampler, logbase_density_fn = construct_my_prior_and_target(tau)
