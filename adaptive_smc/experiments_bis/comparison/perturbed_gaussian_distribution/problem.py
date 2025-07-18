import jax.numpy as jnp

from adaptive_smc.experiments_bis.GLOBAL import *


def construct_my_prior_and_target(config):
    r"""
    The prior is a standard Gaussian distribution.
    The target is a perturbed Gaussian distribution N(0, \tau^2 I)\ prod_{i} \exp(-x_i^3 \exp(-\abs{x_i}^2)),
    """

    config_problem = config.get('problem')
    tau = config_problem.get('tau')
    dim = config.get('dim')
    beta = config_problem.get('beta', 1.0)
    loglikelihood_fn = lambda x: - 0.5 * x.T @ x * (1 / tau ** 2 - 1) - beta * jnp.sum(
        x ** 3 * jnp.exp(-jnp.abs(x) ** 2),
        axis=-1)

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.zeros(dim), cov=jnp.eye(dim))

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn
