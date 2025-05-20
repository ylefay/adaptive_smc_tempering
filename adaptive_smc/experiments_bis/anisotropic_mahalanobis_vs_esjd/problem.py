import jax.numpy as jnp
import jax.scipy.stats.t

from adaptive_smc.experiments_bis.GLOBAL import *
from adaptive_smc.problems.gaussian import create_sparse_problem


def construct_my_prior_and_target_gaussian(key, config):
    r"""
    The prior is a standard Gaussian distribution.
    The target is a Gaussian distribution N(1, C),
    where C is scaled identity on the latent space otherwise scaled down to tau**2.
    """


    config_problem = config.get('problem')
    dim = config.get('dim')
    corr_coeff = config_problem.get('corr_coeff', 0.9)
    latent_dim = config_problem.get('latent_dim', 0)
    key, subkey = jax.random.split(key)
    loadings = jax.random.normal(subkey, shape=(dim, 1))
    loadings = loadings.at[0].set(corr_coeff)  # strong loading for variable 0
    loadings = loadings.at[1:].set(corr_coeff * 0.95)  # slightly lower loadings for others

    # Factor covariance (1-dimensional latent factor)
    factor_cov = jnp.array([[1.0]])

    # Covariance = loadings * factor_cov * loadings.T + diag(noise)
    Sigma = loadings @ factor_cov @ loadings.T + jnp.eye(dim)
    jax.debug.print(Sigma)
    loglikelihood_fn = create_sparse_problem(dim, latent_dim, cov=Sigma)

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.zeros(dim), cov=jnp.eye(dim))

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn


def construct_my_prior_and_target_t_student(config):
    r"""
    The prior is a standard Gaussian distribution.
    the likelihood is a student t distrib on the last component.

    """

    config_problem = config.get('problem')
    freedom = config_problem.get('freedom')
    dim = config.get('dim')

    loglikelihood_fn = lambda x: jax.scipy.stats.t.logpdf(x.at[-1].get(), df=freedom)

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.zeros(dim), cov=jnp.eye(dim))

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn
