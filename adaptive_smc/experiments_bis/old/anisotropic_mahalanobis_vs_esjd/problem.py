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
    loadings = loadings.at[:latent_dim].set(corr_coeff)  # strong loading for variable 0
    loadings = loadings.at[latent_dim:].set(corr_coeff * 0.5)  # slightly lower loadings for others

    # Factor covariance (1-dimensional latent factor)
    factor_cov = jnp.array([[1.0]])

    # Covariance = loadings * factor_cov * loadings.T + diag(noise)
    Sigma = loadings @ factor_cov @ loadings.T + jnp.eye(dim)
    jax.debug.print("{Sigma}", Sigma=Sigma)
    scale = jax.random.uniform(key, shape=(dim - latent_dim,), minval=0.5, maxval=2)
    loglikelihood_fn = create_sparse_problem(dim, latent_dim=latent_dim, scale=scale, cov=Sigma)

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



def construct_my_prior_and_target_t_allcomponents(config):
    r"""
    the target density is a product of t distrib

    """

    config_problem = config.get('problem')
    dim = config.get('dim')
    freedom = jnp.array(config_problem.get('freedom', jnp.arange(3, 3+dim, 1))[:dim])


    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.zeros(dim), cov=jnp.eye(dim))

    def loglikelihood_fn(x):
        return jnp.sum(jax.scipy.stats.t.logpdf(x, df=freedom)) - logbase_density_fn(x)

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn
