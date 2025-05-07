import jax.numpy as jnp
import jax.scipy.stats.t

from adaptive_smc.experiments_bis.GLOBAL import *
from adaptive_smc.problems.gaussian import create_sparse_problem


def construct_my_prior_and_target_gaussian(config):
    r"""
    The prior is a standard Gaussian distribution.
    The target is a Gaussian distribution N(1, C),
    where C is scaled identity on the latent space otherwise scaled down to tau**2.
    """

    r"""
    Take the log-likehood function such that the target is N((1, ..., 1) (or zero?), C),
    with C = (1, ..., 1, \tau^2, ..., \tau^2)
    """

    config_problem = config.get('problem')
    tau = config_problem.get('tau')
    dim = config.get('dim')
    latent_dim = config_problem.get('latent_dim')

    loglikelihood_fn = create_sparse_problem(dim, latent_dim=latent_dim, mean=jnp.zeros(dim),
                                             scale=1 / (1 / tau ** 2 - 1))

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.ones(dim), cov=jnp.eye(dim))

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
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.ones(dim), cov=jnp.eye(dim))

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn
