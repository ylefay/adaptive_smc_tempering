import jax.random

from utils import save
from problems.my_logistic_problem_sonar import *
from smc import GenericAdaptiveWasteFreeTemperingSMC
from datetime import datetime
import os

jax.config.update("jax_enable_x64", True)

logbase_density_fn = logprior_fn
length_of_the_tempering_sequence = 5
my_tempering_sequence = jnp.linspace(0, 1, length_of_the_tempering_sequence)

@jax.vmap
def base_measure_sampler(key):
    return jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))


def build_autoregressive_gaussian_rwmh_proposal(rho, _, __):
    C = jnp.eye(dim)

    def gaussian_rwmh_log_proposal(x, y):
        return jax.scipy.stats.multivariate_normal.logpdf(y, rho * x, (1 - rho ** 2) * C)

    def gaussian_rwmh_sampler(key, x):
        return jax.random.multivariate_normal(key, rho * x, (1 - rho ** 2) * C)

    return gaussian_rwmh_log_proposal, gaussian_rwmh_sampler


def build_gaussian_rwmh_cov_proposal(_, particles, i):
    dim = particles.shape[-1]
    particles = particles.reshape(particles.shape[0], -1, particles.shape[-1])
    C = jax.lax.select(i == 0, jnp.eye(dim), 2.38 ** 2 / dim * jnp.cov(particles.at[i - 1].get(), rowvar=False))

    def gaussian_rwmh_cov_log_proposal(x, y):
        return jax.scipy.stats.multivariate_normal.logpdf(y, x, C)

    def gaussian_rwmh_sampler(key, x):
        return jax.random.multivariate_normal(key, x, C)

    return gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler

def build_gaussian_rwmh_cov_proposal_gamma(gamma, particles, i):
    dim = particles.shape[-1]
    particles = particles.reshape(particles.shape[0], -1, particles.shape[-1])
    C = jax.lax.select(i == 0, jnp.eye(dim), gamma ** 2 / dim * jnp.cov(particles.at[i - 1].get(), rowvar=False))

    def gaussian_rwmh_cov_log_proposal(x, y):
        return jax.scipy.stats.multivariate_normal.logpdf(y, x, C)

    def gaussian_rwmh_sampler(key, x):
        return jax.random.multivariate_normal(key, x, C)

    return gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler

def default_title():
    now = datetime.now()

    output_path = f"{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


if __name__ == "__main__":
    OP_key = jax.random.PRNGKey(0)
    smc = GenericAdaptiveWasteFreeTemperingSMC(logprior_fn, base_measure_sampler, loglikelihood_fn,
                                               build_gaussian_rwmh_cov_proposal)
    smc = GenericAdaptiveWasteFreeTemperingSMC(logprior_fn, base_measure_sampler, loglikelihood_fn,
                                               build_gaussian_rwmh_cov_proposal_gamma)
    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, 2000, 100, jnp.array([2.38]), my_tempering_sequence)

    n_chains = 2
    keys = jax.random.split(OP_key, n_chains)
    with jax.disable_jit(False):
        res = wrapper_smc(keys)
    save(res, default_title())


