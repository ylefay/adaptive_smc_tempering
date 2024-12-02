import jax.random
import matplotlib.pyplot as plt
from src.problems.my_logistic_problem_sonar import *
from src.smc import GenericAdaptiveWasteFreeTemperingSMC

logbase_density_fn = logprior_fn
length_of_the_tempering_sequence = 100
my_tempering_sequence = jnp.linspace(0, 1, length_of_the_tempering_sequence)


@jax.vmap
def base_measure_sampler(key):
    return jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))


def build_autoregressive_gaussian_rwmh_log_proposal(rho, _, __):
    C = jnp.eye(dim)

    def gaussian_rwmh_log_proposal(x, y):
        return - 1 / 2 * (y - rho * x).T @ jnp.linalg.inv((1 - rho ** 2) * C) @ (y - rho * x)

    return gaussian_rwmh_log_proposal


def build_autoregressive_gaussian_rwmh_sampler(rho, _, __):
    C = jnp.eye(dim)

    def gaussian_rwmh_sampler(key, x):
        return jax.random.multivariate_normal(key, rho * x, (1 - rho ** 2) * C)

    return gaussian_rwmh_sampler


def build_gaussian_rwmh_cov_log_proposal(_, particles, i):
    dim = particles.shape[-1]
    particles = particles.reshape(particles.shape[0], -1, particles.shape[-1])
    C = jax.lax.select(i == 0, jnp.eye(dim), 2.38 / dim * jnp.cov(particles.at[i].get(), rowvar=False))

    def gaussian_rwmh_cov_log_proposal(x, y):
        return -1 / 2 * (y - x).T @ jnp.linalg.inv(C) @ (y - x)

    return gaussian_rwmh_cov_log_proposal


def build_autoregressive_gaussian_rwmh_sampler(_, particles, i):
    dim = particles.shape[-1]
    particles = particles.reshape(particles.shape[0], -1, particles.shape[-1])
    C = jax.lax.select(i == 0, jnp.eye(dim), 2.38 / dim * jnp.cov(particles.at[i].get(), rowvar=False))

    def gaussian_rwmh_sampler(key, x):
        return jax.random.multivariate_normal(key, x, C)

    return gaussian_rwmh_sampler


if __name__ == "__main__":
    OP_key = jax.random.PRNGKey(0)
    """smc = GenericAdaptiveWasteFreeTemperingSMC(logprior_fn, base_measure_sampler, loglikelihood_fn,
                                               build_autoregressive_gaussian_rwmh_log_proposal,
                                               build_autoregressive_gaussian_rwmh_sampler)"""
    smc = GenericAdaptiveWasteFreeTemperingSMC(logprior_fn, base_measure_sampler, loglikelihood_fn,
                                               build_gaussian_rwmh_cov_log_proposal,
                                               build_autoregressive_gaussian_rwmh_sampler)
    with jax.disable_jit(False):
        res = smc.sample(OP_key, 200, 3, jnp.array([0.]), my_tempering_sequence)
    print(res)
    plt.plot(res[0][-1].mean(axis=0).mean(axis=0))
    plt.plot(res[0][0].mean(axis=0).mean(axis=0))

    plt.show()