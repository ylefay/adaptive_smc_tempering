import jax
import jax.numpy as jnp


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
    if dim > 1:
        C = jax.lax.select(i == 0, jnp.eye(dim), gamma ** 2 / dim * jnp.cov(particles.at[i - 1].get(), rowvar=False))
    else:
        C = jax.lax.select(i == 0, jnp.eye(dim), gamma ** 2 * jnp.array([jnp.var(particles.at[i - 1].get(), axis=0)]))

    def gaussian_rwmh_cov_log_proposal(x, y):
        return jax.scipy.stats.multivariate_normal.logpdf(y, x, C)

    def gaussian_rwmh_sampler(key, x):
        return jax.random.multivariate_normal(key, x, C)

    return gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler
