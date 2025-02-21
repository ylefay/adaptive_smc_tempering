import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc.smc import SMCState
from adaptive_smc.smc_types import LogDensity
from adaptive_smc.estimates import cov_estimate

__all__ = [
    "build_gaussian_rw_proposal",
    "build_gaussian_rwmh_cov_proposal",
    "build_gaussian_rwmh_cov_proposal_gamma",
]


def build_gaussian_rw_proposal(C: ArrayLike):
    """
    Gaussian RW with fixed covariance matrix C
    """

    def gaussian_rwmh_cov_log_proposal(x, y):
        return jax.scipy.stats.multivariate_normal.logpdf(y, x, C)

    def gaussian_rwmh_sampler(key, x):
        return jax.random.multivariate_normal(key, x, C)

    return gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler, jnp.empty(1)


def build_gaussian_rwmh_cov_proposal(state: SMCState, log_tgt_density_fn: LogDensity, log_likelihood_fn: LogDensity,
                                     i: int):
    """
    Adaptative RWMH kernels with scaling set to the optimal asymptotic scaling, i.e. 2.38^2/dim.
    See Optimal scaling for various Metropolis-Hastings algorithms, Gareth O. Roberts and Jeffrey S. Rosenthal
    """
    state.mh_proposal_parameters = state.mh_proposal_parameters.at[i - 1].set(2.38)
    return build_gaussian_rwmh_cov_proposal_gamma(state, log_tgt_density_fn, log_likelihood_fn, i)


def build_gaussian_rwmh_cov_proposal_gamma(state: SMCState, _: LogDensity, __: LogDensity, i: int):
    """
    Same as build_gaussian_rwmh_cov_proposal with gamma**2/dim in front of the covariance matrix
    """
    gamma = state.mh_proposal_parameters.at[i - 1].get()
    particles = state.particles
    dim = particles.shape[-1]
    log_weights = state.log_weights
    optimal_scale = gamma ** 2 / dim

    def fun_to_be_called_if_i_greater_than_one():
        r"""
        Compute the covariance estimate of \pi_{t-1} given t\geq 1
        """
        particles_at_i_minus_one = particles.at[i - 1].get().reshape(-1, particles.shape[-1])
        log_weights_at_i_minus_one = log_weights.at[i - 1].get().reshape(-1, )
        weights_at_i_minus_one = jnp.exp(log_weights_at_i_minus_one)
        cov_hat = cov_estimate(particles_at_i_minus_one, weights_at_i_minus_one)
        return cov_hat

    C = optimal_scale * fun_to_be_called_if_i_greater_than_one()

    gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler, _ = build_gaussian_rw_proposal(C)

    return gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler, jnp.empty(1)
