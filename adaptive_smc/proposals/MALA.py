from typing import Optional

import jax
import jax.numpy as jnp

from adaptive_smc.estimates import cov_estimate
from adaptive_smc.smc_types import LogDensity, SMCStatebis

__all__ = [
    "build_mala_proposal_gamma",
]


def build_mala_proposal_gamma(state: SMCStatebis, log_tgt_density_fn: LogDensity, _: LogDensity, i: int,
                              j: Optional[int] = None):
    """
    Metropolis Adjusted Langevin proposal with a gamma parameter
    """
    particles = state.particles
    log_weights = state.log_weights
    gamma = state.mh_proposal_parameters.at[i - 1].get()
    dim = particles.shape[-1]

    j = j or i

    def fun_to_be_called_if_j_greater_than_one():
        r"""
        Compute the covariance estimate of \pi_{t-1} given t\geq 1
        """
        particles_at_j_minus_one = particles.at[j - 1].get().reshape(-1, particles.shape[-1])
        log_weights_at_j_minus_one = log_weights.at[j - 1].get().reshape(-1, )
        weights_at_j_minus_one = jnp.exp(log_weights_at_j_minus_one)
        cov_hat = cov_estimate(particles_at_j_minus_one, weights_at_j_minus_one)
        return cov_hat

    cov_hat = fun_to_be_called_if_j_greater_than_one()

    def gaussian_mala_log_proposal(x, y):
        return jax.scipy.stats.multivariate_normal.logpdf(y, x + 0.5 * gamma ** 2 / dim ** (1 / 3) *
                                                          cov_hat @ jax.jacfwd(
            log_tgt_density_fn)(x),
                                                          gamma ** 2 / dim ** (1 / 3) * cov_hat)

    def gaussian_mala_sampler(key, x):
        return jax.random.multivariate_normal(key, x + 0.5 * gamma ** 2 / dim ** (1 / 3) *
                                              cov_hat @ jax.jacfwd(
            log_tgt_density_fn)(x),
                                              gamma ** 2 / dim ** (1 / 3) * cov_hat)

    return gaussian_mala_log_proposal, gaussian_mala_sampler, jnp.empty(1)
