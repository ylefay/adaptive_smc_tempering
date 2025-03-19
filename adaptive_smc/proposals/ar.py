import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc.estimates import cov_increment_estimate, cov_estimate
from adaptive_smc.smc_types import SMCStatebis
from adaptive_smc.smc_types import LogDensity

__all__ = ["build_build_autoregressive_gaussian_proposal",
           "build_autoregressive_gaussian_proposal",
           "build_autoregressive_gaussian_proposal_with_nicolas_cov_estimate",
           "build_autoregressive_gaussian_proposal_with_cov_estimate"
           ]


def build_build_autoregressive_gaussian_proposal(mu: ArrayLike, C: ArrayLike):
    r"""
    Construct the build function for autoregressive proposal:
    q(y\mid x) = N(mu + \rho (x-mu), (1-\rho^2)C),
    where C is a given matrix, and mu given vector.
    """

    def _build(state: SMCStatebis, _: LogDensity, __: LogDensity, i):
        rho = state.mh_proposal_parameters.at[i - 1].get()

        def gaussian_ar_log_proposal(x, y):
            return jax.scipy.stats.multivariate_normal.logpdf(y, mu + rho * (x-mu), (1 - rho ** 2) * C)

        def gaussian_ar_sampler(key, x):
            return jax.random.multivariate_normal(key, mu + rho * (x-mu), (1 - rho ** 2) * C)

        return gaussian_ar_log_proposal, gaussian_ar_sampler, jnp.empty(1)

    return _build


def build_autoregressive_gaussian_proposal(state: SMCStatebis, log_tgt_density_fn: LogDensity,
                                           log_likelihood_fn: LogDensity, i: int):
    r"""
    Autoregressive proposal:
    q(y\mid x) = N(\rho x, (1-\rho^2)C),
    where C is a fixed matrix, here I_n.
    """
    dim = state.particles.shape[-1]
    C = jnp.eye(dim)
    mu = jnp.zeros(dim)
    return build_build_autoregressive_gaussian_proposal(mu, C)(state, log_tgt_density_fn, log_likelihood_fn, i)


def build_autoregressive_gaussian_proposal_with_nicolas_cov_estimate(state: SMCStatebis, log_tgt_density_fn: LogDensity,
                                                                     log_likelihood_fn: LogDensity, i: int):
    r"""
    Autoregressive proposal:
    q(y\mid x) = N(\rho (x-mu) + mu, (1-\rho^2)C),
    where mu, C is estimated using the particles and weights at iteration i-1,
    using the covariance increment estimate proposed by Nicolas.
    Should we target the covariance estimate of \pi_{t-1} or \pi_t?
    Targeting \pi_t does not work.
    """
    particles = state.particles
    log_weights = state.log_weights
    previous_cov = state.others.at[i - 1, :-1, :].get() # assuming row stacking of cov and mu.
    previous_mu = state.others.at[i-1, -1].get()
    dlmbda = jax.lax.cond(i > 1,
                          lambda _: state.tempering_sequence.at[i - 1].get() - state.tempering_sequence.at[i - 2].get(),
                          lambda _: state.tempering_sequence.at[0].get(), None)  # setting \lambda_{-1} = 0.

    # dlmbda = state.tempering_sequence.at[i].get() - state.tempering_sequence.at[i - 1].get()
    def fun_to_be_called_if_i_greater_than_one():
        r"""
        Should we target the covariance estimate of \pi_{t-1} or \pi_t?
        Compute the covariance estimate of \pi_{t} given t\geq 1 as proposed by Nicolas
        """
        particles_at_i_minus_one = particles.at[i - 1].get().reshape(-1, particles.shape[-1])
        # log_weights_at_i_minus_one = log_weights.at[i - 1].get().reshape(-1, )  # approximate well \pi_{t-2}
        # weights_at_i_minus_one = jnp.exp(log_weights_at_i_minus_one)
        no_weights = jnp.ones((log_weights.shape[1] * log_weights.shape[2]))
        dcov, dmu = cov_increment_estimate(particles_at_i_minus_one, no_weights,
                                                        dlmbda, log_likelihood_fn)
        new_cov = previous_cov + dcov
        new_mu = previous_mu + dmu
        return new_cov, new_mu

    C, mu = fun_to_be_called_if_i_greater_than_one()
    proposal, sampler, _ = build_build_autoregressive_gaussian_proposal(mu, C)(state, log_tgt_density_fn, log_likelihood_fn,
                                                                           i)
    return proposal, sampler, C


def build_autoregressive_gaussian_proposal_with_cov_estimate(state: SMCStatebis, log_tgt_density_fn: LogDensity,
                                                             log_likelihood_fn: LogDensity, i: int):
    particles = state.particles
    log_weights = state.log_weights

    def fun_to_be_called_if_i_greater_than_one():
        r"""
        Compute the covariance estimate of \pi_{t-1} given t\geq 1
        """
        particles_at_i_minus_one = particles.at[i - 1].get().reshape(-1, particles.shape[-1])
        log_weights_at_i_minus_one = log_weights.at[i - 1].get().reshape(-1, )
        weights_at_i_minus_one = jnp.exp(log_weights_at_i_minus_one)
        cov_hat, mu_hat = cov_estimate(particles_at_i_minus_one, weights_at_i_minus_one)
        return cov_hat, mu_hat

    C, mu = fun_to_be_called_if_i_greater_than_one()
    return build_build_autoregressive_gaussian_proposal(mu, C)(state, log_tgt_density_fn, log_likelihood_fn,
                                                           i)
