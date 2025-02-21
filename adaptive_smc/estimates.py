import jax.numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc.smc_types import LogDensity


def cov_estimate(particles: ArrayLike, weights: ArrayLike) -> ArrayLike:
    r"""
    Given particles and weights at iteration t, of shapes resp. (N, dim), and (N, ),
    the weighted covariance estimate of \pi_t is
    \sum_{1\leq i\leq N} W_{t}^{i} (X^i_t - \hat{\mu}_t) (X^i_t - \hat{\mu}_t)^{\top},
    where \hat{\mu}_t = \sum_{1\leq i\leq N} X^i_t w^i_t
    """
    mu_hat = jnp.sum(weights[:, jnp.newaxis] * particles, axis=0)
    to_sum = (particles - mu_hat)
    cov_hat = jnp.einsum('ij,ik,i->jk', to_sum, to_sum, weights)
    return cov_hat


def cov_increment_estimate(particles: ArrayLike, weights: ArrayLike, dlambda: ArrayLike,
                           log_likelihood_fn: LogDensity) -> ArrayLike:
    r"""
    At iteration t+1, compute the estimate of the first-order increment of the covariance of \pi_{t+1}
    using samples and weights from iteration t.
    The increment is equal to
        \dlambda\times \hat{\bbE}_t[(X-\hat{\bbE}_t(X))(X-\hat{\bbE}_t(X))^{\top} . (s-\hat{\bbE}(s))],
    where s is the log-likelihood function, and $\hat{\bbE}_t is the weighted mean operator at iteration t
    targeting expectations under \pi_t (i.e., X_t approx. \sim \pi_{t-1})

    """
    weights = weights / jnp.sum(weights)  # ensuring normalised weights
    likelihoods = log_likelihood_fn(particles)
    mean_log_likelihood = jnp.mean(weights * likelihoods)
    mu_hat = jnp.sum(weights[:, jnp.newaxis] * particles, axis=0)
    to_sum = (particles - mu_hat)
    cov_hat = jnp.einsum('ij,ik,i->jk', to_sum, to_sum, weights)
    stat_cov = to_sum.reshape(to_sum.shape + (1,)) @ jnp.swapaxes(to_sum.reshape(to_sum.shape + (1,)), 1, 2) - cov_hat
    stat_dcov = stat_cov * (likelihoods - mean_log_likelihood)[:, jnp.newaxis, jnp.newaxis]
    dcov = jnp.sum(weights[:, jnp.newaxis, jnp.newaxis] * stat_dcov, axis=0)
    return dlambda * dcov
