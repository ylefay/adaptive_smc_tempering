import jax.numpy as jnp
from jax.typing import ArrayLike


def cov_estimate(particles: ArrayLike, weights: ArrayLike) -> ArrayLike:
    r"""
    Given particles and weights at iteration t, of shapes resp. (N, dim), and (N, ),
    the weighted covariance estimate of \pi_t is
    (N - 1)^{-1}\sum_{1\leq i\leq N} W_{t}^{i} (X^i_t - \hat{\mu}_t) (X^i_t - \hat{\mu}_t)^{\top},
    where \hat{\mu}_t = N^{-1}\sum_{1\leq i\leq N} X^i_t
    """
    N = particles.shape[0]
    mu_hat = jnp.sum(weights[:, jnp.newaxis] * particles, axis=0) / N
    to_sum = (particles - mu_hat)
    cov_hat = 1 / (N - 1) * jnp.einsum('ij,ik,i->jk', to_sum, to_sum, weights)
    """if particles.shape[-1]>1:
        cov_hat = jnp.cov(particles, rowvar=False)
    else:
        cov_hat = jnp.var(particles).reshape((1, 1))"""
    return cov_hat

