import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc.smc_types import SMCStatebis


def square_distance(x: ArrayLike, y: ArrayLike, _: SMCStatebis, __: int) -> ArrayLike:
    """
    Expected square jumping distance criterion: square distance between x and y.
    """
    return jnp.sum(jnp.square(x - y), axis=-1)


def mahalanobis(x: ArrayLike, y: ArrayLike, state: SMCStatebis, i: int) -> ArrayLike:
    """
    At iteration i, for particles x, and proposed particles y, compute the Mahalanobis distances between x and y.
    The scaling matrix is the estimated covariance of the particles at iteration i - 1.
    """
    particles = state.particles
    dim = particles.shape[-1]

    def _mahalanobis(x, y):
        if dim > 1:
            cov = jnp.cov(particles.at[i - 1].get().reshape((particles.shape[1] * particles.shape[2]),
                                                            dim), rowvar=False)
        else:
            cov = jnp.var(particles.at[i - 1].get().reshape((particles.shape[1] * particles.shape[2]),
                                                            dim), axis=0).reshape((1, 1))
        return jnp.einsum('j,k,jk->', x - y, x - y, jnp.linalg.inv(cov))

    return jax.lax.select(i == 0, jnp.sum(jnp.square(x - y), axis=-1), _mahalanobis(x, y))
