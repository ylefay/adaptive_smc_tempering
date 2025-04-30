from typing import Tuple, Optional

import jax
import jax.random
from jax import numpy as jnp, Array
from jax.typing import ArrayLike


def normalize_log_weights(log_weights: ArrayLike) -> Tuple[ArrayLike, float]:
    """
    Normalize the log weights \exp \log w_i and return the log normalization constant:
        \log N^{-1}\sum_{i=1}^N \exp(\log w_i)
    """
    log_normalization = jax.scipy.special.logsumexp(log_weights)
    return log_weights - log_normalization, log_normalization - jnp.log(log_weights.shape[0] * log_weights.shape[1])


def log_ess(delta: float, log_weights: Array) -> float:
    """
    See Algorithm 17.3,
    Introduction to Sequential Monte Carlo, Chopin, Papaspiliopoulos
    """
    N_particles = jnp.prod(jnp.array(log_weights.shape))
    log_ess = 2 * jax.scipy.special.logsumexp(delta * log_weights) - jax.scipy.special.logsumexp(
        2 * delta * log_weights)
    log_ess_scaled = log_ess - jnp.log(N_particles)
    return log_ess_scaled



def vec(X: ArrayLike) -> Array:
    """
    Vectorization of a matrix.
    """
    return X.reshape(-1, order='F')

def unvec(vecX: ArrayLike, shape=Optional[Tuple[int, int]]) -> Array:
    """
    Invert the previous operation
    """
    if shape is None:
        shape = (int(vecX.shape[0] ** 0.5), int(vecX.shape[0] ** 0.5))
    return vecX.reshape(shape, order='F')
