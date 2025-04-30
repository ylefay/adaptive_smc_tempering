from typing import Tuple, Optional, Callable, Union

import jax
import jax.random
from jax import numpy as jnp, Array
from jax.typing import ArrayLike



def apply_vmap_batch(fun: Callable[[ArrayLike], ArrayLike], arg: ArrayLike, batch: Union[int, jnp.inf],
                     output_shape=()):
    """
    Given a vmappable function, apply it to arg in batch of size batch.
    """
    if arg.ndim == 1:
        shape = (arg.shape[0], *output_shape)
    else:
        shape = (*arg.shape[:-1], *output_shape)
    res = jnp.empty(shape)

    if batch > res.shape[0]:
        return fun(arg)

    def iter(i, res):
        my_arg = jax.lax.dynamic_slice(arg, (i * batch, *(0,) * (arg.ndim - 1)), (batch, *arg.shape[1:]))
        _res = fun(my_arg)
        res = jax.lax.dynamic_update_slice(res, _res, (i * batch, *(0, ) * (res.ndim - 1)))
        return res

    res = jax.lax.cond(res.shape[0] // batch > 0,
                       lambda _: jax.lax.fori_loop(0, res.shape[0] // batch, iter, res),
                       lambda _: res, None)
    res = res.at[res.shape[0] // batch * batch:].set(fun(arg.at[res.shape[0] // batch * batch:].get()))
    return res

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
