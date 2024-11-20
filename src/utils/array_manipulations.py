import jax.numpy as jnp
from blackjax.smc import tempered
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from jax.typing import ArrayLike

def temperedsmcstate_to_array(state: tempered.TemperedSMCState) -> jnp.ndarray:
    if isinstance(state, tempered.TemperedSMCState):
        particles = state[0][0]
        weights = state[1]
    elif isinstance(state, StateWithParameterOverride):
        state = state[0]
        particles = state[0][0]
        weights = state[1]
    else:
        particles = state[0][0][0]
        weights = state[0][1]
    weights = jnp.expand_dims(weights, axis=1)
    return jnp.concatenate((particles, weights), axis=-1)

def repeat(a: ArrayLike, n: int) -> ArrayLike:
    return jnp.repeat(a[jnp.newaxis, ...], n, axis=0)