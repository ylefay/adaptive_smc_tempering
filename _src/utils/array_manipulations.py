from typing import Dict, Tuple, Union

import jax.numpy as jnp
from blackjax.mcmc.random_walk import RWInfo, RWState
from blackjax.smc import tempered
from blackjax.smc.base import SMCInfo
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from jax.typing import ArrayLike

from src.SMC.online_waste_free.mcmc.random_walk import RWInfo as RWInfoWithProposedState


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


def RWState_to_array(state: RWState) -> jnp.ndarray:
    position = state.position[0]
    logdensity = jnp.expand_dims(state.logdensity, -1)
    return jnp.concatenate((position, logdensity), axis=-1)


def take_first_of_extra_parameters(extra_parameters: Dict) -> Dict:
    """
    Take a dict. of array like and return the first element of each array.
    -------

    """
    extra_parameters_copy = extra_parameters.copy()
    for k, v in extra_parameters.items():
        extra_parameters_copy[k] = v[0, ...]
    return extra_parameters_copy

def from_RWinfo_to_array(_info: Union[RWInfo, RWInfoWithProposedState]) -> jnp.ndarray:
    proposal_as_array = RWState_to_array(_info.proposal)
    proposed_states_as_array = RWState_to_array(_info.proposed_states)
    return jnp.array([proposal_as_array, proposed_states_as_array])


def repeat(a: ArrayLike, n: int) -> ArrayLike:
    return jnp.repeat(a[jnp.newaxis, ...], n, axis=0)
