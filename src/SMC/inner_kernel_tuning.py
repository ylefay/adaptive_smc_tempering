from typing import Callable, Dict, Tuple

import blackjax.smc.inner_kernel_tuning as _inner_kernel_tuning
from blackjax.smc.base import SMCInfo, SMCState
from blackjax.types import ArrayTree, PRNGKey
from jax.typing import ArrayLike

StateWithParameterOverride = _inner_kernel_tuning.StateWithParameterOverride
init = _inner_kernel_tuning.init


def build_kernel(
        smc_algorithm,
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        mcmc_step_fn: Callable,
        mcmc_init_fn: Callable,
        resampling_fn: Callable,
        mcmc_parameter_update_fn: Callable[[SMCState, ArrayLike, int, SMCInfo], Dict[str, ArrayTree]],
        num_mcmc_steps: int = 10,
        **extra_parameters,
) -> Callable:
    """
    Similar implementation to blackjax.smc.inner_kernel_tuning.build_kernel, but with the addition of the previous particles.

    Same documentation as blackjax.smc.inner_kernel_tuning.build_kernel documentation.

    Parameters
    ----------

    smc_algorithm
        Either src.SMC.smc.adaptive_tempered_smc or src.SMC.smc.tempered_smc (or any other implementation of
        a sampling algorithm that returns an SMCState and SMCInfo pair and handles previous states as argument).
    mcmc_step_fn:
        The transition kernel, should take as parameters the dictionary output of mcmc_parameter_update_fn.
        mcmc_step_fn(rng_key, state, cumul_states, tempered_logposterior_fn, **mcmc_parameter_update_fn())

    mcmc_parameter_update_fn
        A callable that takes the SMCState, an Array containg SMCState from the previous iterations, SMCInfo at step i and constructs a parameter to be used by the inner kernel in i+1 iteration.
    """

    def kernel(
            rng_key: PRNGKey, state: StateWithParameterOverride, cumul_states: ArrayLike, i: int,
            **extra_step_parameters
    ) -> Tuple[StateWithParameterOverride, SMCInfo]:
        step_fn = smc_algorithm(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=mcmc_step_fn,
            mcmc_init_fn=mcmc_init_fn,
            mcmc_parameters=state.parameter_override,
            resampling_fn=resampling_fn,
            num_mcmc_steps=num_mcmc_steps,
            **extra_parameters,
        ).step
        new_state, info = step_fn(rng_key, state.sampler_state, **extra_step_parameters)
        new_parameter_override = mcmc_parameter_update_fn(new_state, cumul_states, i, info)
        return StateWithParameterOverride(new_state, new_parameter_override), info

    return kernel
