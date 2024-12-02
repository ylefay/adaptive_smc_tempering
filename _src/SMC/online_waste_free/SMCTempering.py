from functools import partial
from typing import Callable, Dict, Optional

import blackjax
import jax
import jax.numpy as jnp
from blackjax.smc import tempered
from blackjax.smc.resampling import multinomial
from blackjax.smc.waste_free import update_waste_free

from src.SMC.online_waste_free import inner_kernel_tuning
from src.SMC.online_waste_free.inner_kernel_tuning import get_shape_of_update_info
from src.proposals import mcmc_proposal
from src.utils.array_manipulations import repeat, temperedsmcstate_to_array, from_RWinfo_to_array

__all__ = [
    "TemperedSMC",
    "WasteFreeTemperedSMC"
]


class TemperedSMC:

    def __init__(self, logprior_fn: Callable, loglikelihood_fn: Callable, dim: int,
                 build_kernel_and_mcmc_parameter_update_fn: mcmc_proposal, **kwargs):
        build_kernel, mcmc_parameter_update_fn = build_kernel_and_mcmc_parameter_update_fn
        self.logprior_fn = logprior_fn
        self.loglikelihood_fn = loglikelihood_fn
        self.dim = dim
        self.resample_fn = multinomial
        self.mcmc_step_fn = build_kernel
        self.mcmc_parameter_update_fn = mcmc_parameter_update_fn

    def adaptative_schedule_tempered_smc(self, key: jax.Array, init_particles: jnp.ndarray,
                                         initial_parameter_value: Dict,
                                         num_mcmc_steps: Optional[int], target_ess: float, num_tempering_steps: int,
                                         extra_parameters: Dict):
        alg_smc = blackjax.adaptive_tempered_smc
        extra_parameters_update = {'target_ess': target_ess}
        extra_parameters.update(extra_parameters_update)

        def mcmc_init_fn(position, logdensity_fn):
            return blackjax.mcmc.random_walk.init(position=position, logdensity_fn=logdensity_fn)

        kernel = inner_kernel_tuning.build_kernel(
            alg_smc,
            self.logprior_fn,
            self.loglikelihood_fn,
            self.mcmc_step_fn,
            mcmc_init_fn,
            self.resample_fn,
            self.mcmc_parameter_update_fn,
            num_mcmc_steps=num_mcmc_steps,
            **extra_parameters)
        init_state = inner_kernel_tuning.init(tempered.init, init_particles, initial_parameter_value)

        def body_fn(carry, _):
            i, state, cumul_states, cumul_infos, cumul_ancestors = carry
            subkey = jax.random.fold_in(key, i)
            extra_parameters_step = {}
            new_state, info = kernel(subkey, state, cumul_states, cumul_infos, cumul_ancestors, i, **extra_parameters_step)
            cumul_states = cumul_states.at[i].set(temperedsmcstate_to_array(new_state))
            cumul_infos = cumul_infos.at[i].set(from_RWinfo_to_array(info.update_info))
            cumul_ancestors = cumul_ancestors.at[i].set(info.ancestors)
            return (i + 1, new_state, cumul_states, cumul_infos, cumul_ancestors), (new_state, info)

        init_cumul_states = repeat(temperedsmcstate_to_array(init_state), num_tempering_steps)

        # Constructing a skeleton of the SMCInfo object to be used as initialisation
        shape_of_update_info = get_shape_of_update_info(alg_smc,
                                                        init_state,
                                                        self.logprior_fn,
                                                        self.loglikelihood_fn,
                                                        self.mcmc_step_fn,
                                                        mcmc_init_fn,
                                                        self.resample_fn,
                                                        extra_parameters,
                                                        {},
                                                        num_mcmc_steps,
                                                        )
        init_cumul_infos = jnp.zeros((num_tempering_steps, *shape_of_update_info))
        init_cumul_ancestors = jnp.zeros((num_tempering_steps, shape_of_update_info[1])) # shape_of_update_info[1] is the number of particles going in a parallel manner through the Markov kernels,i.e, M
        (_, result, _, _, _), chain = jax.lax.scan(body_fn, (
        0, init_state, init_cumul_states, init_cumul_infos, init_cumul_ancestors), jnp.zeros(num_tempering_steps))
        return result, chain

    def fixed_schedule_tempered_smc(self, key: jax.Array, init_particles: jnp.ndarray, initial_parameter_value: Dict,
                                    num_mcmc_steps: Optional[int], lmbda_schedule: jnp.ndarray,
                                    extra_parameters: Dict):
        num_tempering_steps = lmbda_schedule.shape[0]

        def mcmc_init_fn(position, logdensity_fn):
            return blackjax.mcmc.random_walk.init(position=position, logdensity_fn=logdensity_fn)

        alg_smc = blackjax.tempered_smc
        extra_parameters_update = {}
        extra_parameters.update(extra_parameters_update)

        kernel = inner_kernel_tuning.build_kernel(
            alg_smc,
            self.logprior_fn,
            self.loglikelihood_fn,
            self.mcmc_step_fn,
            mcmc_init_fn,
            self.resample_fn,
            self.mcmc_parameter_update_fn,
            num_mcmc_steps=num_mcmc_steps,
            **extra_parameters)
        init_state = inner_kernel_tuning.init(tempered.init, init_particles, initial_parameter_value)

        def body_fn(carry, lmbda):
            i, state, cumul_states, cumul_infos, cumul_ancestors = carry
            subkey = jax.random.fold_in(key, i)
            extra_parameters_step = {"lmbda": lmbda}
            new_state, info = kernel(subkey, state, cumul_states, cumul_infos, cumul_ancestors, i, **extra_parameters_step)
            cumul_states = cumul_states.at[i].set(temperedsmcstate_to_array(new_state))
            cumul_infos = cumul_infos.at[i].set(from_RWinfo_to_array(info.update_info))
            cumul_ancestors = cumul_ancestors.at[i].set(info.ancestors)
            return (i + 1, new_state, cumul_states, cumul_infos, cumul_ancestors), (new_state, info)

        init_cumul_states = repeat(temperedsmcstate_to_array(init_state), num_tempering_steps)

        # Constructing a skeleton of the SMCInfo object to be used as initialisation
        shape_of_update_info = get_shape_of_update_info(alg_smc,
                                                        init_state,
                                                        self.logprior_fn,
                                                        self.loglikelihood_fn,
                                                        self.mcmc_step_fn,
                                                        mcmc_init_fn,
                                                        self.resample_fn,
                                                        extra_parameters,
                                                        {},
                                                        num_mcmc_steps,
                                                        )
        init_cumul_infos = jnp.zeros((num_tempering_steps, *shape_of_update_info))
        init_cumul_ancestors = jnp.zeros((num_tempering_steps, shape_of_update_info[1])) # shape_of_update_info[1] is the number of particles going in a parallel manner through the Markov kernels,i.e, M
        (_, result, _, _, _), chain = jax.lax.scan(body_fn, (
            0, init_state, init_cumul_states, init_cumul_infos, init_cumul_ancestors), jnp.zeros(num_tempering_steps))
        return result, chain


class WasteFreeTemperedSMC(TemperedSMC):
    """
    Waste-free implementation
    Hai-Dang Dau, Nicolas Chopin. Waste-Free Sequential Monte Carlo. Journal of the Royal Statistical
    Society: Series B, 2022, 84 (1), pp.114-148. ff10.1111/rssb.12475ff. ffhal-04273259f
    """

    def __init__(self, logprior_fn: Callable, loglikelihood_fn: Callable, dim: int,
                 build_kernel_and_mcmc_parameter_update_fn: mcmc_proposal, **kwargs):
        super().__init__(logprior_fn, loglikelihood_fn, dim, build_kernel_and_mcmc_parameter_update_fn,
                         **kwargs)

    def adaptative_schedule_tempered_smc(self, key: jax.Array, init_particles: jnp.ndarray,
                                         initial_parameter_value: Dict, num_mcmc_steps: int, target_ess: float,
                                         num_tempering_steps: int, extra_parameters: Dict):
        p = num_mcmc_steps + 1
        num_resampled = int(init_particles[0].shape[0] / p)
        extra_parameters_update = {'update_strategy': partial(update_waste_free, p=p, num_resampled=num_resampled)}
        extra_parameters.update(extra_parameters_update)

        return super().adaptative_schedule_tempered_smc(key, init_particles, initial_parameter_value, None, target_ess,
                                                        num_tempering_steps, extra_parameters)

    def fixed_schedule_tempered_smc(self, key: jax.Array, init_particles: jnp.ndarray, initial_parameter_value: Dict,
                                    num_mcmc_steps: int, lmbda_schedule: jnp.ndarray, extra_parameters):
        p = num_mcmc_steps + 1
        num_resampled = int(init_particles[0].shape[0] / p)
        extra_parameters_update = {'update_strategy': partial(update_waste_free, p=p, num_resampled=num_resampled)}
        extra_parameters.update(extra_parameters_update)
        return super().fixed_schedule_tempered_smc(key, init_particles, initial_parameter_value, None, lmbda_schedule,
                                                   extra_parameters)
