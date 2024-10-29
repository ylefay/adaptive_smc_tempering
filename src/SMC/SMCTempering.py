from typing import Callable, Dict

import blackjax
import jax
import jax.numpy as jnp
from blackjax.smc import inner_kernel_tuning
from blackjax.smc import tempered
from blackjax.smc.resampling import multinomial

from src.proposals import mcmc_proposal


class TemperedSMC_MCMC():

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
                                         num_mcmc_steps: int, target_ess: float, num_tempering_steps: int):
        alg_smc = blackjax.adaptive_tempered_smc
        extra_parameters = {'target_ess': target_ess}

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
            i, state = carry
            subkey = jax.random.fold_in(key, i)
            extra_parameters = {}
            new_state, info = kernel(subkey, state, **extra_parameters)
            return (i + 1, new_state), (new_state, info)

        (_, result), chain = jax.lax.scan(body_fn, (0, init_state), jnp.zeros(num_tempering_steps))
        return result, chain

    def fixed_schedule_tempered_smc(self, key: jax.Array, init_particles: jnp.ndarray, initial_parameter_value: Dict,
                                    num_mcmc_steps: int, lmbda_schedule: jnp.ndarray):
        def mcmc_init_fn(position, logdensity_fn):
            return blackjax.mcmc.random_walk.init(position=position, logdensity_fn=logdensity_fn)

        alg_smc = blackjax.tempered_smc
        extra_parameters = {}

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
            i, state = carry
            subkey = jax.random.fold_in(key, i)
            extra_parameters = {"lmbda": lmbda}
            new_state, info = kernel(subkey, state, **extra_parameters)
            return (i + 1, new_state), (new_state, info)

        (_, result), chain = jax.lax.scan(body_fn, (0, init_state), lmbda_schedule)
        return result, chain
