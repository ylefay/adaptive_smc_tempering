from typing import Callable

import blackjax
import jax
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

    def fixed_schedule_tempered_smc(self, key, init_particles, initial_parameter_value, num_mcmc_steps,
                                    lmbda_schedule=None, target_ess=None,
                                    mcmc_parameters={}):
        if lmbda_schedule is None:
            assert target_ess is not None, "Target ESS must be provided if no lambda schedule is provided."

        def mcmc_init_fn(position, logdensity_fn):
            return blackjax.mcmc.random_walk.init(position=position, logdensity_fn=logdensity_fn)

        if lmbda_schedule is None:
            alg_smc = blackjax.adaptive_tempered_smc
            extra_parameters = {'target_ess': target_ess}
        else:
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
        if lmbda_schedule is None:
            def body_fn(carry, lmbda):
                i, state = carry
                subkey = jax.random.fold_in(key, i)
                new_state, info = kernel(subkey, state, mcmc_parameters=mcmc_parameters)
                return (i + 1, new_state), (new_state, info)
        else:
            def body_fn(carry, lmbda):
                i, state = carry
                subkey = jax.random.fold_in(key, i)
                extra_parameters = {"lmbda": lmbda}
                new_state, info = kernel(subkey, state, **extra_parameters)
                return (i + 1, new_state), (new_state, info)

        (_, result), chain = jax.lax.scan(body_fn, (0, init_state), lmbda_schedule)
        return result, chain
