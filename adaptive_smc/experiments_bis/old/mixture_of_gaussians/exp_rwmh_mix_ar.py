import os
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random
import yaml

from adaptive_smc import optimise
from adaptive_smc import proposals
from adaptive_smc.experiments_bis.mixture_of_gaussians.problem import construct_my_prior_and_target
from adaptive_smc.save_and_read_and_postprocess import save
from adaptive_smc.smc_bis import GenericAdaptiveWasteFreeTemperingSMC

OP_key = jax.random.PRNGKey(0)
_, key = jax.random.split(OP_key)

def default_title(prefix=''):
    now = datetime.now()

    output_path = f"{prefix}_{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


def experiment_mixture_ar_rwm(config, keys):
    dim = config.get('dim')

    target_ess = config.get('target_ess')
    num_parallel_chain = config.get('num_parallel_chain')
    num_mcmc_steps = config.get('num_mcmc_steps')

    optimization_method_str = "make_optimize_within_a_fixed_grid"

    tempering_length = config.get('tempering_length', 10 + dim)
    my_tempering_sequence = jnp.linspace(0, 1, tempering_length)

    beta_grid = jnp.linspace(0, 1, 50)  # beta_grid = jnp.array([0.5])
    gamma_grid = jnp.linspace(1, 4, 10)
    rho_grid = jnp.linspace(0, 0.99, 20)
    new_grid = jnp.array([[x, y, z] for x in beta_grid for y in gamma_grid for z in rho_grid])
    params_optimization_method = {"grid": new_grid}

    loglikelihood_fn, base_measure_sampler, logbase_density_fn = construct_my_prior_and_target(config)

    init_param = jnp.array([0.5, 2.38, 0.])

    config.update(
        {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
         "proposal": "build_build_mixture_ar_rwm",
         "tempering_sequence": my_tempering_sequence,
         "init_param": init_param})

    my_proposal = getattr(proposals, config['proposal'])(jnp.zeros(dim), jnp.eye(dim))

    if config['optimization_method']:
        optimization_method = getattr(optimise, config['optimization_method'])(**params_optimization_method)
    else:
        optimization_method = None

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               my_proposal, optimization_method)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess)

    res = wrapper_smc(keys)
    save(res, config, config.get('OUTPUT_PATH') + default_title(config.get('prefix')))


def experiment_ar(config, keys):
    dim = config.get('dim')

    target_ess = config.get('target_ess')
    num_parallel_chain = config.get('num_parallel_chain')
    num_mcmc_steps = config.get('num_mcmc_steps')

    optimization_method_str = "make_optimize_within_a_fixed_grid"

    tempering_length = config.get('tempering_length', 10 + dim)
    my_tempering_sequence = jnp.linspace(0, 1, tempering_length)

    loglikelihood_fn, base_measure_sampler, logbase_density_fn = construct_my_prior_and_target(config)

    rho_grid = jnp.linspace(0, 0.99, 20)
    params_optimization_method = {"grid": rho_grid}

    init_param = jnp.array([0.])
    config.update(
        {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
         "proposal": "build_build_autoregressive_gaussian_proposal",
         "tempering_sequence": my_tempering_sequence,
         "init_param": init_param})

    my_proposal = getattr(proposals, config['proposal'])(jnp.zeros(dim), jnp.eye(dim))

    if config['optimization_method']:
        optimization_method = getattr(optimise, config['optimization_method'])(**params_optimization_method)
    else:
        optimization_method = None

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               my_proposal, optimization_method)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess)

    res = wrapper_smc(keys)
    save(res, config, config.get('OUTPUT_PATH') + default_title(config.get('prefix')))


def experiment_rwm(config, keys):
    dim = config.get('dim')

    target_ess = config.get('target_ess')
    num_parallel_chain = config.get('num_parallel_chain')
    num_mcmc_steps = config.get('num_mcmc_steps')

    loglikelihood_fn, base_measure_sampler, logbase_density_fn = construct_my_prior_and_target(config)

    optimization_method_str = "make_optimize_within_a_fixed_grid"
    params_optimization_method = {"grid": jnp.linspace(1.0, 4, 500)}

    tempering_length = config.get('tempering_length', 10 + dim)
    my_tempering_sequence = jnp.linspace(0, 1, tempering_length)

    init_param = jnp.array([2.38])

    config.update(
        {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
         "proposal": "build_gaussian_rwmh_cov_proposal_gamma",
         "tempering_sequence": my_tempering_sequence,
         "init_param": init_param})

    my_proposal = getattr(proposals, config['proposal'])

    if config['optimization_method']:
        optimization_method = getattr(optimise, config['optimization_method'])(**params_optimization_method)
    else:
        optimization_method = None

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               my_proposal, optimization_method)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess)

    res = wrapper_smc(keys)
    save(res, config, config.get('OUTPUT_PATH') + default_title(config['prefix']))


if __name__ == "__main__":
    yaml_file = "./exp_rwmh_mix_ar.yaml"
    with open(yaml_file, "r") as file:
        y_config = yaml.load(file, Loader=yaml.FullLoader)
    for name_of_my_config, config in y_config.items():
        if config.get('run', True):
            sequential_repetitions = config.pop('sequential_repetitions', 1)
            parallel_repetitions = config.get('parallel_repetitions')
            seq_keys = jax.random.split(key, sequential_repetitions)
            all_keys = jax.vmap(lambda k: jax.random.split(k, parallel_repetitions))(seq_keys)
            _, key = jax.random.split(seq_keys.at[-1].get())
            for keys in all_keys:
                experiment_mixture_ar_rwm(config, keys)

    yaml_file = "./exp_rwmh_mix_ar.yaml"
    with open(yaml_file, "r") as file:
        y_config = yaml.load(file, Loader=yaml.FullLoader)
    for name_of_my_config, config in y_config.items():
        if config.get('run', True):
            sequential_repetitions = config.pop('sequential_repetitions', 1)
            parallel_repetitions = config.get('parallel_repetitions')
            seq_keys = jax.random.split(key, sequential_repetitions)
            all_keys = jax.vmap(lambda k: jax.random.split(k, parallel_repetitions))(seq_keys)
            _, key = jax.random.split(seq_keys.at[-1].get())
            for keys in all_keys:
                experiment_ar(config, keys)

    yaml_file = "./exp_rwmh_mix_ar.yaml"
    with open(yaml_file, "r") as file:
        y_config = yaml.load(file, Loader=yaml.FullLoader)[0]
    for name_of_my_config, config in y_config.items():
        if config.get('run', True):
            sequential_repetitions = config.pop('sequential_repetitions', 1)
            parallel_repetitions = config.get('parallel_repetitions')
            seq_keys = jax.random.split(key, sequential_repetitions)
            all_keys = jax.vmap(lambda k: jax.random.split(k, parallel_repetitions))(seq_keys)
            _, key = jax.random.split(seq_keys.at[-1].get())
            for keys in all_keys:
                experiment_rwm(config, keys)
