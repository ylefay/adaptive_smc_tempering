import os
from datetime import datetime

import jax.random

from adaptive_smc import optimise
from adaptive_smc import proposals
from adaptive_smc.experiments_bis.gaussians.problem import *
from adaptive_smc.save_and_read_and_postprocess import save
from adaptive_smc.smc_bis import GenericAdaptiveWasteFreeTemperingSMC


def default_title():
    now = datetime.now()

    output_path = f"{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


def experiment_ar(keys):
    optimization_method_str = "make_optimize_within_a_fixed_grid"
    params_optimization_method = {"grid": jnp.linspace(0, 0.99, 100)}
    # params_optimization_method = {"minmax": [0.1, 10.], "interval": [-5., 5.], "n_iter":4}

    init_param = jnp.array([0])
    config = {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
              "proposal": "build_autoregressive_gaussian_proposal_with_cov_estimate",
              "dim": dim, "tempering_sequence": my_tempering_sequence,
              "num_parallel_chain": num_parallel_chain, "num_mcmc_steps": num_mcmc_steps, "init_param": init_param,
              "n_chains": n_chains,
              "target_ess": target_ess,
              "tau": tau}
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
    save(res, config, OUTPUT_PATH + default_title())


def experiment_rwmh(keys):
    optimization_method_str = "make_optimize_within_a_fixed_grid"
    params_optimization_method = {"grid": jnp.linspace(0.01, 5, 200)}
    # params_optimization_method = {}
    # params_optimization_method = {"minmax": [0.1, 10.], "interval": [-5., 5.], "n_iter":4}

    init_param = jnp.array([2.38])
    config = {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
              "proposal": "build_gaussian_rwmh_cov_proposal_gamma",
              "dim": dim, "tempering_sequence": my_tempering_sequence,
              "num_parallel_chain": num_parallel_chain, "num_mcmc_steps": num_mcmc_steps, "init_param": init_param,
              "n_chains": n_chains,
              "target_ess": target_ess,
              "tau": tau}
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
    save(res, config, OUTPUT_PATH + default_title())


if __name__ == "__main__":
    for keys in all_keys:
        experiment_ar(keys)
        experiment_rwmh(keys)
