import jax.numpy as jnp
import jax
import os
from datetime import datetime

import jax.random

from adaptive_smc import optimise
from adaptive_smc import proposals
from adaptive_smc.experiments_bis.mixture_of_gaussians.problem import *
from adaptive_smc.save_and_read_and_postprocess import save
from adaptive_smc.smc_bis import GenericAdaptiveWasteFreeTemperingSMC


def default_title():
    now = datetime.now()

    output_path = f"{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path

def experiment_mixture_ar_rwm(keys):
    optimization_method_str = "make_optimize_within_a_fixed_grid"
    beta_grid = jnp.linspace(0, 1, 10)
    gamma_grid = jnp.linspace(0, 5, 100)
    rho_grid = jnp.linspace(0, 1, 100)
    new_grid = jnp.array([[x, y, z] for x in beta_grid for y in gamma_grid for z in rho_grid])
    params_optimization_method = {"grid": new_grid}


    init_param = jnp.array([0.5, 2.38, 0.])
    config = {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
              "proposal": "build_build_mixture_ar_rwm",
              "dim": dim, "tempering_sequence": my_tempering_sequence,
              "num_parallel_chain": num_parallel_chain, "num_mcmc_steps": num_mcmc_steps, "init_param": init_param,
              "n_chains": n_chains,
              "target_ess": target_ess,
              "tau": tau}
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
    save(res, config, OUTPUT_PATH + default_title())


if __name__ == "__main__":
    for keys in all_keys:
        experiment_mixture_ar_rwm(keys)

