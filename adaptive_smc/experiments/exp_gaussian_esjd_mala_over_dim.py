import os
from datetime import datetime

import jax
# from problems.artificial_logistic import create_problem
import jax.numpy as jnp
import jax.random


from adaptive_smc import optimise
from adaptive_smc import proposals
from adaptive_smc.problems.gaussian import create_problem
from adaptive_smc.smc import GenericAdaptiveWasteFreeTemperingSMC
from adaptive_smc.utils import save

jax.config.update("jax_enable_x64", True)


def default_title():
    now = datetime.now()

    output_path = f"{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


def experiment(dim: int):
    OP_key = jax.random.PRNGKey(0)

    loglikelihood_fn = create_problem(dim, scale=jnp.sqrt(0.5))

    length_of_the_tempering_sequence = 50
    my_tempering_sequence = jnp.linspace(0, 1, length_of_the_tempering_sequence)

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim) + 20, 5 * jnp.eye(dim))

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.zeros(dim) + 20, cov=5 * jnp.eye(dim))

    optimization_method_str = "make_optimize_within_a_fixed_grid"
    params_optimization_method = {"grid": jnp.linspace(1, 5, 100)}
    # params_optimization_method = {"minmax": [0.1, 10.], "interval": [-5., 5.], "n_iter":4}

    num_parallel_chain = 4000
    num_mcmc_steps = 5
    init_param = jnp.array([1.310])
    n_chains = 5
    config = {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
              "proposal": "build_mala_proposal_gamma",
              "dim": dim, "tempering_sequence": my_tempering_sequence,
              "num_parallel_chain": num_parallel_chain, "num_mcmc_steps": num_mcmc_steps, "init_param": init_param,
              "n_chains": n_chains}
    my_proposal = getattr(proposals, config['proposal'])
    if config['optimization_method']:
        optimization_method = getattr(optimise, config['optimization_method'])(**params_optimization_method)
    else:
        optimization_method = None

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               my_proposal, optimization_method)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, None)

    keys = jax.random.split(OP_key, n_chains)
    with jax.disable_jit(False):
        with jax.default_device(jax.devices("cpu")[0]):
            with jax.debug_nans(False):
                res = wrapper_smc(keys)
    save(res, config, ['optimization_method', 'dim', 'init_param', 'num_parallel_chain', 'num_mcmc_steps'],
         [length_of_the_tempering_sequence],
         default_title())


if __name__ == "__main__":
    dims = [1, 2, 3, 5, 10]
    for d in dims:
        experiment(d)
