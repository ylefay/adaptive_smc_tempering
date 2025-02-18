import os
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random

from adaptive_smc import optimise
from adaptive_smc import proposals
from adaptive_smc.problems.gaussian import create_problem
from adaptive_smc.smc import GenericAdaptiveWasteFreeTemperingSMC
from adaptive_smc.utils import save

jax.config.update("jax_enable_x64", False)
OP_key = jax.random.PRNGKey(0)


def default_title():
    now = datetime.now()

    output_path = f"{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


def construct_my_prior_and_target(dim, tau):
    """
    The prior is a standard Gaussian distribution.
    The target is a Gaussian distribution N(1, tau**2 I)
    """

    """
    Take the log-likehood function such that the target is N(1, tau**2 * I)
    """
    loglikelihood_fn = create_problem(dim, mean=jnp.ones(dim), cov=jnp.eye(dim) * 1 / (1 / tau ** 2 - 1))

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.zeros(dim), cov=jnp.eye(dim))

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn


def experiment_ar(dim: int, tau: float):
    loglikelihood_fn, base_measure_sampler, logbase_density_fn = construct_my_prior_and_target(dim, tau)

    length_of_the_tempering_sequence = 30 + dim
    my_tempering_sequence = jnp.linspace(0, 1, length_of_the_tempering_sequence)

    optimization_method_str = "make_optimize_within_a_fixed_grid"
    params_optimization_method = {"grid": jnp.linspace(0, 0.99, 100)}
    # params_optimization_method = {"minmax": [0.1, 10.], "interval": [-5., 5.], "n_iter":4}

    init_param = jnp.array([0])
    config = {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
              "proposal": "build_autoregressive_gaussian_rwmh_proposal",
              "dim": dim, "tempering_sequence": my_tempering_sequence,
              "num_parallel_chain": num_parallel_chain, "num_mcmc_steps": num_mcmc_steps, "init_param": init_param,
              "n_chains": n_chains,
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
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, 0.9)

    keys = jax.random.split(OP_key, n_chains)
    with jax.disable_jit(False):
        with jax.default_device(jax.devices("cpu")[0]):
            with jax.debug_nans(False):
                res = wrapper_smc(keys)
    save(res, config, ['optimization_method', 'dim', 'init_param', 'num_parallel_chain', 'num_mcmc_steps'],
         [length_of_the_tempering_sequence],
         default_title())


def experiment_rwmh(dim: int, tau: float):
    loglikelihood_fn, base_measure_sampler, logbase_density_fn = construct_my_prior_and_target(dim, tau)

    length_of_the_tempering_sequence = 30 + dim
    my_tempering_sequence = jnp.linspace(0, 1, length_of_the_tempering_sequence)

    optimization_method_str = "make_optimize_within_a_fixed_grid"
    params_optimization_method = {"grid": jnp.linspace(1, 5, 100)}
    # params_optimization_method = {}
    # params_optimization_method = {"minmax": [0.1, 10.], "interval": [-5., 5.], "n_iter":4}

    init_param = jnp.array([2.38])
    config = {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
              "proposal": "build_gaussian_rwmh_cov_proposal_gamma",
              "dim": dim, "tempering_sequence": my_tempering_sequence,
              "num_parallel_chain": num_parallel_chain, "num_mcmc_steps": num_mcmc_steps, "init_param": init_param,
              "n_chains": n_chains,
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
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, 0.9)

    keys = jax.random.split(OP_key, n_chains)
    with jax.disable_jit(False):
        with jax.default_device(jax.devices("cpu")[0]):
            with jax.debug_nans(False):
                res = wrapper_smc(keys)
    save(res, config, ['optimization_method', 'dim', 'init_param', 'num_parallel_chain', 'num_mcmc_steps'],
         [length_of_the_tempering_sequence],
         default_title())


if __name__ == "__main__":
    num_parallel_chain = 4
    num_mcmc_steps = 1000
    n_chains = 5

    dims = [2]
    taus = jnp.sqrt(jnp.array([0.1]))

    for tau in taus:
        for d in dims:
            experiment_rwmh(d, tau)

    for tau in taus:
        for d in dims:
            experiment_ar(d, tau)
