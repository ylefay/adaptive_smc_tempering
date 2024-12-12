import os
from datetime import datetime
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random
from jax.typing import ArrayLike
import optimise
from problems.artificial_logistic import create_problem
from proposals import build_gaussian_rwmh_cov_proposal_gamma
from smc import GenericAdaptiveWasteFreeTemperingSMC
from utils import save
from problems.my_logistic_problem_sonar import *

jax.config.update("jax_enable_x64", True)


def default_title():
    now = datetime.now()

    output_path = f"{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


if __name__ == "__main__":
    OP_key = jax.random.PRNGKey(0)
    """
    dim = 2
    C = jax.random.multivariate_normal(jax.random.PRNGKey(0), jnp.zeros(dim), jnp.eye(dim), shape=(dim,))
    mu = jax.random.multivariate_normal(jax.random.PRNGKey(0), jnp.ones(dim), jnp.eye(dim))
    loglikelihood_fn, logprior_fn = create_problem(jax.random.PRNGKey(0), mu, C @ C.T / dim, 1000)
    logbase_density_fn = logprior_fn
    """

    length_of_the_tempering_sequence = 10
    my_tempering_sequence = jnp.linspace(0, 1, length_of_the_tempering_sequence)


    @jax.vmap
    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))


    def build_autoregressive_gaussian_rwmh_proposal(rho, _, __):
        C = jnp.eye(dim)

        def gaussian_rwmh_log_proposal(x, y):
            return jax.scipy.stats.multivariate_normal.logpdf(y, rho * x, (1 - rho ** 2) * C)

        def gaussian_rwmh_sampler(key, x):
            return jax.random.multivariate_normal(key, rho * x, (1 - rho ** 2) * C)

        return gaussian_rwmh_log_proposal, gaussian_rwmh_sampler


    #optimization_method_str = "optimize_within_a_grid"
    #optimization_method = optimise.make_optimize_within_a_grid([1, 4], [-0.1, 0.1], 20)

    optimization_method_str = "None"
    optimization_method = None

    num_parallel_chain = 1000
    num_mcmc_steps = 3
    init_param = jnp.array([2.38])
    n_chains = 1
    config = {"optimization_method": optimization_method_str, "dim": dim, "tempering_sequence": my_tempering_sequence,
              "num_parallel_chain": num_parallel_chain, "num_mcmc_steps": num_mcmc_steps, "init_param": init_param,
              "n_chains": n_chains}

    smc = GenericAdaptiveWasteFreeTemperingSMC(logprior_fn, base_measure_sampler, loglikelihood_fn,
                                               build_gaussian_rwmh_cov_proposal_gamma, optimization_method)


    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence)


    keys = jax.random.split(OP_key, n_chains)
    with jax.disable_jit(False):
        with jax.default_device(jax.devices("cpu")[0]):
            with jax.debug_nans(False):
                res = wrapper_smc(keys)
    save(res, config, ['optimization_method', 'dim', 'init_param', 'num_parallel_chain', 'num_mcmc_steps'],
         default_title())
