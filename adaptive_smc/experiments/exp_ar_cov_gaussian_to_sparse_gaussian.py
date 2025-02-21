import os
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random

from adaptive_smc import optimise
from adaptive_smc import proposals
from adaptive_smc.problems.gaussian import create_sparse_problem
from adaptive_smc.save_and_read_and_postprocess import save
from adaptive_smc.smc import GenericAdaptiveWasteFreeTemperingSMC

jax.config.update("jax_enable_x64", False)
OP_key = jax.random.PRNGKey(0)


def default_title():
    now = datetime.now()

    output_path = f"{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


def construct_my_prior_and_target(dim, tau):
    """
    The prior is a standard Gaussian distribution.
    The target is a Gaussian distribution N(0, C),
    with C sparse (appr. 3/4 of the covariance entries are scaled down)
    """

    """
    Take the log-likehood function such that the target is N(0, C),
    """
    loglikelihood_fn = create_sparse_problem(dim, latent_dim=dim // 4, mean=jnp.zeros(dim),
                                             scale=jnp.eye(dim) * 1 / (1 / tau ** 2 - 1))

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
    init_other = jnp.eye(dim)

    config = {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
              "proposal": "build_autoregressive_gaussian_proposal_with_nicolas_cov_estimate",
              "dim": dim, "tempering_sequence": my_tempering_sequence,
              "num_parallel_chain": num_parallel_chain, "num_mcmc_steps": num_mcmc_steps, "init_param": init_param,
              "n_chains": n_chains,
              "target_ess": target_ess,
              "other": init_other,
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
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess,
                          init_other)

    keys = jax.random.split(OP_key, n_chains)
    with jax.disable_jit(False):
        with jax.default_device(jax.devices("cpu")[0]):
            res = wrapper_smc(keys)
    save(res, config, default_title())


if __name__ == "__main__":
    num_parallel_chain = 4
    num_mcmc_steps = 4000
    n_chains = 5
    target_ess = 0.5

    dims = [10]
    taus = jnp.sqrt(jnp.array([0.1]))

    for tau in taus:
        for d in dims:
            experiment_ar(d, tau)
