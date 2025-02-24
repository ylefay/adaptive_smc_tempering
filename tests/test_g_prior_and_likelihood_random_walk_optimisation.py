from functools import partial

import jax
# from problems.artificial_logistic import create_problem
import jax.numpy as jnp
import jax.random

from adaptive_smc.optimise import make_optimize_within_a_fixed_grid
from adaptive_smc.problems.gaussian import create_problem
from adaptive_smc.proposals import build_gaussian_rwmh_cov_proposal_gamma
from adaptive_smc.smc import GenericAdaptiveWasteFreeTemperingSMC

jax.config.update("jax_enable_x64", False)


def test():
    r"""
    Checking that the samples from the tempered posterior distribution given by a temperature \lambda,
        - Gaussian prior N(m_P, C_P)
        - Likelihood N(m_L, C_L),
    defined as  N(m_P, C_P) \times N(m_L, C_L)^{\lambda}, and obtained via SMC,
    have empirical mean m and covariance C approximately satisfying
        C^{-1} = C_P^{-1} + \lambda C_L^{-1},
        C^{-1}m = C_P^{-1}m_P + \lambda C_L^{-1}m_L.
    To check the first equality, we multiply C_P^{-1} + \lambda C_L^{-1} by C, and compute the distance to the identity,
        absolute tolerance of 5 % * sqrt(dim).
    Component-wise relative tolerance up to 5% for the second inequality.
    Checking this for all intermediate temperatures.
    """
    OP_key = jax.random.PRNGKey(0)
    dim = 2

    mean_likelihood = jnp.ones(dim) * 20
    cov_likelihood = jnp.eye(dim) * 0.5 ** 2
    loglikelihood_fn = create_problem(dim, mean=mean_likelihood, cov=cov_likelihood)
    mean_prior = jnp.ones(dim) * 5
    cov_prior = 5 * jnp.eye(dim)

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, mean_prior, cov_prior)

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=mean_prior, cov=cov_prior)

    length_of_the_tempering_sequence = 25
    my_tempering_sequence = jnp.linspace(0, 1, length_of_the_tempering_sequence)

    num_parallel_chain = 10
    num_mcmc_steps = 20000
    init_param = jnp.array([2.38])
    n_chains = 2

    optimization_method = make_optimize_within_a_fixed_grid(jnp.linspace(0.01, 5, 200))

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               build_gaussian_rwmh_cov_proposal_gamma, optimization_method)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess=0.5)

    keys = jax.random.split(OP_key, n_chains)
    with jax.disable_jit(False):
        with jax.default_device(jax.devices("cpu")[0]):
            res = wrapper_smc(keys)

    n_particles = res[0].shape[2] * res[0].shape[3]
    temperatures = res[6]
    temperatures = jnp.insert(temperatures, 0, 0., -1)
    assert jnp.all(temperatures[:, -1] == 1.0)  # assert all temperatures at the end are 1.0

    @partial(jax.vmap, in_axes=(0, None, None, None, None))
    def get_mean_var(lmbda, mean_prior, mean_ll, cov_prior, cov_ll):
        r"""
        Compute the mean and var of a Gaussian distribution of the form
            \calN(mean_prior, var_prior) \times \calN(mean_ll, var_ll)^{\lmbda}

        Parameters
        ----------
        lmbda: temperature
        """
        cov = jnp.linalg.inv(jnp.linalg.inv(cov_prior) + jnp.linalg.inv(cov_ll) * lmbda)
        mean = cov @ (jnp.linalg.inv(cov_prior) @ mean_prior + jnp.linalg.inv(cov_ll) @ mean_ll * lmbda)
        return mean, cov

    rtol = 5e-2
    """
    When the current distribution is the target (T = -1), 
    We are sure the optimal parameter is 2.38.
    """
    assert jnp.all(jnp.allclose(res[2][:, -1], jnp.array([2.38]), rtol=rtol))
