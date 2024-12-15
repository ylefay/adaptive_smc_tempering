import jax
# from problems.artificial_logistic import create_problem
import jax.numpy as jnp
import jax.random

from adaptative_smc.problems.gaussian import create_problem
from adaptative_smc.proposals import build_gaussian_rwmh_cov_proposal_gamma
from adaptative_smc.smc import GenericAdaptiveWasteFreeTemperingSMC

jax.config.update("jax_enable_x64", True)


def test():
    """
    Checking that the samples from the posterior distribution given by a
        - Gaussian prior N(m_P, C_P)
        - Likelihood N(m_L, C_L),
    obtained via SMC, have empirical mean and covariance approximately satisfying
        C^{-1} = C_P^{-1} + C_L^{-1},
        C^{-1}m = C_P^{-1}m_P + C_L^{-1}m_L.
    """
    OP_key = jax.random.PRNGKey(0)
    dim = 2

    mean_likelihood = jnp.ones(dim) * 20
    cov_likelihood = jnp.eye(dim) * 0.5 ** 2
    loglikelihood_fn = create_problem(dim, mean=mean_likelihood, cov=cov_likelihood)
    mean_prior = jnp.ones(dim) * 5
    cov_prior = 5 * jnp.eye(dim)

    @jax.vmap
    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, mean_prior, cov_prior)

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=mean_prior, cov=cov_prior)

    length_of_the_tempering_sequence = 1000
    my_tempering_sequence = jnp.linspace(0, 1, length_of_the_tempering_sequence)

    optimization_method_str = "None"
    optimization_method = None

    num_parallel_chain = 5000
    num_mcmc_steps = 10
    init_param = jnp.array([2.38])
    n_chains = 4

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               build_gaussian_rwmh_cov_proposal_gamma, optimization_method)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence)

    keys = jax.random.split(OP_key, n_chains)
    with jax.default_device(jax.devices("cpu")[0]):
        res = wrapper_smc(keys)

    n_particles = res[0].shape[2] * res[0].shape[3]
    cov = jax.vmap(lambda X: jnp.cov(X, rowvar=False))(
        res[0][:, -1].reshape((*res[0][:, -1].shape[:1], n_particles, res[0].shape[-1])))
    mean = res[0][:, -1].mean(axis=[1, 2])

    rtol = 5 * 1e-2
    jax.vmap(lambda X: jnp.allclose(X, jnp.linalg.inv(cov_prior) @ mean_prior + jnp.linalg.inv(
        cov_likelihood) @ mean_likelihood, rtol=rtol))(jax.vmap(lambda X, Y: X@Y)(jnp.linalg.inv(cov), mean)).all()

    jax.vmap(lambda X: jnp.allclose(X, jnp.linalg.inv(cov_prior) + jnp.linalg.inv(cov_likelihood), rtol=rtol))(
        jnp.linalg.inv(cov)).all()
