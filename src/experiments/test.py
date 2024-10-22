import matplotlib.pyplot as plt
import numpy as np

from src.SMC.SMCTempering import TemperedSMC_MCMC
from src.experiments.my_logistic_problem_sonar import *
from src.proposals import gaussian_238_empirical_proposal

if __name__ == "__main__":
    OP_key = jax.random.PRNGKey(0)

    N_chains = 5
    num_particles = 40
    num_tempering_steps = 100
    num_mcmc_steps = 10

    keys = jax.random.split(OP_key, N_chains)

    log_scale_init = np.log(np.random.exponential(1, num_particles * dim)).reshape(num_particles, dim)
    initial_parameter_value = {'cov_particles': jnp.array([jnp.eye(dim)] * num_particles)}
    coeffs_init = np.random.randn(num_particles)
    lmbda_schedule = np.logspace(-5, 0, num_tempering_steps)

    kwargs_for_default_mcmc_kernel = {}

    init_particles = [log_scale_init]
    my_smc = TemperedSMC_MCMC(logprior_fn, loglikelihood_fn, dim,
                              build_kernel_and_mcmc_parameter_update_fn=gaussian_238_empirical_proposal,
                              kwargs=kwargs_for_default_mcmc_kernel)
    res, chain = jax.vmap(my_smc.fixed_schedule_tempered_smc, in_axes=(0, None, None, None, None))(keys, init_particles,
                                                                                                   initial_parameter_value,
                                                                                                   num_mcmc_steps,
                                                                                                   lmbda_schedule)
    plt.plot(chain[1].update_info.acceptance_rate.mean(axis=-1).mean(axis=-1).T)
    plt.savefig(f"{num_tempering_steps}_{num_particles}_{num_mcmc_steps}_acceptance_rate.png")
    plt.clf()
    print(res[0][0][0].mean(axis=1))
    print(res[0][0][0].mean(axis=1).mean(axis=0))
    plt.plot(res[0][0][0].mean(axis=1).mean(axis=0))
    plt.savefig(f"{num_tempering_steps}_{num_particles}_{num_mcmc_steps}_mean.png")
    plt.clf()
    for idx in range(len(res[0][0][0])):
        plt.plot(res[0][0][0][idx].mean(axis=0))
    plt.savefig(f"{num_tempering_steps}_{num_particles}_{num_mcmc_steps}_{len(res[0][0])}.png")
