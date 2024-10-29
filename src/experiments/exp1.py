import os

import numpy as np

from src.SMC.SMCTempering import TemperedSMC_MCMC
from src.problems.my_logistic_problem_sonar import *
from src.proposals import gaussian_238_empirical_proposal
from src.utils.save import plot, save

from datetime import datetime

def default_title():
    now = datetime.now()

    output_path = f"{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


if __name__ == "__main__":
    OP_key = jax.random.PRNGKey(0)

    N_chains = 2
    num_particles = 500
    num_tempering_steps = 100
    num_mcmc_steps = 1

    keys = jax.random.split(OP_key, N_chains)

    initial_parameter_value = {'cov_particles': jnp.array([jnp.eye(dim)] * num_particles)}
    lmbda_schedule = np.logspace(-5, 0, num_tempering_steps)
    kwargs_for_default_mcmc_kernel = {}

    config = {'num_particles': num_particles, 'num_tempering_steps': num_tempering_steps,
              'num_mcmc_steps': num_mcmc_steps,
              'params_proposal': None,
              'kwargs_for_default_mcmc_kernel': kwargs_for_default_mcmc_kernel,
              'lmbda_schedule': lmbda_schedule,
              'target_ESS': None,
              'initial_parameter_value': initial_parameter_value,
              'kernel_id': 'gaussian_238_empirical_proposal',
              'file': os.path.basename(__file__)}

    my_smc = TemperedSMC_MCMC(logprior_fn, loglikelihood_fn, dim,
                              build_kernel_and_mcmc_parameter_update_fn=gaussian_238_empirical_proposal(),
                              kwargs=kwargs_for_default_mcmc_kernel)


    @jax.vmap
    def wrapped_smc(key):
        log_scale_init = jnp.log(jax.random.exponential(key, shape=(num_particles, dim)))
        init_particles = [log_scale_init]
        return my_smc.fixed_schedule_tempered_smc(key, init_particles, initial_parameter_value, num_mcmc_steps,
                                                  lmbda_schedule)


    res, chain = wrapped_smc(keys)
    save(chain, f"{os.path.basename(__file__)}_SONAR_G238_{num_tempering_steps}_{num_particles}_{num_mcmc_steps}",
         config, output_path=default_title())
    plot(chain, f"{os.path.basename(__file__)}_SONAR_G238_{num_tempering_steps}_{num_particles}_{num_mcmc_steps}")
