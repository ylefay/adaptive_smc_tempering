import os
from datetime import datetime

import jax.random
import numpy as np

from src.SMC.SMCTempering import TemperedSMC_MCMC
from src.problems.my_logistic_problem_sonar import *
from src.proposals import cranck_nicholson_proposal
from src.utils.save import save, plot


def default_title():
    now = datetime.now()

    output_path = f"{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


if __name__ == "__main__":
    OP_key = jax.random.PRNGKey(0)

    N_chains = 4
    num_particles = 1000
    num_tempering_steps = 25
    num_mcmc_steps = 1

    delta = 0.1
    C = jnp.eye(dim)

    keys = jax.random.split(OP_key, N_chains)

    initial_parameter_value = {}
    coeffs_init = np.random.randn(num_particles)
    lmbda_schedule = np.logspace(-5, 0, num_tempering_steps)
    kwargs_for_default_mcmc_kernel = {}

    config = {'num_particles': num_particles, 'num_tempering_steps': num_tempering_steps,
              'num_mcmc_steps': num_mcmc_steps,
              'params_proposal': (delta, C),
              'kwargs_for_default_mcmc_kernel': kwargs_for_default_mcmc_kernel,
              'lmbda_schedule': lmbda_schedule,
              'target_ESS': None,
              'initial_parameter_value': initial_parameter_value,
              'kernel_id': 'cranck_nicholson_proposal',
              'file': os.path.basename(__file__)}

    my_smc = TemperedSMC_MCMC(logprior_fn, loglikelihood_fn, dim,
                              cranck_nicholson_proposal(*config['params_proposal']),
                              kwargs=kwargs_for_default_mcmc_kernel)


    @jax.vmap
    def wrapped_smc(key):
        init_particles = [jax.random.normal(key, shape=(num_particles, dim))]
        return my_smc.fixed_schedule_tempered_smc(key, init_particles, initial_parameter_value, num_mcmc_steps,
                                                  lmbda_schedule)


    res, chain = wrapped_smc(keys)

    save(chain, f"{os.path.basename(__file__)}_SONAR_CN_{num_tempering_steps}_{num_particles}_{num_mcmc_steps}", config,
         output_path=default_title())
    plot(chain, f"{os.path.basename(__file__)}_SONAR_CN_{num_tempering_steps}_{num_particles}_{num_mcmc_steps}")
