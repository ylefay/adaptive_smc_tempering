import os
from datetime import datetime

import numpy as np

import src.SMC.SMCTemperingPreviousStates as SMC
import src.proposals as proposals
from src.experiments.utils import particle_initialisation_logexp
from src.problems.my_logistic_problem_sonar import *
from src.utils.save import plot, save


def default_title():
    now = datetime.now()

    output_path = f"{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


if __name__ == "__main__":
    now = datetime.now()

    OP_key = jax.random.PRNGKey(0)

    N_chains = 2
    num_particles = 1000
    num_tempering_steps = 25
    num_mcmc_steps = 1

    keys = jax.random.split(OP_key, N_chains)

    coeffs_init = np.random.randn(num_particles)
    initial_parameter_value = {'cov_particles': jnp.array([jnp.eye(dim)] * num_particles)}
    kwargs_for_default_mcmc_kernel = {}
    extra_parameters = {}
    decay_rate = 0.5
    period_max = 2
    params_proposal = (decay_rate, period_max)
    target_ESS = 0.5

    config = {'num_particles': num_particles, 'num_tempering_steps': num_tempering_steps,
              'num_mcmc_steps': num_mcmc_steps,
              'params_proposal': params_proposal,
              'kwargs_for_default_mcmc_kernel': kwargs_for_default_mcmc_kernel,
              'lmbda_schedule': None,
              'target_ESS': target_ESS,
              'initial_parameter_value': initial_parameter_value,
              'kernel_id': 'gaussian_238_empirical_geometric_proposal',
              'extra_parameters': extra_parameters,
              'SMC': 'TemperedSMC',
              'description': 'SONAR',
              'file': os.path.basename(__file__)}

    my_smc = getattr(SMC, config['SMC'])
    my_proposal = getattr(proposals, config['kernel_id'])
    my_smc = my_smc(logprior_fn, loglikelihood_fn, dim,
                    my_proposal(*config['params_proposal']),
                    kwargs=kwargs_for_default_mcmc_kernel)


    #    proposals.make_proposal_previous_states_compatible(my_proposal(*config['params_proposal'])),

    @jax.vmap
    def wrapped_smc(key):
        init_particles = particle_initialisation_logexp(key, num_particles, dim)
        return my_smc.adaptative_schedule_tempered_smc(key, init_particles, initial_parameter_value, num_mcmc_steps,
                                                       config['target_ESS'], num_tempering_steps, extra_parameters)


    with jax.disable_jit(False):
        res, chain = wrapped_smc(keys)
        save(chain,
             f"{os.path.basename(__file__)}_{config['SMC']}_{config['kernel_id']}_{config['description']}_{num_tempering_steps}_{num_particles}_{num_mcmc_steps}",
             config, output_path=default_title())
    plot(chain,
         f"{os.path.basename(__file__)}_{config['SMC']}_{config['kernel_id']}_{config['description']}_{num_tempering_steps}_{num_particles}_{num_mcmc_steps}")
