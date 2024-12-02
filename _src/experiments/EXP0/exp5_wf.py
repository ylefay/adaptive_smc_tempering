import os
from datetime import datetime

import src.SMCTemperingPreviousStates as SMC
import src.proposals as proposals
from src.experiments.utils import particle_initialisation_logexp
from src.problems.my_logistic_problem_sonar import *
from src.utils.save import save

OP_key = jax.random.PRNGKey(0)

N_chains = 4
num_tempering_steps = 25
num_mcmc_steps = 3

keys = jax.random.split(OP_key, N_chains)


def default_title():
    now = datetime.now()

    output_path = f"{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


def exp(num_particles, period_max):
    initial_parameter_value = {'cov_particles': jnp.expand_dims(jnp.eye(dim), axis=0)}
    kwargs_for_default_mcmc_kernel = {}
    extra_parameters = {}
    decay_rate = 1 / (period_max + 1)
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
              'SMC': 'WasteFreeTemperedSMC',
              'description': 'SONAR',
              'file': os.path.basename(__file__)}

    my_smc = getattr(SMC, config['SMC'])
    my_proposal = getattr(proposals, config['kernel_id'])
    my_smc = my_smc(logprior_fn, loglikelihood_fn, dim,
                    my_proposal(*config['params_proposal']),
                    kwargs=kwargs_for_default_mcmc_kernel)

    @jax.vmap
    def wrapped_smc(key):
        init_particles = particle_initialisation_logexp(key, num_particles, dim)
        return my_smc.adaptative_schedule_tempered_smc(key, init_particles, initial_parameter_value, num_mcmc_steps,
                                                       config['target_ESS'], num_tempering_steps, extra_parameters)

    res, chain = wrapped_smc(keys)
    save(chain,
         f"{os.path.basename(__file__)}_{config['SMC']}_{config['kernel_id']}_{config['description']}_{num_tempering_steps}_{num_particles}_{num_mcmc_steps}",
         config, output_path=default_title())


if __name__ == "__main__":
    N = jnp.array([60, 100, 200, 500, 1000])
    period_max = jnp.array([1, 2, 3])
    for n in N:
        for p in period_max:
            exp(int(n), int(p))
