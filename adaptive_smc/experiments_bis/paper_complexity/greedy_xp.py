from adaptive_smc.SMC import GenericGreedyWasteFreeTemperingSMC
from adaptive_smc.experiments_bis.paper_complexity.heavy_tail_increment_weights_gaussians import make_model
from adaptive_smc.experiments_bis.paper_complexity.proposal import \
    build_gaussian_rw_proposal_fixed_scaling
from adaptive_smc.experiments_bis.paper_complexity.save import save

import os
from datetime import datetime
import jax
import jax.numpy as jnp
import yaml

jax.config.update("jax_enable_x64", False)

OP_key = jax.random.PRNGKey(2)
_, key = jax.random.split(OP_key)


def default_title(prefix='', eps=None):
    now = datetime.now()
    filename = (
        f"{prefix}_eps{eps:.4f}_"
        f"{os.path.basename(__file__)}_"
        f"{now.strftime('%m%d%H%M%S')}.pkl"
    )
    return filename


def xp(config, keys):
    # Copy config so we never mutate the original
    run_config = dict(config)

    dim = run_config["dim"]
    target_ess = 0.5
    eps = run_config["eps"]
    T = int(jnp.sqrt(dim))

    num_parallel_chain = run_config["num_parallel_chain"]
    num_mcmc_steps = int(
        run_config["num_mcmc_steps"] * dim ** 2.0 // 2)
    tempering_length = T * 2

    num_mcmc_steps_schedule = jnp.zeros((tempering_length,), dtype=int)
    num_mcmc_steps_schedule = num_mcmc_steps_schedule.at[:-1].set(int(num_mcmc_steps * eps ** 2))
    num_mcmc_steps_schedule = num_mcmc_steps_schedule.at[-1].set(num_mcmc_steps)

    tempering_sequence = jnp.linspace(0, 1, tempering_length)

    # Generate model
    model_key = jax.random.PRNGKey(0)
    loglikelihood_fn, base_measure_sampler, logbase_density_fn, true_log_Z = make_model(dim,
                                                                                        config.get('heavy_factor', 1.0),
                                                                                        model_key)

    run_config.update({
        "eps": eps,
        "Ps": num_mcmc_steps_schedule,
        "M": num_parallel_chain,
        "logZ": true_log_Z,
        "tempering_sequence": tempering_sequence
    })

    smc = GenericGreedyWasteFreeTemperingSMC(
        logbase_density_fn,
        base_measure_sampler,
        loglikelihood_fn,
        build_gaussian_rw_proposal_fixed_scaling
    )

    num_mcmc_steps = int(jnp.max(num_mcmc_steps_schedule))

    @jax.jit
    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(
            key,
            num_parallel_chain,
            num_mcmc_steps,
            num_mcmc_steps_schedule,
            tempering_sequence,
            target_ess
        )

    res = wrapper_smc(keys)

    output_file = (
            run_config["OUTPUT_PATH"]
            + default_title(run_config.get("prefix", ""), eps)
    )

    save(res, run_config, output_file)


if __name__ == "__main__":

    yaml_file = "greedy_xp.yaml"

    with open(yaml_file, "r") as file:
        y_config = yaml.safe_load(file)

    for name_of_my_config, config in y_config.items():

        if config.get("run", True):
            # Set a list of dimensions to vary
            dim_list = config.get("dim_list", [5])

            sequential_repetitions = config.pop("sequential_repetitions", 1)
            parallel_repetitions = config["parallel_repetitions"]

            seq_keys = jax.random.split(key, sequential_repetitions)

            all_keys = jax.vmap(
                lambda k: jax.random.split(k, parallel_repetitions)
            )(seq_keys)

            _, key = jax.random.split(seq_keys[-1])

            # Loop over dimensions
            for dim in dim_list:
                # update config with current dim
                config["dim"] = dim

                for keys in all_keys:
                    xp(config, keys)
