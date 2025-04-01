import jax

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)
# jax.config.update("default_device", "cpu")

OP_key = jax.random.PRNGKey(0)

def default_config():
    num_parallel_chain = 16
    num_mcmc_steps = 4000
    n_chains = 1
    target_ess = 0.5
    dim = 2
    OUTPUT_PATH = "./output/"
    sequential_repetitions = 1

    seq_keys = jax.random.split(OP_key, sequential_repetitions)
    all_keys = jax.vmap(lambda key: jax.random.split(key, n_chains))(seq_keys)

    prefix = "default_config"

    config = {'dim': dim, 'num_parallel_chain': num_parallel_chain, 'num_mcmc_steps': num_mcmc_steps,
              'n_chains': n_chains, 'target_ess': target_ess, 'OUTPUT_PATH': OUTPUT_PATH,
              'sequential_repetitions': sequential_repetitions, 'seq_keys': seq_keys, 'all_keys': all_keys,
              'prefix': prefix}
    return config

