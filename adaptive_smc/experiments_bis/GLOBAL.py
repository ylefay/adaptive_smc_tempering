import jax

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)
# jax.config.update("default_device", "cpu")

OP_key = jax.random.PRNGKey(0)

num_parallel_chain = 16
num_mcmc_steps = 4000
n_chains = 1
target_ess = 0.5

dim = 1

OUTPUT_PATH = "./output/"

sequential_repetitions = 1

seq_keys = jax.random.split(OP_key, sequential_repetitions)
all_keys = jax.vmap(lambda key: jax.random.split(key, n_chains))(seq_keys)
