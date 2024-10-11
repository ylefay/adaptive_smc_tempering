import blackjax.smc.tempered as smc
import jax.numpy as jnp
import jax.scipy.stats.norm
from blackjax.mcmc import hmc
from blackjax.smc.resampling import multinomial
from particles import datasets

from adaptive_smc_tempering.src.logistic import get_log_likelihood
from adaptive_smc_tempering.src.proposals import build_crank_nicholson_kernel


def get_dataset(flip=True):
    dataset = datasets.Sonar()
    data = dataset.preprocess(dataset.raw_data, return_y=not flip)
    return data


if __name__ == "__main__":
    flipped_predictors = get_dataset()
    N, dim = flipped_predictors.shape

    resample_fn = multinomial


    def logprior_fn(x):
        return jax.scipy.stats.norm.logpdf(x, loc=jnp.zeros(dim), scale=jnp.ones(dim))


    loglikehood_fn = get_log_likelihood(flipped_predictors)

    mcmc_step_fn = build_crank_nicholson_kernel(0.1, jnp.eye(dim))


    def mcmc_init_fn(position, logdensity_fn):
        return hmc.init(position=position, logdensity_fn=logdensity_fn)


    smc_kernel = smc.build_kernel(logprior_fn, loglikehood_fn, mcmc_step_fn, mcmc_init_fn, resample_fn)


    def inference_loop(rng_key, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = smc_kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)
        return states

        initial_position = {"loc": jnp.zeros(dim)}

        inference_loop(jax.random.PRNGKey(0), smc.init(N, jnp.zeros((N, dim))), 10)
