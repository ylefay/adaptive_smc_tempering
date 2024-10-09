import blackjax.smc.tempered as smc
import jax.numpy as jnp
import jax.scipy.stats.norm
from blackjax.mcmc import hmc
from particles import datasets

from src.logistic import get_log_likelihood
from src.proposals import build_crank_nicholson_kernel
from src.smc import resample_fn


def get_dataset(flip=True):
    dataset = datasets.Sonar()
    data = dataset.preprocess(dataset.raw_data, return_y=not flip)
    return data


if __name__ == "__main__":
    flipped_predictors = get_dataset()
    N, dim = flipped_predictors.shape


    def logprior_fn(x):
        return jax.scipy.stats.norm.logpdf(x, loc=jnp.zeros(dim), scale=jnp.ones(dim))


    loglikehood_fn = get_log_likelihood(flipped_predictors)

    mcmc_step_fn = build_crank_nicholson_kernel(0.1, jnp.eye(dim))


    def mcmc_init_fn(position, logdensity_fn):
        return hmc.init(position=position, logdensity_fn=logdensity_fn)


    smc = smc.build_kernel(logprior_fn, loglikehood_fn, mcmc_step_fn, mcmc_init_fn, resample_fn)


    def inference_loop(rng_key):
        raise NotImplementedError
