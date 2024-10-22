import jax.numpy as jnp
import jax.scipy.stats.norm
from particles import datasets

from src.logistic import get_log_likelihood


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
