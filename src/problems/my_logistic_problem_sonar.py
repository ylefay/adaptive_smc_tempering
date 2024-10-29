import jax
import jax.numpy as jnp

from src.problems.logistic import get_dataset
from src.problems.logistic import get_log_likelihood

flipped_predictors = get_dataset(dataset="Sonar")
N, dim = flipped_predictors.shape

_loglikelihood_fn = get_log_likelihood(flipped_predictors)
loglikelihood_fn = lambda x: _loglikelihood_fn(x[0])


def logprior_fn(x):
    return jax.scipy.stats.norm.logpdf(x[0], loc=jnp.zeros(dim), scale=jnp.ones(dim)).sum()
