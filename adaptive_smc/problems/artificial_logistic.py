import jax
import jax.numpy as jnp

from adaptive_smc.problems.logistic import get_log_likelihood


def create_problem(OP_key, mean, cov, N):
    dim = mean.shape[0]
    predictors = jax.random.uniform(OP_key, (N, dim))
    beta = jax.random.multivariate_normal(OP_key, mean, cov)
    labels = jnp.sign(predictors @ beta)
    loglikelihood_fn = get_log_likelihood(predictors * labels[:, None])
    return loglikelihood_fn
