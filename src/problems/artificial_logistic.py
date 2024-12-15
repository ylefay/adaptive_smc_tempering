import jax
import jax.numpy as jnp

from problems.logistic import get_log_likelihood

def create_problem(OP_key, mean, cov, N):
    dim = mean.shape[0]
    predictors = jax.random.uniform(OP_key, (N, dim))
    beta = jax.random.multivariate_normal(OP_key, mean, cov)
    print(mean)
    print(jnp.diag(jnp.linalg.cholesky(cov)))
    print(beta)
    labels = jnp.sign(predictors@beta)
    _loglikelihood_fn = get_log_likelihood(predictors*labels[:, None])
    __loglikelihood_fn = lambda x: _loglikelihood_fn(x)
    def logprior_fn(x):
        return jax.scipy.stats.norm.logpdf(x[0], loc=jnp.zeros(dim), scale=jnp.ones(dim)).sum()

    loglikelihood_fn = lambda x: __loglikelihood_fn(x[0])
    return loglikelihood_fn, logprior_fn

