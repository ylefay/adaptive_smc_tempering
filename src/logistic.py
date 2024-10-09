import jax
import jax.numpy as jnp


def logistic(x):
    # return 1 / (1 + jnp.exp(-x))
    return jax.scipy.special.expit(x)


def normal_cdf(x):
    return 0.5 * (1 + jax.scipy.special.erf(x / jnp.sqrt(2)))


def get_log_likelihood(flipped_predictors, cdf=logistic):
    """
    Define the log target density of the posterior distribution of the logistic regression model,
    assuming a Gaussian prior.
    """
    if cdf == logistic:
        def tgt_log_density(beta):
            logcdf = - jnp.log1p(jnp.exp(-flipped_predictors @ beta.T))
            logcdf = jnp.nan_to_num(logcdf, False, nan=0.0, posinf=0.0, neginf=0.0)
            log_likelihood = jnp.sum(logcdf, axis=-1)
            return log_likelihood
    else:
        def tgt_log_density(beta):
            logcdf = jnp.log(cdf(flipped_predictors @ beta.T))
            logcdf = jnp.nan_to_num(logcdf, False, nan=0.0, posinf=0.0, neginf=0.0)
            log_likelihood = jnp.sum(logcdf, axis=-1)
            return log_likelihood

    return tgt_log_density
