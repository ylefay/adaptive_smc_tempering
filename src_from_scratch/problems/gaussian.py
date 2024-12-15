import jax
import jax.numpy as jnp


def create_problem(dim, scale=None, mean=None, cov=None):
    """
    Create a Gaussian problem with a given dimension and scale.
    The target density is a Gaussian distribution with zero mean and identity covariance matrix multiplied by scale squared.
    """
    if mean is None:
        mean = jnp.zeros(dim)
    if cov is None:
        cov = jnp.eye(dim)
        if scale:
            cov *= scale ** 2

    def loglikelihood_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=mean, cov=cov)

    return loglikelihood_fn
