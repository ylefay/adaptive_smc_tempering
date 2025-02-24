import jax
import jax.numpy as jnp


def create_problem(dim, scale=None, mean=None, cov=None):
    """
    Create a Gaussian problem with a given dimension and scale.
    The target density is a Gaussian distribution with zero mean and covariance matrix multiplied by scale squared.
    """
    return create_sparse_problem(dim, 0, scale, mean, cov)

def create_sparse_problem(dim, latent_dim, scale=None, mean=None, cov=None):
    """
    Create a Gaussian problem with a given dimension and scale.
    The covariance of the likelihood is made sparse
    by setting the covariance of the excluded variables to scale squared.
    """
    if mean is None:
        mean = jnp.zeros(dim)
    if cov is None:
        cov = jnp.eye(dim)
    if scale is None:
        scale = 1.0

    cov = cov.at[latent_dim:, latent_dim:].set(cov.at[latent_dim:, latent_dim:].get() * scale**2)

    def loglikelihood_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=mean, cov=cov)

    return loglikelihood_fn
