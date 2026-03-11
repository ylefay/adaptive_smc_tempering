import jax
import jax.numpy as jnp
import jax.scipy as jsp


def random_well_conditioned_cov(key, dim, eig_min=0.5, eig_max=2.0):
    """
    Generate a random SPD matrix with bounded condition number.
    The condition number <= eig_max / eig_min, independent of dim.
    """
    key_q, key_eig = jax.random.split(key)

    # Random orthogonal matrix via QR
    A = jax.random.normal(key_q, (dim, dim))
    Q, R = jnp.linalg.qr(A)

    # Fix sign to make QR deterministic
    Q = Q * jnp.sign(jnp.diag(R))

    # Eigenvalues in bounded range
    eigvals = jax.random.uniform(
        key_eig, (dim,), minval=eig_min, maxval=eig_max
    )

    return Q @ jnp.diag(eigvals) @ Q.T


def make_model(dim, key):
    """
    Returns:
        loglikelihood_fn(x)
        base_measure_sampler(key)
        logbase_density_fn(x)
    with well-conditioned covariance matrices.
    """

    key_cov, key_cov_base = jax.random.split(key, 2)

    cov = random_well_conditioned_cov(key_cov, dim)
    cov_base = random_well_conditioned_cov(key_cov_base, dim)
    precision = jnp.linalg.inv(cov)
    prec_base = jnp.linalg.inv(cov_base)

    def loglikelihood_fn(x):
        return -jnp.einsum('i,ij,j->', x, precision, x) * 0.5

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), cov_base)

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, jnp.zeros(dim), cov_base)

    _, logdet = jnp.linalg.slogdet(precision + prec_base)
    logdet = - logdet
    _, logdetbase = jnp.linalg.slogdet(cov_base)
    true_log_normalising_constant = logdet * 0.5 - 0.5 * logdetbase

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn, true_log_normalising_constant
