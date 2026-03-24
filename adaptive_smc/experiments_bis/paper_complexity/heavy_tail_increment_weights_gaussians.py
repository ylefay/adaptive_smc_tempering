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


def make_model(dim, heavy_factor, key, mean=None):
    """
    Returns:
        loglikelihood_fn(x)
        base_measure_sampler(key)
        logbase_density_fn(x)
    with light or heavy-tailed weights.
    """

    key_cov, key_cov_base = jax.random.split(key, 2)
    # -----------------------
    # Base: keep well-conditioned
    # -----------------------
    cov_base = random_well_conditioned_cov(key_cov_base, dim)
    prec_base = jnp.linalg.inv(cov_base)

    # Take base covariance and multiply eigenvalues by a factor 
    base_eigvals, base_vecs = jnp.linalg.eigh(cov_base)
    cov = (base_vecs @ jnp.diag(base_eigvals * heavy_factor) @ base_vecs.T)
    precision = jnp.linalg.inv(cov)
    if not mean:
        mean = jnp.zeros((dim,))
    if isinstance(mean, float):
        mean = jnp.ones((dim, ))*mean
    # -----------------------
    # Log-likelihood (unnormalized)
    # -----------------------
    def loglikelihood_fn(x):
        return -0.5 * jnp.einsum('i,ij,j->', x-mean, precision-prec_base, x-mean)

    # -----------------------
    # Base sampler and log-density
    # -----------------------
    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), cov_base)

    def logbase_density_fn(x):
        return jsp.stats.multivariate_normal.logpdf(x, jnp.zeros(dim), cov_base)

    _, logdet = jnp.linalg.slogdet(precision)
    logdet = -logdet
    _, logdetbase = jnp.linalg.slogdet(cov_base)
    posterior_mean = cov @ (precision - prec_base) @ mean
    true_log_normalising_constant = 0.5 * logdet - 0.5 * logdetbase + (-0.5 * mean.T@(precision-prec_base)@mean + 0.5 * posterior_mean.T @ precision @ posterior_mean)

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn, true_log_normalising_constant