import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def cov_estimate(particles: ArrayLike, weights: ArrayLike) -> ArrayLike:
    r"""
    Given particles and weights at iteration t, of shapes resp. (N, dim), and (N, ),
    the weighted covariance estimate of \pi_t is
    (N - 1)^{-1}\sum_{1\leq i\leq N} W_{t}^{i} (X^i_t - \hat{\mu}_t) (X^i_t - \hat{\mu}_t)^{\top},
    where \hat{\mu}_t = N^{-1}\sum_{1\leq i\leq N} X^i_t
    """
    N = particles.shape[0]
    mu_hat = jnp.sum(weights[:, jnp.newaxis] * particles, axis=0) / N
    to_sum = (particles - mu_hat)
    cov_hat = 1 / (N - 1) * jnp.einsum('ij,ik,i->jk', to_sum, to_sum, weights)
    """if particles.shape[-1]>1:
        cov_hat = jnp.cov(particles, rowvar=False)
    else:
        cov_hat = jnp.var(particles).reshape((1, 1))"""
    return cov_hat


def build_gaussian_rw_proposal(C: ArrayLike):
    def gaussian_rwmh_cov_log_proposal(x, y):
        return jax.scipy.stats.multivariate_normal.logpdf(y, x, C)

    def gaussian_rwmh_sampler(key, x):
        return jax.random.multivariate_normal(key, x, C)

    return gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler


def build_gaussian_rwmh_cov_proposal(_, particles, log_weights, log_tgt_density_fn, i):
    """
    Adaptative RWMH kernels with scaling set to the optimal asymptotic scaling, i.e. 2.38^2/dim.
    See Optimal scaling for various Metropolis-Hastings algorithms, Gareth O. Roberts and Jeffrey S. Rosenthal
    """
    return build_gaussian_rwmh_cov_proposal_gamma(2.38, particles, log_weights, i)


def build_gaussian_rwmh_cov_proposal_gamma(gamma, particles, log_weights, log_tgt_density_fn, i):
    dim = particles.shape[-1]
    optimal_scale = gamma ** 2 / dim

    def fun_to_be_called_if_i_greater_than_one():
        r"""
        Compute the covariance estimate of \pi_{t-1} given t\geq 1
        """
        particles_at_i_minus_one = particles.at[i - 1].get().reshape(-1, particles.shape[-1])
        log_weights_at_i_minus_one = log_weights.at[i - 1].get().reshape(-1, )
        weights_at_i_minus_one = jnp.exp(log_weights_at_i_minus_one * 0)
        cov_hat = cov_estimate(particles_at_i_minus_one, weights_at_i_minus_one)
        return cov_hat

    def fun_to_be_called_if_i_equal_zero():
        r"""
        For the first iteration, use the estimated covariance from the particles X_{0}^{i}\sim \nu, the base measure
        """
        particles_at_0 = particles.at[0].get().reshape(-1, particles.shape[-1])
        if dim > 1:
            cov_hat = jnp.cov(particles_at_0, rowvar=False)
        else:
            cov_hat = jnp.var(particles_at_0, axis=0).reshape((1, 1))
        return cov_hat

    C = jax.lax.select(i == 0, optimal_scale * fun_to_be_called_if_i_equal_zero(),
                       optimal_scale * fun_to_be_called_if_i_greater_than_one())

    gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler = build_gaussian_rw_proposal(C)

    return gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler


def build_autoregressive_gaussian_rwmh_proposal(rho, _, __, log_tgt_density_fn, ___):
    dim = _.shape[-1]
    C = jnp.eye(dim)

    def gaussian_rwmh_log_proposal(x, y):
        return jax.scipy.stats.multivariate_normal.logpdf(y, rho * x, (1 - rho ** 2) * C)

    def gaussian_rwmh_sampler(key, x):
        return jax.random.multivariate_normal(key, rho * x, (1 - rho ** 2) * C)

    return gaussian_rwmh_log_proposal, gaussian_rwmh_sampler


def build_mala_proposal_gamma(gamma, particles, log_weights, log_tgt_density_fn, i):
    dim = particles.shape[-1]

    def fun_to_be_called_if_i_greater_than_one():
        r"""
        Compute the covariance estimate of \pi_{t-1} given t\geq 1
        """
        particles_at_i_minus_one = particles.at[i - 1].get().reshape(-1, particles.shape[-1])
        log_weights_at_i_minus_one = log_weights.at[i - 1].get().reshape(-1, )
        weights_at_i_minus_one = jnp.exp(log_weights_at_i_minus_one * 0)
        cov_hat = cov_estimate(particles_at_i_minus_one, weights_at_i_minus_one)
        return cov_hat

    def fun_to_be_called_if_i_equal_zero():
        r"""
        For the first iteration, use the estimated covariance from the particles X_{0}^{i}\sim \nu, the base measure
        """
        particles_at_0 = particles.at[0].get().reshape(-1, particles.shape[-1])
        if dim > 1:
            cov_hat = jnp.cov(particles_at_0, rowvar=False)
        else:
            cov_hat = jnp.var(particles_at_0, axis=0).reshape((1, 1))
        return cov_hat

    cov_hat = jax.lax.select(i == 0, fun_to_be_called_if_i_equal_zero(),
                             fun_to_be_called_if_i_greater_than_one())

    def gaussian_mala_log_proposal(x, y):
        return jax.scipy.stats.multivariate_normal.logpdf(y, x + 0.5 * gamma ** 2 / dim ** (1 / 3) * jnp.diag(
            cov_hat) * jax.jacfwd(
            log_tgt_density_fn)(x),
                                                          gamma ** 2 / dim ** (1 / 3) * cov_hat)

    def gaussian_mala_sampler(key, x):
        return jax.random.multivariate_normal(key, x + 0.5 * gamma ** 2 / dim ** (1 / 3) * jnp.diag(
            cov_hat) * jax.jacfwd(
            log_tgt_density_fn)(x),
                                              gamma ** 2 / dim ** (1 / 3) * cov_hat)

    return gaussian_mala_log_proposal, gaussian_mala_sampler
