import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc_tempering.adaptive_smc.estimates import cov_estimate


def build_build_autoregressive_gaussian_rwmh_proposal(C):
    r"""
    Construct the build function for autoregressive proposal:
    q(y\mid x) = N(\rho x, (1-\rho^2)C),
    where C is a given matrix.
    """

    def _build(rho, _, __, log_tgt_density_fn, ___):
        def gaussian_rwmh_log_proposal(x, y):
            return jax.scipy.stats.multivariate_normal.logpdf(y, rho * x, (1 - rho ** 2) * C)

        def gaussian_rwmh_sampler(key, x):
            return jax.random.multivariate_normal(key, rho * x, (1 - rho ** 2) * C)

        return gaussian_rwmh_log_proposal, gaussian_rwmh_sampler

    return _build


def build_build_pmala_proposal(C):
    r"""
    Build the builder (...).
    Preconditioned MALA proposal,
    Auxiliary gradient-based sampling algorithms,
    q(y\mid x) = N(y, (1-\delta/2)x+\delta/2 C \grad f, \delta C),
    where C is a given matrix
    """

    def _build(rho, _, __, log_tgt_density_fn, ___):
        dim = _.shape[-1]
        C = jnp.eye(dim)

        def gaussian_rwmh_log_proposal(x, y):
            return jax.scipy.stats.multivariate_normal.logpdf(y, rho * x, (1 - rho ** 2) * C)

        def gaussian_rwmh_sampler(key, x):
            return jax.random.multivariate_normal(key, rho * x, (1 - rho ** 2) * C)

        return gaussian_rwmh_log_proposal, gaussian_rwmh_sampler

    return _build


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
    """
    Same as build_gaussian_rwmh_cov_proposal with gamma**2/dim in front of the covariance matrix
    """
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

    C = optimal_scale * fun_to_be_called_if_i_greater_than_one()

    gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler = build_gaussian_rw_proposal(C)

    return gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler


def build_autoregressive_gaussian_rwmh_proposal(rho, _, __, log_tgt_density_fn, ___):
    r"""
    Autoregressive proposal:
    q(y\mid x) = N(\rho x, (1-\rho^2)C),
    where C is a fixed matrix, here I_n.
    """
    dim = _.shape[-1]
    C = jnp.eye(dim)
    return build_build_autoregressive_gaussian_rwmh_proposal(C)(rho, _, __, log_tgt_density_fn, ___)


def build_mala_proposal_gamma(gamma, particles, log_weights, log_tgt_density_fn, i):
    """
    Metropolis Adjusted Langevin proposal with a gamma parameter
    """
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

    cov_hat = fun_to_be_called_if_i_greater_than_one()

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


def build_pmala_proposal(rho, _, __, log_tgt_density_fn, ___):
    r"""
    Preconditioned MALA proposal,
    Auxiliary gradient-based sampling algorithms,
    q(y\mid x) = N(y, (1-\delta/2)x+\delta/2 C \grad f, \delta C),
    where C is a fixed matrix, here I_n.
    """
    dim = _.shape[-1]
    C = jnp.eye(dim)
    return build_build_pmala_proposal(C)(rho, _, __, log_tgt_density_fn, ___)
