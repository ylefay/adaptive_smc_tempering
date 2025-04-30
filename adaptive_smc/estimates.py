from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from particles.distributions import Gamma

from adaptive_smc.smc_types import LogDensity
from adaptive_smc.utils import unvec, vec


def cov_estimate(particles: ArrayLike, weights: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    r"""
    Given particles and weights at iteration t, of shapes resp. (N, dim), and (N, ),
    the weighted covariance estimate of \pi_t is
    \sum_{1\leq i\leq N} W_{t}^{i} (X^i_t - \hat{\mu}_t) (X^i_t - \hat{\mu}_t)^{\top},
    where \hat{\mu}_t = \sum_{1\leq i\leq N} X^i_t w^i_t
    """
    mu_hat = jnp.sum(weights[:, jnp.newaxis] * particles, axis=0)
    to_sum = (particles - mu_hat)
    cov_hat = jnp.einsum('ij,ik,i->jk', to_sum, to_sum, weights)
    return cov_hat, mu_hat


def cov_increment_estimate(particles: ArrayLike, weights: ArrayLike, dlambda: ArrayLike,
                           log_likelihood_fn: LogDensity) -> Tuple[ArrayLike, ArrayLike]:
    r"""
    At iteration t+1, compute the estimate of the first-order increment of the covariance of \pi_{t+1}
    using samples and weights from iteration t. Same for mean.
    The increment is equal to
        \dlambda\times \hat{\bbE}_t[(X-\hat{\bbE}_t(X))(X-\hat{\bbE}_t(X))^{\top} . (s-\hat{\bbE}(s))],
    where s is the log-likelihood function, and $\hat{\bbE}_t is the weighted mean operator at iteration t
    targeting expectations under \pi_t (i.e., X_t approx. \sim \pi_{t-1})
    If at iteration t+1, you aim to compute the estimate for \pi_{t}, then set the weights to 1.
    """

    def dM(f: Callable):
        _weights = weights / jnp.sum(weights)
        likelihoods = log_likelihood_fn(particles)
        evalf = f(particles)
        _weights_reshaped = _weights.reshape(_weights.shape + (1,) * (jnp.ndim(evalf) - 1))
        Ef = jnp.sum(_weights_reshaped * evalf, axis=0)
        Es = jnp.sum(_weights * likelihoods, axis=0)
        Efs = jnp.sum(_weights_reshaped * evalf * likelihoods.reshape(likelihoods.shape + (1,) * (jnp.ndim(evalf) - 1)),
                      axis=0)
        return Efs - Ef * Es

    def ddM(f: Callable):
        _weights = weights / jnp.sum(weights)
        likelihoods = log_likelihood_fn(particles)
        evalf = f(particles)
        _weights_reshaped = _weights.reshape(_weights.shape + (1,) * (jnp.ndim(evalf) - 1))
        Efssq = jnp.sum(
            evalf * (likelihoods ** 2).reshape(likelihoods.shape + (1,) * (jnp.ndim(evalf) - 1)) * _weights_reshaped,
            axis=0)
        Efs = jnp.sum(_weights_reshaped * evalf * likelihoods.reshape(likelihoods.shape + (1,) * (jnp.ndim(evalf) - 1)),
                      axis=0)
        Es = jnp.sum(_weights * likelihoods, axis=0)
        Ef = jnp.sum(_weights_reshaped * evalf, axis=0)
        Essq = jnp.sum(_weights * likelihoods ** 2, axis=0)
        return Efssq - 2 * Efs * Es + 2 * Ef * (Es) ** 2 - Ef * Essq

    """
    weights = weights / jnp.sum(weights)  # ensuring normalised weights
    likelihoods = log_likelihood_fn(particles)
    mean_log_likelihood = jnp.sum(weights * likelihoods)
    mu_hat = jnp.sum(weights[:, jnp.newaxis] * particles, axis=0)
    to_sum = (particles - mu_hat)
    cov_hat = jnp.einsum('ij,ik,i->jk', to_sum, to_sum, weights)
    stat_cov = to_sum.reshape(to_sum.shape + (1,)) @ jnp.swapaxes(to_sum.reshape(to_sum.shape + (1,)), 1, 2) - cov_hat
    stat_dcov = stat_cov * (likelihoods - mean_log_likelihood)[:, jnp.newaxis, jnp.newaxis]
    return dlambda * dcov
    
    """
    """
    weights = weights / jnp.sum(weights)  # ensuring normalised weights
    likelihoods = log_likelihood_fn(particles)
    mean_log_likelihood = jnp.sum(weights * likelihoods)
    xx_hat = jnp.einsum('ij,ik,i->jk', particles, particles, weights)
    stat_xxt = particles.reshape(particles.shape + (1,)) @ jnp.swapaxes(particles.reshape(particles.shape + (1,)), 1, 2) - xx_hat
    stat_dxxt = stat_xxt * (likelihoods - mean_log_likelihood)[:, jnp.newaxis, jnp.newaxis]
    dxxt = jnp.sum(weights[:, jnp.newaxis, jnp.newaxis] * stat_dxxt, axis=0)
    x_hat = jnp.sum(weights[:, jnp.newaxis] * particles, axis=0)
    stat_x = particles - x_hat
    stat_dx = stat_x * (likelihoods - mean_log_likelihood)[:, jnp.newaxis]
    dx = jnp.sum(weights[:, jnp.newaxis] * stat_dx, axis=0)
    dcov = dxxt - x_hat[:,jnp.newaxis] @ dx[jnp.newaxis,:] - dx[:,jnp.newaxis] @ x_hat[jnp.newaxis,:] 
    return dlambda * dcov
    
    """
    weights = weights / weights.sum()
    fX = lambda X: X
    fXXT = jax.vmap(lambda X: X[:, jnp.newaxis] @ X[jnp.newaxis, :])
    MX = jnp.sum(weights[:, jnp.newaxis] * particles, axis=0)
    return dlambda * dM(fXXT) + 0.5 * dlambda ** 2 * ddM(fXXT) - (dlambda * (
            MX @ dM(fX).T + dM(fX) @ MX.T) + dlambda ** 2 * (dM(fX) @ dM(fX).T + 0.5 * (
            MX @ ddM(fX).T + ddM(fX) @ MX.T))), dlambda * dM(fX) + 0.5 * dlambda ** 2 * ddM(fX)


def estimate_I(particles: ArrayLike, weights: ArrayLike, log_target_density_fn, dlambda: ArrayLike):
    # NOT TESTED.
    r"""
    Weighted-average estimate of
         I = \bbE_{z\sim \pi}[J J^{\top}],
    where J = Jac \log \pi(z).
    No theoretical proof that this quantity is the one that should be used
    in the dimensional non-independent setting for \pi.
    """

    def J(z):
        return jax.jacobian(log_target_density_fn)(z)

    Jz = J(particles)
    JzJzT = jnp.einsum('ij,jk->ik', Jz, Jz)
    estimate = jnp.sum(JzJzT * weights[:, jnp.newaxis, jnp.newaxis], axis=0)
    return estimate


def inverse_FIM_gaussian_approx(particles: ArrayLike, weights: ArrayLike, log_target_density_fn, dlambda: ArrayLike):
    # NOT TESTED
    r"""
    gradient-free inverse FIM estimate under Gaussian approximation
    """
    dim = particles.shape[-1]

    @jax.vmap
    def modified_statistic(z):
        vecZZt = vec(z[:, jnp.newaxis] @ z[:, jnp.newaxis].T)
        vecZZt = vecZZt.at[0::(dim + 1)].set((vecZZt.at[0::(dim + 1)].get() - 1) / jnp.sqrt(2))
        vectriuunvecvecZZt = vec(unvec(vecZZt, (dim, dim)).at[jnp.triu_indices(dim)].get())
        return jnp.concatenate([z, vectriuunvecvecZZt, jnp.array([1.])])

    cov, mu = cov_estimate(particles, weights)
    chol_cov = jax.scipy.linalg.cholesky(cov)
    gamma = jnp.mean(modified_statistic(particles) * log_target_density_fn(particles), axis=0)
    Gamma_matrix = jnp.diag(jnp.sqrt(2) /2* gamma.at[dim::].get())
    mask = ~jnp.isin(jnp.arange(0, dim*dim-1), jnp.arange(0, dim*dim-1, dim))

    Gamma_matrix = Gamma_matrix.at[jnp.triu_indices(dim)].set(
        gamma.at[mask].get()
    )
    Gamma_matrix = Gamma_matrix.T + Gamma_matrix
    return chol_cov.T @ jnp.linalg.inv(Gamma_matrix) @ chol_cov, mu
