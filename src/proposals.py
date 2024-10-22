from typing import Callable, NamedTuple
import src.mcmc_kernels.gaussian_238_empirical as gaussian_238_empirical


class mcmc_proposal(NamedTuple):
    mcmc_parameter_update_fn: Callable
    build_kernel: Callable



"""
Classic Gaussian proposal, zero-mean, covariance matrix equal to the empirical covariance of the particles scaled by gamma.
Require an initial covariance matrix to be passed:
    e.g., 'covariance_matrix': jnp.array([jnp.eye(dim)] * num_particles)
"""
gaussian_238_empirical_proposal = mcmc_proposal(gaussian_238_empirical.build_mcmc_kernel_default(),
                                                gaussian_238_empirical.mcmc_parameter_update_fn_default)
