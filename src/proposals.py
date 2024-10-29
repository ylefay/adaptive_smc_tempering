from typing import Callable, NamedTuple

import src.mcmc_kernels.cranck_nicholson as cranck_nicholson
import src.mcmc_kernels.cranck_nicholson_RWM as cranck_nicholson_RWM
import src.mcmc_kernels.gaussian_238_empirical as gaussian_238_empirical


class mcmc_proposal(NamedTuple):
    mcmc_parameter_update_fn: Callable
    build_kernel: Callable


"""
Classic Gaussian proposal, zero-mean, covariance matrix equal to the empirical covariance of the particles scaled by gamma.
Require an initial covariance matrix to be passed:
    e.g., 'covariance_matrix': jnp.array([jnp.eye(dim)] * num_particles)
"""
gaussian_238_empirical_proposal = lambda: mcmc_proposal(gaussian_238_empirical.build_mcmc_kernel(),
                                                        gaussian_238_empirical.mcmc_parameter_update_fn)

cranck_nicholson_proposal = lambda delta, C: mcmc_proposal(cranck_nicholson.build_kernel(delta, C),
                                                           cranck_nicholson.mcmc_parameter_update_fn)

cranck_nicholson_RWM_proposal = lambda delta, C: mcmc_proposal(cranck_nicholson_RWM.build_kernel(delta, C),
                                                                   cranck_nicholson_RWM.mcmc_parameter_update_fn)
