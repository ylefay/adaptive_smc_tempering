import jax
import jax.numpy as jnp

import src.mcmc_kernels.gaussian_238_empirical
from src.utils.array_manipulations import temperedsmcstate_to_array

__all__ = [
    "get_mcmc_parameter_update_fn",
    "build_mcmc_kernel"
]


def get_mcmc_parameter_update_fn(decay_rate: float, period_max: int):
    """
    covariance matrix equal to the normalized geometric sum of empirical covariances of the (previous) particles,
    maximum of period_max previous iteratoins taken into account
    scaled by gamma.
    """

    def mcmc_parameter_update_fn(SMCState_var, cumul_states, i, SMCInfo_var):
        particles = SMCState_var[0][0]
        # Need to figure why I need to do that:
        cumul_states = cumul_states.at[i].set(temperedsmcstate_to_array(SMCState_var))
        # Dynamically selecting the period_max previous particles
        selected_previous_states = jax.lax.dynamic_slice_in_dim(cumul_states, jax.lax.max(i + 1 - period_max, 0),
                                                                period_max, axis=0)[..., :-1]
        cov_particles = jax.vmap(lambda x: jnp.cov(x, rowvar=False))(selected_previous_states)
        weights = jnp.pow(decay_rate, jnp.arange(period_max, 0, -1))
        normalization_constant_for_weights = jnp.sum(weights)
        cov_particles = jnp.sum(cov_particles * weights[:, None, None], axis=0) / normalization_constant_for_weights
        return {'cov_particles': jnp.expand_dims(cov_particles, axis=0)}

    return mcmc_parameter_update_fn


build_mcmc_kernel = src.mcmc_kernels.gaussian_238_empirical.build_mcmc_kernel
