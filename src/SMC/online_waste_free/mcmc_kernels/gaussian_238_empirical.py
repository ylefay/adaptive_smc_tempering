import jax.numpy as jnp
from blackjax.mcmc.random_walk import generate_gaussian_noise
from blackjax.mcmc.random_walk import init
import src.SMC.online_waste_free.mcmc.random_walk as corrected_random_walk
import jax

__all__ = [
    "mcmc_parameter_update_fn",
    "build_mcmc_kernel"
]


def mcmc_parameter_update_fn(SMCState_var, SMCInfo_var):
    particles = SMCState_var[0][0]
    cov_particles = jnp.cov(particles, rowvar=False)
    return {'cov_particles': jnp.expand_dims(cov_particles, axis=0)}


def build_mcmc_kernel():
    """
    Same as src.mcmc_kernels.gaussian_238_empirical.py, but with the addition of the proposed state.
    TODO: Check that the proposed state is correctly returned thanks to:
        src.SMC.online_waste_free.mcmc.random_walk import *
    -------

    """

    def kernel(rng_key, state, logdensity_fn, cov_particles):
        def propose(rng_key, position):
            x = position[0]
            gamma = 2.38 / x.shape[0] ** 0.5
            D, V = jax.scipy.linalg.eigh(cov_particles)
            sqrtm = (V * jnp.sqrt(D)) @ V.T
            C = sqrtm * gamma
            return generate_gaussian_noise(rng_key, position=position, mu=jnp.zeros_like(x), sigma=C)

        state = init(position=state.position, logdensity_fn=logdensity_fn)
        new_state, info = corrected_random_walk.build_additive_step()(rng_key, state, logdensity_fn, propose)
        return new_state, info

    return kernel
