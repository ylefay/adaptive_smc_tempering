import jax
import jax.numpy as jnp
from blackjax.mcmc.random_walk import generate_gaussian_noise
from blackjax.mcmc.random_walk import init
from src.utils.array_manipulations import temperedsmcstate_to_array
import src.SMC.online_waste_free.mcmc.random_walk as corrected_random_walk
from blackjax.smc.base import SMCState
from functools import partial

__all__ = [
    "get_mcmc_parameter_update_fn",
    "build_mcmc_kernel"
]


def get_mcmc_parameter_update_fn(C):
    def log_density_gaussian(x, y, rho):
        return -0.5 * (y-rho * x).T @ jnp.linalg.inv((1-rho**2)*C) @ (y-rho * x)
    def acceptation_reject(rho):
        raise NotImplementedError
        return jax.lax.min(1, logratio)
    def mcmc_parameter_update_fn(SMCState_var: SMCState, cumul_states, cumul_infos, cumul_ancestors, i, SMCInfo_var):
        # assuming p>=1 (p>=2 in the notations)
        T, _, M, P, _ = cumul_infos.shape
        proposal_states = cumul_infos.at[:,0,...].get()
        proposed_states = cumul_infos.at[:,1,...].get()
        barXtmp = proposed_states.at[i, :, 1:].get()
        tildeXtmpm1 = proposal_states.at[:, i, :, :- 1].get()
        sqnormdiff = jnp.sum(jnp.square(barXtmp - tildeXtmpm1), axis=-1)
        weights = SMCState_var.weights.reshape(M, P+1)

        # Need to figure why I need to do that:
        cumul_states = cumul_states.at[i].set(temperedsmcstate_to_array(SMCState_var))


        cov_particles = jnp.cov(particles, rowvar=False)
        return {'cov_particles': jnp.expand_dims(cov_particles, axis=0)}
    return mcmc_parameter_update_fn


def build_mcmc_kernel():
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

