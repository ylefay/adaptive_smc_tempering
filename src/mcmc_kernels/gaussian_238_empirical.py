import jax.numpy as jnp
from blackjax.mcmc.random_walk import generate_gaussian_noise
from blackjax.mcmc.random_walk import init, build_additive_step

__all__ = [
    "mcmc_parameter_update_fn",
    "build_mcmc_kernel"
]


def mcmc_parameter_update_fn(SMCState_var, SMCInfo_var):
    particles = SMCState_var[0][0]
    cov_particles = jnp.cov(particles, rowvar=False)
    return {'cov_particles': jnp.array([cov_particles] * particles.shape[0])}


def build_mcmc_kernel():
    """
    Default MCMC kernel for the inner kernel of the SMC sampler.
    Gaussian proposal, zero-mean, covariance matrix equal to the empirical covariance of the particles scaled by gamma.
    -------

    """

    def kernel(rng_key, state, logdensity_fn, cov_particles):
        def propose(rng_key, position):
            x = position[0]
            gamma = 2.38 / x.shape[0] ** 0.5
            C = jnp.linalg.cholesky(cov_particles) * gamma
            return generate_gaussian_noise(rng_key, position=position, mu=jnp.zeros_like(x), sigma=C)

        state = init(position=state.position, logdensity_fn=logdensity_fn)
        new_state, info = build_additive_step()(rng_key, state, logdensity_fn, propose)
        return new_state, info

    return kernel
