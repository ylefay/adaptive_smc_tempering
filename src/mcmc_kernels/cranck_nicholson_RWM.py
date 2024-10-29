import jax.numpy as jnp
from blackjax.mcmc.random_walk import generate_gaussian_noise
from blackjax.mcmc.random_walk import init, build_additive_step


def mcmc_parameter_update_fn(SMCState_var, SMCInfo_var):
    particles = SMCState_var[0][0]
    lmbda = SMCState_var.lmbda
    cov_particles = jnp.cov(particles, rowvar=False)
    return {'lmbda': jnp.array([lmbda] * particles.shape[0]),
            'cov_particles': jnp.array([cov_particles] * particles.shape[0])}


def build_kernel(delta, C):
    """
    Auxiliary gradient-based sampling algorithms, Titsias, Papaspiliopoulos, 2018
    Eq. 4 in https://arxiv.org/pdf/1610.09641
    Interpolation between RWM and CNp (Eq. 4).
    """

    def kernel(rng_key, state, logdensity_fn, lmbda, cov_particles):
        def propose(rng_key, position):
            # jax.debug.print('{lmbda}', lmbda=lmbda)
            x = position[0]
            gamma = 2.38 ** 2 / x.shape[0]
            S = cov_particles * gamma
            P = (delta * (delta + 4)) / (2 + delta) ** 2 * C
            PropChol = jnp.linalg.cholesky(
                jnp.linalg.inv(jnp.linalg.inv(S) * lmbda + (1 - lmbda) * jnp.linalg.inv(P)))  # *lmbda *(1-lmbda)
            # jax.debug.print('{PropChol}', PropChol=PropChol)
            return generate_gaussian_noise(rng_key, position=position,
                                           mu=(1 - lmbda) ** 0.5 * (2 / (2 + delta) * x - x),
                                           sigma=PropChol)

        state = init(position=state.position, logdensity_fn=logdensity_fn)
        new_state, info = build_additive_step()(rng_key, state, logdensity_fn, propose)
        return new_state, info

    return kernel
