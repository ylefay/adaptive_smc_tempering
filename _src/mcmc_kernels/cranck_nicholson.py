import jax.numpy as jnp
from blackjax.mcmc.random_walk import generate_gaussian_noise
from blackjax.mcmc.random_walk import init, build_additive_step

__all__ = [
    "mcmc_parameter_update_fn",
    "build_kernel"
]


def mcmc_parameter_update_fn(_, __):
    return {}


def build_kernel(delta, C):
    """
    Auxiliary gradient-based sampling algorithms, Titsias, Papaspiliopoulos, 2018
    Eq. 4 in https://arxiv.org/pdf/1610.09641
    """

    def propose(rng_key, position):
        x = position[0]
        P = (delta * (delta + 4)) / (2 + delta) ** 2 * C
        cholP = jnp.linalg.cholesky(P)
        return generate_gaussian_noise(rng_key, position=position, mu=2 / (2 + delta) * x - x, sigma=
        cholP)

    def kernel(rng_key, state, logdensity_fn):
        state = init(position=state.position, logdensity_fn=logdensity_fn)
        new_state, info = build_additive_step()(rng_key, state, logdensity_fn, propose)
        return new_state, info

    return kernel
