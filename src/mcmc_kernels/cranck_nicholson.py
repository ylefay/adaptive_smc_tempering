from blackjax.mcmc.random_walk import generate_gaussian_noise
from blackjax.mcmc.random_walk import init, build_additive_step


def mcmc_parameter_update_fn_default(_, __):
    return {}


def build_crank_nicholson_kernel(delta, C):
    """
    Auxiliary gradient-based sampling algorithms, Titsias, Papaspiliopoulos, 2018
    Eq. 4 in https://arxiv.org/pdf/1610.09641
    """

    def propose(rng_key, position, _):
        x = position[0]
        return generate_gaussian_noise(rng_key, position=position, mu=2 / (2 + delta) * x - 1, sigma=
        delta * (delta + 4) / (2 + delta) ** 2 * C)

    def kernel(rng_key, state, logdensity_fn, **kwargs):
        state = init(position=state.position, logdensity_fn=logdensity_fn)
        new_state, info = build_additive_step()(rng_key, state, logdensity_fn, propose)
        return new_state, info

    return kernel
