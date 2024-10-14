from blackjax.mcmc.random_walk import additive_step_random_walk
from blackjax.mcmc.random_walk import generate_gaussian_noise
from blackjax.mcmc.random_walk import init, build_additive_step

def build_crank_nicholson_kernel(delta, C):
    """
    Auxiliary gradient-based sampling algorithms, Titsias, Papaspiliopoulos, 2018
    Eq. 4 in https://arxiv.org/pdf/1610.09641
    """

    def propose(rng_key, position):
        x = position[0]
        return generate_gaussian_noise(rng_key, position=position, mu=2 / (2 + delta) * x - 1, sigma=
        delta * (delta + 4) / (2 + delta) ** 2 * C)

    def cn_rw(logdensity_fn):
        return additive_step_random_walk(logdensity_fn, propose)

    def kernel(rng_key, state, logdensity_fn, **kwargs):
        #rw_gaussian = cn_rw(logdensity_fn)
        #state = rw_gaussian.init(state.position)
        #new_state, info = rw_gaussian.step(rng_key, state)
        state = init(position=state.position, logdensity_fn=logdensity_fn)
        new_state, info = build_additive_step()(rng_key, state, logdensity_fn, propose)
        return new_state, info

    return kernel
