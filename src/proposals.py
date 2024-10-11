import blackjax
import jax


def build_crank_nicholson_kernel(delta, C):
    """
    Auxiliary gradient-based sampling algorithms, Titsias, Papaspiliopoulos, 2018
    Eq. 4 in https://arxiv.org/pdf/1610.09641
    """

    def kernel(rng_key, state, logdensity_fn, **kwargs):
        x = state.position[0]
        position = 2 / (2 + delta) * x
        rw_gaussian = blackjax.additive_step_random_walk(logdensity_fn, delta * (delta + 4) / (2 + delta) ** 2 * C)
        state = rw_gaussian.init(position)
        #step = jax.jit(rw_gaussian.step)
        new_state, info = rw_gaussian.step(rng_key, state)
        return new_state, info

    return kernel
