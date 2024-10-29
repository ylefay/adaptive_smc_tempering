import jax
import jax.numpy as jnp


def logZ_logW(chain):
    r"""
    G_t(X_{t+1}) = chain[0].sampler_state.weights
    w_i = \prod_{t=0}^{T-1} G_t(X_{t+1})
    Normalize.
    -------

    """
    logG = jnp.log(chain[0].sampler_state.weights)
    logW = logG.cumsum(axis=1)
    logZ = jax.scipy.special.logsumexp(logW, axis=-1)
    logW = logW - logZ[..., None]
    return logZ, logW


def esjd(chain):
    """acceptance_rate = chain[1].update_info.acceptance_rate
    proposals = chain[1].update_info.proposal.position[0]"""
    particles = chain[0].sampler_state.particles[0]
    dists = jnp.linalg.norm(particles[:, 1:, ...] - particles[:, :-1, ...], axis=-1)
    ESJDs = dists.mean(axis=-1)
    return ESJDs
