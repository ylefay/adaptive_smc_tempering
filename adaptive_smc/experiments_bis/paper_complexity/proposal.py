from adaptive_smc.smc_types import SMCStatebis, LogDensity
from typing import Optional
import jax.numpy as jnp
from adaptive_smc.proposals.rw import build_gaussian_rw_proposal


def build_gaussian_rw_proposal_fixed_scaling(state: SMCStatebis, _: LogDensity, __: LogDensity, i: int,
                                             j: Optional[int] = None):
    particles = state.particles
    dim = particles.shape[-1]
    optimal_scale = 2.38 ** 2 / dim

    return build_gaussian_rw_proposal(jnp.eye(dim) * optimal_scale)
