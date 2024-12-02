from typing import Callable, NamedTuple

import src.SMC.online_waste_free.mcmc_kernels.adapt_esjd_gaussian_AR as adapt_esjd_gaussian_AR
import src.SMC.online_waste_free.mcmc_kernels.gaussian_238_empirical as gaussian_238_empirical

__all__ = [
    "mcmc_proposal",
    "gaussian_238_empirical_proposal",
    "make_proposal_cumul_compatible",
    "adapt_esjd_gaussian_AR_proposal",
]


class mcmc_proposal(NamedTuple):
    build_kernel: Callable
    mcmc_parameter_update_fn: Callable


def make_proposal_cumul_compatible(proposal: mcmc_proposal) -> mcmc_proposal:
    """
    Make a proposal compatible with the cumulative arguments
    """

    def mcmc_parameter_update_fn(state, cumul_states, cumul_infos, cumul_ancestors, i, info):
        return proposal.mcmc_parameter_update_fn(state, info)

    return mcmc_proposal(proposal.build_kernel, mcmc_parameter_update_fn)


"""
Classic Gaussian proposal, zero-mean, covariance matrix equal to the empirical covariance of the particles scaled by gamma.
Require an initial covariance matrix to be passed:
    e.g., 'covariance_matrix': jnp.array([jnp.eye(dim)] * num_particles)
    or 'covariance_matrix': jnp.expand_dims(jnp.eye(dim), axis=0) (passed to all particles)
"""
gaussian_238_empirical_proposal = lambda: make_proposal_cumul_compatible(
    mcmc_proposal(gaussian_238_empirical.build_mcmc_kernel(),
                  gaussian_238_empirical.mcmc_parameter_update_fn))

adapt_esjd_gaussian_AR_proposal = lambda C: mcmc_proposal(adapt_esjd_gaussian_AR.build_mcmc_kernel(),
                                                          adapt_esjd_gaussian_AR.get_mcmc_parameter_update_fn(C))
