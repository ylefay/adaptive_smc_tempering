import jax
from jax.typing import ArrayLike

PRNGKey = jax.Array


def accept_reject_MH_step(key: PRNGKey, log_density_for_proposed, log_density_for_current,
                          log_proposal_for_current, log_proposal_for_proposed) -> ArrayLike:
    logU = -jax.random.exponential(key)
    log_mh_ratio = jax.lax.min(0.,
                               log_density_for_proposed - log_density_for_current + log_proposal_for_current - log_proposal_for_proposed)
    return logU <= log_mh_ratio
