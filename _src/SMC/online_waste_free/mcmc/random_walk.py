from typing import Callable, Optional, NamedTuple

import jax
import jax.numpy as jnp
from blackjax.mcmc import proposal
from blackjax.types import PRNGKey
from blackjax.mcmc.random_walk import RWState, build_rmh_transition_energy

class RWInfo(NamedTuple):
    """Additional information on the RW chain.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    proposal
        The state proposed by the proposal.

    """

    acceptance_rate: float
    is_accepted: bool
    proposal: RWState
    proposed_states: RWState

def build_additive_step():
    """Similar to blackjax.mcmc.random_walk.build_additive_step
    """

    def kernel(
            rng_key: PRNGKey, state: RWState, logdensity_fn: Callable, random_step: Callable
    ) -> tuple[RWState, RWInfo]:
        def proposal_generator(key_proposal, position):
            move_proposal = random_step(key_proposal, position)
            new_position = jax.tree_util.tree_map(jnp.add, position, move_proposal)
            return new_position

        inner_kernel = build_rmh()
        return inner_kernel(rng_key, state, logdensity_fn, proposal_generator)

    return kernel


def build_rmh():
    """Similar to blackjax.mcmc.random_walk.build_rmh
    """

    def kernel(
            rng_key: PRNGKey,
            state: RWState,
            logdensity_fn: Callable,
            transition_generator: Callable,
            proposal_logdensity_fn: Optional[Callable] = None,
    ) -> tuple[RWState, RWInfo]:
        """Move the chain by one step using the Rosenbluth Metropolis Hastings
        algorithm.

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random
           numbers.
        logdensity_fn:
            A function that returns the log-probability at a given position.
        transition_generator:
            A function that generates a candidate transition for the markov chain.
        proposal_logdensity_fn:
            For non-symmetric proposals, a function that returns the log-density
            to obtain a given proposal knowing the current state. If it is not
            provided we assume the proposal is symmetric.
        state:
            The current state of the chain.

        Returns
        -------
        The next state of the chain and additional information about the current
        step, including proposed states.

        """
        transition_energy = build_rmh_transition_energy(proposal_logdensity_fn)

        compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
            transition_energy
        )

        proposal_generator = rmh_proposal(
            logdensity_fn, transition_generator, compute_acceptance_ratio
        )
        new_state, do_accept, p_accept, proposed_state = proposal_generator(rng_key, state)
        return new_state, RWInfo(p_accept, do_accept, new_state, proposed_state)

    return kernel


def rmh_proposal(
        logdensity_fn: Callable,
        transition_distribution: Callable,
        compute_acceptance_ratio: Callable,
        sample_proposal: Callable = proposal.static_binomial_sampling,
) -> Callable:
    """
    Similar to blackjax.smc.random_walk but with access to the proposed states
    """

    def generate(rng_key, previous_state: RWState) -> tuple[RWState, bool, float, RWState]:
        key_proposal, key_accept = jax.random.split(rng_key, 2)
        position, _ = previous_state
        new_position = transition_distribution(key_proposal, position)
        proposed_state = RWState(new_position, logdensity_fn(new_position))
        log_p_accept = compute_acceptance_ratio(previous_state, proposed_state)
        accepted_state, info = sample_proposal(
            key_accept, log_p_accept, previous_state, proposed_state
        )
        do_accept, p_accept, _ = info
        return accepted_state, do_accept, p_accept, proposed_state

    return generate
