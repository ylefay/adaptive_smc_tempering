from typing import Callable, Tuple

import jax.random
from blackjax.smc.resampling import multinomial
from jax import numpy as jnp, Array
from jax.typing import ArrayLike

PRNGKey = jax.Array


def accept_reject_MH_step(key: PRNGKey, log_density_for_proposed, log_density_for_current,
                          log_proposal_for_current, log_proposal_for_proposed) -> Array:
    logU = -jax.random.exponential(key)
    log_mh_ratio = jax.lax.min(0.,
                               log_density_for_proposed - log_density_for_current + log_proposal_for_current - log_proposal_for_proposed)
    return logU <= log_mh_ratio


def normalize_log_weights(log_weights: ArrayLike) -> ArrayLike:
    return log_weights - jax.scipy.special.logsumexp(log_weights)


def sq_distance(x: ArrayLike, y: ArrayLike, _: ArrayLike, __: int) -> ArrayLike:
    return jnp.sum(jnp.square(x - y), axis=-1)


def mahalanobis(x: ArrayLike, y: ArrayLike, particles: ArrayLike, i: int) -> ArrayLike:
    """
    At iteration i, for particles x, and proposed particles y, compute the Mahalanobis distances between x and y.
    The scaling matrix is the estimated covariance of the particles at iteration i - 1.
    """
    dim = particles.shape[-1]

    def _mahalanobis(x, y):
        if dim > 1:
            cov = jnp.cov(particles.at[i - 1].get().reshape((particles.shape[1] * particles.shape[2]),
                                                            dim), rowvar=False)
        else:
            cov = jnp.var(particles.at[i - 1].get().reshape((particles.shape[1] * particles.shape[2]),
                                                            dim), axis=0).reshape((1, 1))
        return jnp.einsum('j,k,jk->', x - y, x - y, jnp.linalg.inv(cov))

    return jax.lax.select(i == 0, jnp.sum(jnp.square(x - y), axis=-1), _mahalanobis(x, y))


def ess(weights):
    return weights.shape[-1] / jnp.sum(weights ** 2, axis=-1)


class GenericAdaptiveWasteFreeTemperingSMC:
    """
    A class that implements an adaptive Tempering SMC algorithm with waste-free SMC using parametric random-walk Metropolis-Hastings proposals.
    """

    def __init__(self, logbase_density_fn: Callable[[ArrayLike], ArrayLike],
                 base_measure_sampler: Callable[[PRNGKey], ArrayLike],
                 log_likelihood_fn: Callable[[ArrayLike], ArrayLike],
                 build_mh_proposal: Callable[
                     [ArrayLike, ArrayLike, int], Tuple[Callable[[ArrayLike, ArrayLike], ArrayLike],
                     Callable[[PRNGKey, ArrayLike], ArrayLike]]],
                 optimisation: Callable[[Callable[[ArrayLike], ArrayLike], ArrayLike], ArrayLike] = None,
                 criteria_function: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike] = sq_distance
                 ) -> None:
        self.logbase_density_fn = logbase_density_fn
        self.base_measure_sampler = base_measure_sampler
        self.log_likelihood_fn = log_likelihood_fn
        self.build_mh_proposal = build_mh_proposal
        self.optimisation = optimisation
        self.criteria_function = criteria_function

    def log_weights_fn(self, dlmbda):
        def _log_weights_fn(x):
            return dlmbda * self.log_likelihood_fn(x)

        return _log_weights_fn

    def sample(self, key: PRNGKey, num_parallel_chain: int, num_mcmc_steps: int,
               initial_mh_proposal_parameter: ArrayLike, tempering_sequence: ArrayLike) -> Tuple[
        ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """

        Parameters
        ----------
        key: PRNGKey
            A JAX PRNGKey
        num_parallel_chain: int
            Number of particles
        num_mcmc_steps: int
            Number of MCMC steps
        initial_mh_proposal_parameter: ArrayLike
            Initial parameter for the first mh proposal
        """
        GRID_CRITERIA = jnp.linspace(0.01, 10, 50)
        diff_tempering_sequence = jnp.diff(tempering_sequence)
        iteration = len(diff_tempering_sequence)
        diff_tempering_sequence = jnp.insert(diff_tempering_sequence, 0, 0.)
        criteria = jnp.zeros((iteration, GRID_CRITERIA.shape[-1]))
        P = num_mcmc_steps + 1
        num_particles = num_parallel_chain * P

        subkey = jax.random.fold_in(key, -1)
        subkeys = jax.random.split(subkey, num_particles)
        init_particles = self.base_measure_sampler(subkeys)
        shape = init_particles.shape[1:]
        particles = jnp.zeros((iteration + 1, num_parallel_chain, P, *shape))
        particles = particles.at[0].set(init_particles.reshape((num_parallel_chain, P, *shape)))

        mh_proposal_parameters = jnp.zeros((iteration + 1, *initial_mh_proposal_parameter.shape))
        mh_proposal_parameters = mh_proposal_parameters.at[0].set(initial_mh_proposal_parameter)

        subkey = jax.random.fold_in(key, 0)
        subkeys = jax.random.split(subkey, num_particles)

        log_proposal, proposal_sampler = self.build_mh_proposal(initial_mh_proposal_parameter, particles, 0)
        init_proposed_particles = jax.vmap(proposal_sampler, in_axes=(0, 0))(subkeys, init_particles)

        log_G0_fn = jax.vmap(self.log_weights_fn(diff_tempering_sequence.at[0].get()))
        vmapped_logbase_density_fn = jax.vmap(self.logbase_density_fn)

        log_G0_init_particles = log_G0_fn(init_particles)
        init_log_weights = normalize_log_weights(log_G0_init_particles)
        init_log_weights = init_log_weights.reshape((num_parallel_chain, P))

        log_weights = jnp.zeros((iteration + 1, num_parallel_chain, P))
        log_weights = log_weights.at[0].set(init_log_weights)

        log_g_1 = log_G0_fn(init_proposed_particles) - log_G0_fn(init_particles) + vmapped_logbase_density_fn(
            init_proposed_particles) - vmapped_logbase_density_fn(init_particles)
        d_1 = jax.vmap(self.criteria_function, in_axes=(0, 0, None, None))(init_proposed_particles, init_particles,
                                                                           particles, 0)

        if self.optimisation:
            @jax.jit
            def m_estimate_of_criteria_function(param):
                _new_particles = init_particles.reshape((num_particles, *shape))
                _new_proposed_particles = init_proposed_particles.reshape((num_particles, *shape))
                _log_proposal_fn, _ = self.build_mh_proposal(param, particles, 1)
                log_proposal_fn = jax.vmap(_log_proposal_fn)
                log_proposal_from_particles_to_proposed_particles = log_proposal_fn(_new_particles,
                                                                                    _new_proposed_particles)
                diff_log_proposal = log_proposal_fn(_new_proposed_particles,
                                                    _new_particles) - log_proposal_from_particles_to_proposed_particles
                log_ratio = diff_log_proposal + log_g_1
                to_sum = d_1 * jax.lax.min(1., jnp.exp(log_ratio))
                # / num_particles
                return jnp.sum(to_sum)

            criteria = criteria.at[0].set(jax.vmap(m_estimate_of_criteria_function)(GRID_CRITERIA))
            new_mh_proposal_parameter = self.optimisation(m_estimate_of_criteria_function,
                                                          initial_mh_proposal_parameter)
        else:
            new_mh_proposal_parameter = initial_mh_proposal_parameter
        mh_proposal_parameters = mh_proposal_parameters.at[1].set(new_mh_proposal_parameter)

        init_particles = init_particles.reshape((num_parallel_chain, P, *shape))
        particles = particles.at[0].set(init_particles)

        def body_fn(i, carry):
            particles, log_weights, mh_proposal_parameters, acceptance_bools, important_sampling_log_weights_from_proposal_to_new_proposal, criteria = carry
            subkey = jax.random.fold_in(key, i)
            ancestors = multinomial(subkey, jnp.exp(log_weights.at[i - 1].get().reshape(-1)), num_parallel_chain)
            resampled_particles = \
                particles.at[i - 1].get().reshape((num_particles, *shape)).at[
                    ancestors].get()
            log_target_density_at_t_fn = lambda x: self.log_weights_fn(tempering_sequence.at[i].get())(
                x) + self.logbase_density_fn(x)
            log_target_density_at_t_minus_one_fn = lambda x: self.log_weights_fn(
                tempering_sequence.at[i - 1].get())(x) + self.logbase_density_fn(x)
            log_proposal, proposal_sampler = self.build_mh_proposal(mh_proposal_parameters.at[i].get(), particles,
                                                                    i)

            @jax.vmap
            def inside_body_fn(key, particle):
                _proposed_particles = jnp.zeros((num_mcmc_steps, *shape))
                _particles = jnp.zeros((P, *shape))
                _particles = _particles.at[0].set(particle)
                _log_g = jnp.zeros((num_mcmc_steps,))
                _d = jnp.zeros((num_mcmc_steps,))
                _log_q = jnp.zeros((num_mcmc_steps,))
                _acceptance_bools = jnp.zeros((num_mcmc_steps,), dtype=int)

                def inside_inside_body_fn(p, carry):
                    key, proposed_particles_across_p, particles_across_p, log_g_across_p, d_across_p, log_q_across_p, acceptance_bool_across_p, = carry
                    particle = particles_across_p.at[p - 1].get()
                    subkey_p = jax.random.fold_in(key, p)
                    _, key = jax.random.split(key)

                    new_proposed_particle = proposal_sampler(subkey_p, particle)
                    proposed_particles_across_p = proposed_particles_across_p.at[p - 1].set(new_proposed_particle)
                    new_log_g = log_target_density_at_t_fn(new_proposed_particle) - log_target_density_at_t_fn(particle)

                    new_d = self.criteria_function(particle, new_proposed_particle, particles, i)
                    log_g_across_p = log_g_across_p.at[p - 1].set(new_log_g)
                    d_across_p = d_across_p.at[p - 1].set(new_d)
                    new_log_q = log_proposal(particle, new_proposed_particle)
                    log_q_across_p = log_q_across_p.at[p - 1].set(new_log_q)
                    log_target_density_at_t_minus_one_fn_particle = log_target_density_at_t_minus_one_fn(particle)
                    log_current_tgt_density_new_proposed_particle = log_target_density_at_t_minus_one_fn(
                        new_proposed_particle)

                    accept_MH_boolean = accept_reject_MH_step(key,
                                                              log_current_tgt_density_new_proposed_particle,
                                                              log_target_density_at_t_minus_one_fn_particle,
                                                              log_proposal(new_proposed_particle, particle),
                                                              new_log_q
                                                              )
                    new_particle = jax.lax.select(accept_MH_boolean, new_proposed_particle, particle)

                    particles_across_p = particles_across_p.at[p].set(new_particle)
                    acceptance_bool_across_p = acceptance_bool_across_p.at[p - 1].set(accept_MH_boolean)
                    return key, proposed_particles_across_p, particles_across_p, log_g_across_p, d_across_p, log_q_across_p, acceptance_bool_across_p

                _, _proposed_particles, _particles, _log_g, _d, _log_q, _acceptance_bools = jax.lax.fori_loop(1,
                                                                                                              P,
                                                                                                              inside_inside_body_fn,
                                                                                                              (
                                                                                                                  key,
                                                                                                                  _proposed_particles,
                                                                                                                  _particles,
                                                                                                                  _log_g,
                                                                                                                  _d,
                                                                                                                  _log_q,
                                                                                                                  _acceptance_bools)
                                                                                                              )
                return _particles, _proposed_particles, _log_g, _d, _log_q, _acceptance_bools

            keys = jax.random.split(subkey, num_parallel_chain)
            new_particles, new_proposed_particles, new_log_g, new_d, new_log_q, new_acceptance_bools = inside_body_fn(
                keys, resampled_particles)

            particles = particles.at[i].set(new_particles)
            log_Gi_fn = jax.vmap(self.log_weights_fn(diff_tempering_sequence.at[i].get()))
            new_log_weights = log_Gi_fn(new_particles.reshape(num_particles, *shape)).reshape(
                (num_parallel_chain, P))
            new_log_weights = normalize_log_weights(new_log_weights)
            log_weights = log_weights.at[i].set(new_log_weights)
            truncated_weights = jnp.exp(normalize_log_weights(new_log_weights.at[:, 1:].get()))

            if self.optimisation:
                @jax.jit
                def m_estimate_of_criteria_function(param: ArrayLike) -> ArrayLike:
                    """
                    Estimate of the criteria function for the next iteration using current samples.
                    See the notes.
                    """
                    _new_particles = new_particles.at[:, :-1, :].get().reshape(
                        (num_particles - num_parallel_chain,
                         *shape))  # there was a silly mistake here, new_particles[:,1:, :] instead of [:,:-1,:],
                    # thus proposed particles and new_particles had particles in common, yielding pdf of the form 1/scale exp(-0^2/scale**2) -> 1/scale.
                    _new_proposed_particles = new_proposed_particles.reshape(
                        (num_particles - num_parallel_chain, *shape))
                    _log_proposal_fn, _ = self.build_mh_proposal(param, particles, i + 1)
                    log_proposal_fn = jax.vmap(_log_proposal_fn)
                    log_proposal_from_particles_to_proposed_particles = log_proposal_fn(_new_particles,
                                                                                        _new_proposed_particles)

                    diff_log_proposal = log_proposal_fn(_new_proposed_particles,
                                                        _new_particles) - log_proposal_from_particles_to_proposed_particles
                    log_ratio = diff_log_proposal.reshape((num_parallel_chain, num_mcmc_steps)) + new_log_g

                    importance_sampling_log_weight_from_proposal_to_new_proposal = log_proposal_from_particles_to_proposed_particles.reshape(
                        (num_parallel_chain, num_mcmc_steps)) - new_log_q

                    to_sum = truncated_weights * new_d * \
                             jnp.exp(jax.lax.min(0.,
                                                 log_ratio) + importance_sampling_log_weight_from_proposal_to_new_proposal)
                    # / (num_parallel_chain * num_mcmc_steps)
                    return jnp.sum(to_sum)

                new_mh_proposal_parameter = self.optimisation(m_estimate_of_criteria_function,
                                                              mh_proposal_parameters.at[i].get())
                criteria = criteria.at[i].set(jax.vmap(m_estimate_of_criteria_function)(GRID_CRITERIA))
            else:
                new_mh_proposal_parameter = initial_mh_proposal_parameter  # need to implement a minimization procedure

            acceptance_bools = acceptance_bools.at[i - 1].set(new_acceptance_bools)

            # The next lines are only useful for saving the IS weights
            _new_particles = new_particles.at[:, 1:, :].get().reshape(
                (num_particles - num_parallel_chain, *shape))
            _new_proposed_particles = new_proposed_particles.reshape(
                (num_particles - num_parallel_chain, *shape))
            _log_proposal_fn, _ = self.build_mh_proposal(new_mh_proposal_parameter, particles, i + 1)
            log_proposal_fn = jax.vmap(_log_proposal_fn)
            log_proposal_from_particles_to_proposed_particles = log_proposal_fn(_new_particles,
                                                                                _new_proposed_particles)
            new_important_sampling_log_weights_from_proposal_to_new_proposal = log_proposal_from_particles_to_proposed_particles.reshape(
                (num_parallel_chain, num_mcmc_steps)) - new_log_q
            important_sampling_log_weights_from_proposal_to_new_proposal = \
                important_sampling_log_weights_from_proposal_to_new_proposal.at[i - 1].set(
                    new_important_sampling_log_weights_from_proposal_to_new_proposal)

            mh_proposal_parameters = mh_proposal_parameters.at[i + 1].set(new_mh_proposal_parameter)
            return particles, log_weights, mh_proposal_parameters, acceptance_bools, important_sampling_log_weights_from_proposal_to_new_proposal, criteria

        acceptance_bools = jnp.zeros((iteration, num_parallel_chain, num_mcmc_steps), dtype=int)
        important_sampling_log_weights_from_proposal_to_new_proposal = jnp.zeros(
            (iteration, num_parallel_chain, num_mcmc_steps))
        particles, log_weights, mh_proposal_parameters, acceptance_bools, important_sampling_log_weights_from_proposal_to_new_proposal, criteria = jax.lax.fori_loop(
            1, iteration + 1,
            body_fn, (
                particles,
                log_weights,
                mh_proposal_parameters,
                acceptance_bools,
                important_sampling_log_weights_from_proposal_to_new_proposal,
                criteria))
        return particles, log_weights, mh_proposal_parameters, acceptance_bools, important_sampling_log_weights_from_proposal_to_new_proposal, criteria
