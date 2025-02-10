from typing import Callable, Tuple, Optional

import jax.random
from blackjax.smc.resampling import multinomial
from blackjax.smc.solver import dichotomy
from jax import numpy as jnp, Array
from jax.typing import ArrayLike

from adaptive_smc.criteria_functions import square_distance

PRNGKey = jax.Array


def accept_reject_MH_step(key: PRNGKey, log_density_for_proposed, log_density_for_current,
                          log_proposal_for_current, log_proposal_for_proposed) -> Array:
    logU = -jax.random.exponential(key)
    log_mh_ratio = jax.lax.min(0.,
                               log_density_for_proposed - log_density_for_current + log_proposal_for_current - log_proposal_for_proposed)
    return logU <= log_mh_ratio


def normalize_log_weights(log_weights: ArrayLike) -> Tuple[ArrayLike, float]:
    log_normalization = jax.scipy.special.logsumexp(log_weights)
    return log_weights - log_normalization, log_normalization - jnp.log(log_weights.shape[0] * log_weights.shape[1])


def log_ess(delta: float, log_weights: Array) -> float:
    """
    See Algorithm 17.3,
    Introduction to Sequential Monte Carlo, Chopin, Papaspiliopoulos
    """
    N_particles = jnp.prod(jnp.array(log_weights.shape))
    log_ess = 2 * jax.scipy.special.logsumexp(delta * log_weights) - jax.scipy.special.logsumexp(
        2 * delta * log_weights)
    log_ess_scaled = log_ess - jnp.log(N_particles)
    return log_ess_scaled


class GenericAdaptiveWasteFreeTemperingSMC:
    """
    A class that implements an adaptive Tempering SMC algorithm with waste-free SMC using parametric random-walk Metropolis-Hastings proposals.
    """

    def __init__(self, logbase_density_fn: Callable[[ArrayLike], ArrayLike],
                 base_measure_sampler: Callable[[PRNGKey], ArrayLike],
                 log_likelihood_fn: Callable[[ArrayLike], ArrayLike],
                 build_mh_proposal: Callable[
                     [ArrayLike, ArrayLike, Optional[ArrayLike], Callable, int], Tuple[
                         Callable[[ArrayLike, ArrayLike], ArrayLike],
                         Callable[[PRNGKey, ArrayLike], ArrayLike]]],
                 optimisation: Callable[[Callable[[ArrayLike], ArrayLike], ArrayLike], ArrayLike] = None,
                 criteria_function: Callable[[ArrayLike, ArrayLike, ArrayLike, int], ArrayLike] = square_distance
                 ) -> None:
        self.logbase_density_fn = logbase_density_fn
        self.base_measure_sampler = base_measure_sampler
        self.vmapped_base_measure_sampler = jnp.vectorize(base_measure_sampler,
                                                          signature='(2)->(d)')  # Assuming a vmappable function
        self.log_likelihood_fn = log_likelihood_fn
        self.vmapped_log_likelihood_fn = jax.vmap(log_likelihood_fn)
        self.build_mh_proposal = build_mh_proposal
        self.optimisation = optimisation
        self.criteria_function = criteria_function

    def log_tgt_fn(self, lmbda):
        def _log_tgt_fn(x):
            return lmbda * self.log_likelihood_fn(x) + self.logbase_density_fn(x)

        return _log_tgt_fn

    def log_weights_fn(self, dlmbda):
        def _log_weights_fn(x):
            return dlmbda * self.log_likelihood_fn(x)

        return _log_weights_fn

    def sample(self, key: PRNGKey, num_parallel_chain: int, num_mcmc_steps: int,
               initial_mh_proposal_parameter: ArrayLike,
               tempering_sequence: ArrayLike,
               target_ess: Optional[float] = None) -> Tuple[
        ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        r"""

        Parameters
        ----------
        key: PRNGKey
            A JAX PRNGKey
        num_parallel_chain: int
            Number of particles, M in the notes
        num_mcmc_steps: int
            Number of MCMC steps, P - 1 in the notes
        initial_mh_proposal_parameter: ArrayLike
            Initial parameter for the first mh proposal, \theta_0 in the notes
        target_ess: Optional, float
            if not None, then adaptively find the optimal tempering step to achieve an ESS greater than target_ess

        Remarks:
            if tempering_sequence starts at 0, then the 0-th target density is the base measure.
            tempering_sequence = (\lambda_0, \ldots, \lambda_T)
            diff_tempering_sequence = (\lambda_0, \lambda_1-\lambda_0, \ldots, \lambda_T-\lambda_{T-1})
            particles is a (T + 1, M, P, dim) arrays containing for each iteration 0<=t<=T), the particles X_t^{p, m},
            such that (X_t^n) has marginal \pi_{t-1}, with \pi_{-1} = the base measure
        """
        iteration = len(tempering_sequence)
        diff_tempering_sequence = jnp.diff(tempering_sequence)
        diff_tempering_sequence = jnp.insert(diff_tempering_sequence, 0, tempering_sequence.at[0].get())

        GRID_CRITERIA = jnp.linspace(0.01, 8, 100)
        criteria = jnp.zeros((iteration, GRID_CRITERIA.shape[-1]))

        P = num_mcmc_steps + 1
        num_particles = num_parallel_chain * P

        subkey = jax.random.fold_in(key, -1)
        subkeys = jax.random.split(subkey, (num_parallel_chain, P))

        init_particles = self.vmapped_base_measure_sampler(subkeys)
        dim = init_particles.shape[-1]
        init_particles = init_particles.reshape((num_parallel_chain, P, dim))
        particles = jnp.zeros((iteration + 1, num_parallel_chain, P, dim))
        particles = particles.at[0].set(init_particles)

        mh_proposal_parameters = jnp.zeros((iteration + 1, *initial_mh_proposal_parameter.shape))
        mh_proposal_parameters = mh_proposal_parameters.at[0].set(initial_mh_proposal_parameter)

        log_normalizations = jnp.zeros((iteration + 1, ))

        subkey = jax.random.fold_in(key, 0)
        subkeys = jax.random.split(subkey, num_particles)

        log_weights = jnp.zeros((iteration + 1, num_parallel_chain, P))

        _log_proposal_fn, proposal_sampler = self.build_mh_proposal(initial_mh_proposal_parameter, particles,
                                                                    log_weights,
                                                                    self.log_tgt_fn(tempering_sequence.at[0].get()),
                                                                    0)
        log_proposal_fn = jnp.vectorize(_log_proposal_fn, signature="(d),(d)->()")
        init_proposed_particles = jax.vmap(proposal_sampler, in_axes=(0, 0))(subkeys, init_particles.reshape(
            (num_particles, dim))).reshape((num_parallel_chain, P, dim))
        if target_ess:
            _log_weights = self.vmapped_log_likelihood_fn(init_particles)
            dlmbda = dichotomy(lambda dlmbda: log_ess(dlmbda, _log_weights) - jnp.log(target_ess), 0., 1.0, 1e-2, 10)
            dlmbda = jnp.clip(dlmbda, 0., 1.0)
            tempering_sequence = tempering_sequence.at[0].set(dlmbda)
            diff_tempering_sequence = diff_tempering_sequence.at[0].set(dlmbda)

        log_G0_fn = jax.vmap(self.log_weights_fn(diff_tempering_sequence.at[0].get()))
        vmapped_logbase_density_fn = jax.vmap(self.logbase_density_fn)

        log_G0_init_particles = log_G0_fn(init_particles)
        init_log_weights, log_normalization = normalize_log_weights(log_G0_init_particles)

        log_normalizations = log_normalizations.at[0].set(log_normalization)
        log_weights = log_weights.at[0].set(init_log_weights)

        log_g_1 = log_G0_fn(init_proposed_particles) - log_G0_fn(init_particles) + vmapped_logbase_density_fn(
            init_proposed_particles) - vmapped_logbase_density_fn(init_particles)
        d_1 = jax.vmap(self.criteria_function, in_axes=(0, 0, None, None))(init_proposed_particles, init_particles,
                                                                           particles, 0)
        new_log_q = log_proposal_fn(init_particles, init_proposed_particles)

        if self.optimisation:
            @jax.jit
            def m_estimate_of_criteria_function(param):
                _log_proposal_param_fn, _ = self.build_mh_proposal(param, particles, log_weights,
                                                                   self.log_tgt_fn(tempering_sequence.at[1].get()), 1)
                log_proposal_param_fn = jnp.vectorize(_log_proposal_fn, signature="(d),(d)->()")
                log_proposal_from_particles_to_proposed_particles = log_proposal_param_fn(init_particles,
                                                                                          init_proposed_particles)
                diff_log_proposal = log_proposal_param_fn(init_proposed_particles,
                                                          init_particles) - log_proposal_from_particles_to_proposed_particles
                log_ratio = diff_log_proposal + log_g_1

                importance_sampling_log_weight_from_proposal_to_new_proposal = log_proposal_from_particles_to_proposed_particles - new_log_q

                to_sum = init_log_weights * d_1 * jnp.exp(jax.lax.min(0., log_ratio) +
                    importance_sampling_log_weight_from_proposal_to_new_proposal)
                return jnp.sum(to_sum)

            criteria = criteria.at[0].set(jax.vmap(m_estimate_of_criteria_function)(GRID_CRITERIA))
            new_mh_proposal_parameter = self.optimisation(m_estimate_of_criteria_function,
                                                          initial_mh_proposal_parameter)
        else:
            new_mh_proposal_parameter = initial_mh_proposal_parameter
        mh_proposal_parameters = mh_proposal_parameters.at[1].set(new_mh_proposal_parameter)

        def make_inner_loop(i: int, prev_lmbda: float, mh_proposal_parameter: ArrayLike,
                            particles: ArrayLike):
            """
            Given the iteration 1<=i<=T, construct the inner loop function (over (1<=m<=M, 1<=p<=P)
            """

            log_target_density_at_t_minus_one_fn = lambda x: self.log_weights_fn(prev_lmbda)(
                x) + self.logbase_density_fn(x)
            log_proposal, proposal_sampler = self.build_mh_proposal(mh_proposal_parameter, particles, log_weights,
                                                                    self.log_tgt_fn(tempering_sequence.at[i].get()), i)

            @jax.vmap
            def inside_body_fn(key, particle):
                _proposed_particles = jnp.zeros((num_mcmc_steps, dim))
                _particles = jnp.zeros((P, dim))
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
                    new_log_g = self.log_likelihood_fn(new_proposed_particle) - self.log_likelihood_fn(
                        particle)  # still need to multiply by \lambda_t and add the base measure samplre

                    new_d = self.criteria_function(particle, new_proposed_particle, particles, i)
                    log_g_across_p = log_g_across_p.at[p - 1].set(
                        new_log_g)  # need to rename log_g into log_ratio_likelihood
                    new_log_q = log_proposal(particle, new_proposed_particle)

                    d_across_p = d_across_p.at[p - 1].set(new_d)
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

            return inside_body_fn

        def body_fn(i, carry):
            particles, log_weights, mh_proposal_parameters, acceptance_bools, important_sampling_log_weights_from_proposal_to_new_proposal, criteria, tempering_sequence, diff_tempering_sequence, log_normalizations = carry
            subkey = jax.random.fold_in(key, i)
            ancestors = multinomial(subkey, jnp.exp(log_weights.at[i - 1].get().reshape(-1)), num_parallel_chain)
            resampled_particles = particles.at[i - 1].get().reshape((num_particles, dim)).at[ancestors].get()

            if target_ess:
                _log_weights = self.vmapped_log_likelihood_fn(
                    particles.at[i - 1].get())  # do not use new_particles, this is wrong
                dlmbda = dichotomy(lambda dlmbda: log_ess(dlmbda, _log_weights) - jnp.log(target_ess), 0.,
                                   1.0 - tempering_sequence.at[i - 1].get(), 1e-2, 10)
                dlmbda = jnp.clip(dlmbda, 0., 1.0 - tempering_sequence.at[i - 1].get())
                tempering_sequence = tempering_sequence.at[i].set(tempering_sequence.at[i - 1].get() + dlmbda)
                diff_tempering_sequence = diff_tempering_sequence.at[i].set(dlmbda)

            inside_body_fn = make_inner_loop(i, tempering_sequence.at[i - 1].get(),
                                             mh_proposal_parameters.at[i].get(), particles)

            keys = jax.random.split(subkey, num_parallel_chain)
            new_particles, new_proposed_particles, new_log_g, new_d, new_log_q, new_acceptance_bools = inside_body_fn(
                keys, resampled_particles)
            ###
            new_log_g = new_log_g * tempering_sequence.at[i].get() + vmapped_logbase_density_fn(
                new_proposed_particles) - vmapped_logbase_density_fn(new_particles.at[:, 1:].get())
            # since we may be updating the temperatures using ESS, we do not compute g inside the loop, we first decide of the temperature and then adjust to compute g

            particles = particles.at[i].set(new_particles)

            log_Gi_fn = jax.vmap(self.log_weights_fn(diff_tempering_sequence.at[i].get()))

            new_log_weights = log_Gi_fn(new_particles)
            new_log_weights, log_normalization = normalize_log_weights(new_log_weights)
            log_normalizations = log_normalizations.at[i].set(log_normalization)
            log_weights = log_weights.at[i].set(new_log_weights)
            truncated_weights = jnp.exp(normalize_log_weights(new_log_weights.at[:, 1:].get())[0])

            if self.optimisation:
                @jax.jit
                def m_estimate_of_criteria_function(param: ArrayLike) -> ArrayLike:
                    """
                    Estimate of the criteria function for the next iteration using current samples.
                    See the notes.
                    """
                    _new_particles = new_particles.at[:, :-1, :].get()
                    _new_proposed_particles = new_proposed_particles
                    _log_proposal_fn, _ = self.build_mh_proposal(param, particles, log_weights,
                                                                 self.log_tgt_fn(tempering_sequence.at[i + 1].get()),
                                                                 i + 1)
                    log_proposal_fn = jnp.vectorize(_log_proposal_fn, signature="(d),(d)->()")
                    log_proposal_from_particles_to_proposed_particles = log_proposal_fn(_new_particles,
                                                                                        _new_proposed_particles)

                    diff_log_proposal = log_proposal_fn(_new_proposed_particles,
                                                        _new_particles) - log_proposal_from_particles_to_proposed_particles
                    log_ratio = diff_log_proposal + new_log_g

                    importance_sampling_log_weight_from_proposal_to_new_proposal = log_proposal_from_particles_to_proposed_particles - new_log_q

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
            _new_particles = new_particles.at[:, 1:, :].get()
            _new_proposed_particles = new_proposed_particles
            _log_proposal_fn, _ = self.build_mh_proposal(new_mh_proposal_parameter, particles, log_weights,
                                                         self.log_tgt_fn(tempering_sequence.at[i + 1].get()), i + 1)
            log_proposal_fn = jnp.vectorize(_log_proposal_fn, signature="(d),(d)->()")
            log_proposal_from_particles_to_proposed_particles = log_proposal_fn(_new_particles,
                                                                                _new_proposed_particles)
            new_important_sampling_log_weights_from_proposal_to_new_proposal = log_proposal_from_particles_to_proposed_particles - new_log_q
            important_sampling_log_weights_from_proposal_to_new_proposal = \
                important_sampling_log_weights_from_proposal_to_new_proposal.at[i - 1].set(
                    new_important_sampling_log_weights_from_proposal_to_new_proposal)

            mh_proposal_parameters = mh_proposal_parameters.at[i + 1].set(new_mh_proposal_parameter)
            return particles, log_weights, mh_proposal_parameters, acceptance_bools, important_sampling_log_weights_from_proposal_to_new_proposal, criteria, tempering_sequence, diff_tempering_sequence, log_normalizations

        acceptance_bools = jnp.zeros((iteration, num_parallel_chain, num_mcmc_steps), dtype=int)
        important_sampling_log_weights_from_proposal_to_new_proposal = jnp.zeros(
            (iteration, num_parallel_chain, num_mcmc_steps))
        particles, log_weights, mh_proposal_parameters, acceptance_bools, important_sampling_log_weights_from_proposal_to_new_proposal, criteria, tempering_sequence, diff_tempering_sequence, log_normalizations = jax.lax.fori_loop(
            1, iteration + 1,
            body_fn, (
                particles,
                log_weights,
                mh_proposal_parameters,
                acceptance_bools,
                important_sampling_log_weights_from_proposal_to_new_proposal,
                criteria,
                tempering_sequence,
                diff_tempering_sequence,
                log_normalizations
            ))
        return particles, log_weights, mh_proposal_parameters, acceptance_bools, important_sampling_log_weights_from_proposal_to_new_proposal, criteria, tempering_sequence, log_normalizations
