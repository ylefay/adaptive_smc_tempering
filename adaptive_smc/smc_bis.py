from typing import Tuple, Optional

import jax.random
from blackjax.smc.resampling import multinomial
from blackjax.smc.solver import dichotomy
from jax import numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc.criteria_functions import square_distance
from adaptive_smc.metropolis import accept_reject_mh_step
from adaptive_smc.optimise import OptimisingProcedure, make_constant
from adaptive_smc.smc_types import CriteriaFunction, SMCStatebis, Sampler, LogDensity, PRNGKey, ProposalBuilder
from adaptive_smc.utils import log_ess, normalize_log_weights


class GenericAdaptiveWasteFreeTemperingSMCInefficient:
    """
        A class that implements an adaptive Tempering SMC algorithm with waste-free SMC using parametric Metropolis-Hastings kernels.
    """

    def __init__(self, logbase_density_fn: LogDensity,
                 base_measure_sampler: Sampler,
                 log_likelihood_fn: LogDensity,
                 build_mh_proposal: ProposalBuilder,
                 optimisation: OptimisingProcedure = make_constant(),
                 criteria_function: CriteriaFunction = square_distance
                 ) -> None:
        self.logbase_density_fn = logbase_density_fn
        self.vmapped_logbase_density_fn = jnp.vectorize(logbase_density_fn, signature='(d)->()')
        self.base_measure_sampler = base_measure_sampler
        self.vmapped_base_measure_sampler = jnp.vectorize(base_measure_sampler,
                                                          signature='(2)->(d)')  # Assuming a vmappable function
        self.log_likelihood_fn = log_likelihood_fn
        self.vmapped_log_likelihood_fn = jnp.vectorize(log_likelihood_fn,
                                                       signature='(d)->()')  # Assuming a vmappable function
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

    def vmapped_log_tgt_fn(self, lmbda):
        return jnp.vectorize(self.log_tgt_fn(lmbda), signature='(d)->()')

    def vmapped_log_weights_fn(self, dlmbda):
        return jnp.vectorize(self.log_weights_fn(dlmbda), signature='(d)->()')

    def estimate_expectation_criteria_fun(self, state, i):
        """
        Construct estimate E_{i + 1}[g(X, Y)]
        """
        log_weights = state.log_weights.at[i].get()
        particles = state.particles.at[i].get()
        proposed_particles = state.proposed_particles.at[i].get()
        g = self.criteria_function(particles, proposed_particles, state, i)

        def log_my_current_proposal(x, y):
            def _zero_proposal(x, y):
                return self.logbase_density_fn(y)

            def _log_my_current_proposal(x, y):
                return self.build_mh_proposal(
                    state,
                    self.log_tgt_fn(state.tempering_sequence.at[i - 1].get()),
                    self.log_likelihood_fn,
                    i)[0](x, y)

            return jax.lax.select(i > 0, _log_my_current_proposal(x, y), _zero_proposal(x, y))

        log_weights_current_proposal = jnp.vectorize(log_my_current_proposal, signature="(d),(d)->()")(particles,
                                                                                                       proposed_particles)

        def fun(param):
            _state = SMCStatebis(
                state.particles,
                state.proposed_particles,
                state.log_weights,
                state.mh_proposal_parameters.at[i].set(param),
                state.tempering_sequence,
                state.others
            )

            log_my_new_proposal, _, _ = self.build_mh_proposal(
                _state,
                self.log_tgt_fn(state.tempering_sequence.at[i].get()),
                self.log_likelihood_fn,
                i + 1
            )

            log_target_density_at_t_fn = self.vmapped_log_tgt_fn(state.tempering_sequence.at[i].get())
            log_target_density_at_t_fn_particles = log_target_density_at_t_fn(particles)
            log_current_tgt_density_new_proposed_particles = log_target_density_at_t_fn(
                proposed_particles)

            vmapped_log_my_new_proposal = jnp.vectorize(log_my_new_proposal, signature="(d),(d)->()")

            log_weights_proposal = vmapped_log_my_new_proposal(particles, proposed_particles)
            log_weights_proposal_inv = vmapped_log_my_new_proposal(proposed_particles, particles)

            _weights, _ = normalize_log_weights(log_weights + log_weights_proposal - log_weights_current_proposal)
            _weights = jnp.exp(_weights)

            log_acceptance_ratio = jax.lax.min(0.,
                                               log_current_tgt_density_new_proposed_particles - log_target_density_at_t_fn_particles + log_weights_proposal_inv - log_weights_proposal)
            acceptance_ratio = jnp.exp(log_acceptance_ratio)

            return jnp.sum(_weights * g * acceptance_ratio)

        return fun

    def sample(self, key: PRNGKey, num_parallel_chain: int, num_mcmc_steps: int,
               initial_mh_proposal_parameter: ArrayLike,
               tempering_sequence: ArrayLike,
               target_ess: Optional[float] = None,
               init_other: Optional[ArrayLike] = jnp.empty(1)) -> Tuple[
        ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        r"""

        Parameters
        ----------
        self
        key
        num_parallel_chain
        num_mcmc_steps
        initial_mh_proposal_parameter
        tempering_sequence
        target_ess
        init_other

        Returns
        -------

        """
        iteration = len(tempering_sequence) - 1
        diff_tempering_sequence = jnp.diff(tempering_sequence)
        diff_tempering_sequence = jnp.insert(diff_tempering_sequence, 0, tempering_sequence.at[0].get())

        GRID_CRITERIA = jnp.linspace(0.01, 8, 100)
        criteria = jnp.zeros((iteration + 1, GRID_CRITERIA.shape[-1]))

        P = num_mcmc_steps + 1
        num_particles = num_parallel_chain * P

        subkey = jax.random.fold_in(key, 0)
        subkeys = jax.random.split(subkey, (2 * num_parallel_chain, 2 * P))

        pair_init_particles = self.vmapped_base_measure_sampler(subkeys)
        dim = pair_init_particles.shape[-1]

        couple_particles = jnp.zeros((iteration + 1, 2, num_parallel_chain, P, dim))

        init_particles = pair_init_particles.at[:num_parallel_chain, :P].get()
        init_proposed_particles = pair_init_particles.at[num_parallel_chain:, P:].get()

        couple_particles = couple_particles.at[0, 0].set(init_particles)
        couple_particles = couple_particles.at[0, 1].set(init_proposed_particles)

        mh_proposal_parameters = jnp.zeros((iteration + 1, *initial_mh_proposal_parameter.shape))

        log_normalizations = jnp.zeros((iteration + 1,))
        log_weights = jnp.zeros((iteration + 1, num_parallel_chain, P))

        if target_ess:
            _log_weights = self.vmapped_log_likelihood_fn(init_particles)
            dlmbda = dichotomy(lambda dlmbda: log_ess(dlmbda, _log_weights) - jnp.log(target_ess), 0., 1.0, 1e-2, 10)
            dlmbda = jnp.clip(dlmbda, 0., 1.0)
            tempering_sequence = tempering_sequence.at[0].set(dlmbda)
            diff_tempering_sequence = diff_tempering_sequence.at[0].set(dlmbda)

        log_G0_fn = self.vmapped_log_weights_fn(diff_tempering_sequence.at[0].get())
        init_log_weights = log_G0_fn(init_particles)
        init_log_weights, log_normalization = normalize_log_weights(init_log_weights)

        log_normalizations = log_normalizations.at[0].set(log_normalization)
        log_weights = log_weights.at[0].set(init_log_weights)

        log_weights_couple = jnp.zeros(shape=log_weights.shape)

        others = jnp.zeros((iteration, *init_other.shape))
        others = others.at[0].set(init_other)

        state = SMCStatebis(
            couple_particles.at[:, 0].get(),
            couple_particles.at[:, 1].get(),
            log_weights,
            mh_proposal_parameters, tempering_sequence,
            others
        )

        to_optimize = self.estimate_expectation_criteria_fun(state, 0)
        criteria = criteria.at[0].set(jax.vmap(to_optimize)(GRID_CRITERIA))

        new_mh_proposal_parameter = self.optimisation(
            to_optimize,
            initial_mh_proposal_parameter
        )
        mh_proposal_parameters = mh_proposal_parameters.at[0].set(new_mh_proposal_parameter)

        new_state = SMCStatebis(
            couple_particles.at[:, 0].get(),
            couple_particles.at[:, 1].get(),
            log_weights,
            mh_proposal_parameters, tempering_sequence,
            others
        )

        _log_proposal, _, _ = (
            self.build_mh_proposal(new_state,
                                   self.log_tgt_fn(tempering_sequence.at[0].get()),
                                   self.log_likelihood_fn,
                                   1))
        log_proposal = jnp.vectorize(_log_proposal, signature="(d),(d)->()")

        log_weights_proposal = log_proposal(init_particles, init_proposed_particles)
        new_log_weights_couple = log_weights_proposal + log_weights.at[0].get()
        new_log_weights_couple, _ = normalize_log_weights(new_log_weights_couple)
        log_weights_couple = log_weights_couple.at[0].set(new_log_weights_couple)

        def make_inner_loop(i: int, couple_particles: ArrayLike,
                            log_weights: ArrayLike,
                            mh_proposal_parameters: ArrayLike,
                            tempering_sequence: ArrayLike,
                            others: ArrayLike):
            log_target_density_at_t_minus_one_fn = self.log_tgt_fn(tempering_sequence.at[i - 1].get())
            state = SMCStatebis(couple_particles.at[:, 0].get(), couple_particles.at[:, 1].get(),
                                log_weights, mh_proposal_parameters,
                                tempering_sequence,
                                others)
            log_proposal, proposal_sampler, _ = (
                self.build_mh_proposal(state,
                                       self.log_tgt_fn(tempering_sequence.at[i - 1].get()),
                                       self.log_likelihood_fn,
                                       i))

            @jax.vmap
            def inside_body_fn(key, couple_particle):
                _couple_particles = jnp.zeros((P, 2, dim))
                _couple_particles = _couple_particles.at[0].set(couple_particle)
                _acceptance_bools = jnp.zeros((num_mcmc_steps,), dtype=bool)

                def insinde_inside_body_fn(p, carry):
                    key, couple_particles_across_p, acceptance_bool_across_p = carry
                    couple_particle = couple_particles_across_p.at[p - 1].get()

                    particle = couple_particle.at[0].get()
                    proposed_particle = couple_particle.at[1].get()

                    subkey_p = jax.random.fold_in(key, p)
                    _, key = jax.random.split(key)

                    log_target_density_at_t_minus_one_fn_particle = log_target_density_at_t_minus_one_fn(particle)
                    log_current_tgt_density_new_proposed_particle = log_target_density_at_t_minus_one_fn(
                        proposed_particle)

                    accept_MH_boolean, _ = accept_reject_mh_step(
                        key,
                        log_current_tgt_density_new_proposed_particle,
                        log_target_density_at_t_minus_one_fn_particle,
                        log_proposal(proposed_particle, particle),
                        log_proposal(particle, proposed_particle)
                    )
                    acceptance_bool_across_p = acceptance_bool_across_p.at[p - 1].set(accept_MH_boolean)
                    new_particle = jax.lax.select(accept_MH_boolean, proposed_particle, particle)
                    new_proposed_particle = proposal_sampler(subkey_p, new_particle)
                    couple_particles_across_p = couple_particles_across_p.at[p, 0].set(new_particle)
                    couple_particles_across_p = couple_particles_across_p.at[p, 1].set(new_proposed_particle)
                    return key, couple_particles_across_p, acceptance_bool_across_p

                _, _couple_particles, _acceptance_bools = jax.lax.fori_loop(
                    1,
                    P,
                    insinde_inside_body_fn,
                    (
                        key,
                        _couple_particles,
                        _acceptance_bools
                    )
                )
                return _couple_particles, _acceptance_bools

            return inside_body_fn

        def body_fn(i, carry):
            couple_particles, log_weights_couple, log_weights, mh_proposal_parameters, acceptance_bools, criteria, tempering_sequence, diff_tempering_sequence, log_normalizations, others = carry
            subkey = jax.random.fold_in(key, i)
            ancestors = multinomial(subkey, jnp.exp(log_weights_couple.at[i - 1].get().reshape(-1)), num_parallel_chain)
            resampled_couple_particles = couple_particles.at[i - 1].get().reshape((num_particles, 2, dim)).at[
                ancestors].get()
            particles = couple_particles.at[:, 0].get()
            proposed_particles = couple_particles.at[:, 1].get()
            if target_ess:
                _log_weights = self.vmapped_log_likelihood_fn(
                    particles.at[i - 1].get())  # do not use new_particles, this is wrong
                dlmbda = dichotomy(lambda dlmbda: log_ess(dlmbda, _log_weights) - jnp.log(target_ess), 0.,
                                   1.0 - tempering_sequence.at[i - 1].get(), 1e-2, 10)
                dlmbda = jnp.clip(dlmbda, 0., 1.0 - tempering_sequence.at[i - 1].get())
                tempering_sequence = tempering_sequence.at[i].set(tempering_sequence.at[i - 1].get() + dlmbda)
                diff_tempering_sequence = diff_tempering_sequence.at[i].set(dlmbda)

            inside_body_fn = make_inner_loop(i,
                                             couple_particles,
                                             log_weights,
                                             mh_proposal_parameters,
                                             tempering_sequence,
                                             others)
            # Current SMC state before iteration i.
            state = SMCStatebis(
                particles,
                proposed_particles,
                log_weights,
                mh_proposal_parameters,
                tempering_sequence, others
            )

            _, _, other = self.build_mh_proposal(state,
                                                 self.log_tgt_fn(tempering_sequence.at[i - 1].get()),
                                                 self.log_likelihood_fn,
                                                 i)
            others = others.at[i].set(other)

            keys = jax.random.split(subkey, num_parallel_chain)
            # Running the inner loop for iteration t
            new_couple_particles, new_acceptance_bools = inside_body_fn(keys, resampled_couple_particles)
            acceptance_bools = acceptance_bools.at[i - 1].set(new_acceptance_bools)
            new_particles = new_couple_particles.at[..., 0, :].get()
            new_proposed_particles = new_couple_particles.at[..., 1, :].get()
            couple_particles = couple_particles.at[i, 0].set(new_particles)
            couple_particles = couple_particles.at[i, 1].set(new_proposed_particles)
            log_Gi_fn = self.vmapped_log_weights_fn(diff_tempering_sequence.at[i].get())
            new_log_weights = log_Gi_fn(new_particles)
            new_log_weights, log_normalization = normalize_log_weights(new_log_weights)
            log_normalizations = log_normalizations.at[i].set(log_normalization)
            log_weights = log_weights.at[i].set(new_log_weights)

            new_state = SMCStatebis(
                couple_particles.at[:, 0].get(),
                couple_particles.at[:, 1].get(),
                log_weights,
                mh_proposal_parameters, tempering_sequence, others
            )

            to_optimize = self.estimate_expectation_criteria_fun(new_state, i)
            new_mh_proposal_parameter = self.optimisation(
                to_optimize,
                mh_proposal_parameters.at[i - 1].get()
            )
            criteria = criteria.at[i].set(jax.vmap(to_optimize)(GRID_CRITERIA))
            mh_proposal_parameters = mh_proposal_parameters.at[i].set(new_mh_proposal_parameter)

            new_state = SMCStatebis(
                new_state.particles,
                new_state.proposed_particles,
                new_state.log_weights, mh_proposal_parameters, new_state.tempering_sequence, new_state.others
            )

            _log_my_new_proposal, _, _ = self.build_mh_proposal(
                new_state,
                self.log_tgt_fn(tempering_sequence.at[i].get()),
                self.log_likelihood_fn,
                i + 1
            )

            _log_proposal, _, _ = (
                self.build_mh_proposal(state,
                                       self.log_tgt_fn(tempering_sequence.at[i - 1].get()),
                                       self.log_likelihood_fn,
                                       i))

            log_proposal = jnp.vectorize(_log_proposal, signature="(d),(d)->()")
            log_my_new_proposal = jnp.vectorize(_log_my_new_proposal, signature="(d),(d)->()")

            new_log_weights_proposal = \
                log_my_new_proposal(new_particles, new_proposed_particles) - \
                log_proposal(new_particles, new_proposed_particles)
            new_log_weights_couple = new_log_weights_proposal + new_log_weights
            new_log_weights_couple, _ = normalize_log_weights(new_log_weights_couple)
            log_weights_couple = log_weights_couple.at[i].set(new_log_weights_couple)

            return couple_particles, log_weights_couple, log_weights, mh_proposal_parameters, acceptance_bools, criteria, tempering_sequence, diff_tempering_sequence, log_normalizations, others

        acceptance_bools = jnp.zeros((iteration, num_parallel_chain, num_mcmc_steps), dtype=bool)

        couple_particles, log_weights_couple, log_weights, mh_proposal_parameters, acceptance_bools, criteria, tempering_sequence, diff_tempering_sequence, log_normalizations, others = \
            jax.lax.fori_loop(1, iteration + 1,
                              body_fn,
                              (
                                  couple_particles,
                                  log_weights_couple,
                                  log_weights,
                                  mh_proposal_parameters,
                                  acceptance_bools,
                                  criteria,
                                  tempering_sequence,
                                  diff_tempering_sequence,
                                  log_normalizations,
                                  others
                              ))
        return couple_particles, log_weights_couple, log_weights, mh_proposal_parameters, acceptance_bools, criteria, tempering_sequence, diff_tempering_sequence, log_normalizations, others


class GenericAdaptiveWasteFreeTemperingSMC:
    """
        A class that implements an adaptive Tempering SMC algorithm with waste-free SMC using parametric Metropolis-Hastings kernels.
    """

    def __init__(self, logbase_density_fn: LogDensity,
                 base_measure_sampler: Sampler,
                 log_likelihood_fn: LogDensity,
                 build_mh_proposal: ProposalBuilder,
                 optimisation: OptimisingProcedure = make_constant(),
                 criteria_function: CriteriaFunction = square_distance
                 ) -> None:
        self.logbase_density_fn = logbase_density_fn
        self.vmapped_logbase_density_fn = jnp.vectorize(logbase_density_fn, signature='(d)->()')
        self.base_measure_sampler = base_measure_sampler
        self.vmapped_base_measure_sampler = jnp.vectorize(base_measure_sampler,
                                                          signature='(2)->(d)')  # Assuming a vmappable function
        self.log_likelihood_fn = log_likelihood_fn
        self.vmapped_log_likelihood_fn = jnp.vectorize(log_likelihood_fn,
                                                       signature='(d)->()')  # Assuming a vmappable function
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

    def vmapped_log_tgt_fn(self, lmbda):
        return jnp.vectorize(self.log_tgt_fn(lmbda), signature='(d)->()')

    def vmapped_log_weights_fn(self, dlmbda):
        return jnp.vectorize(self.log_weights_fn(dlmbda), signature='(d)->()')

    def estimate_expectation_criteria_fun(self, state, i):
        """
        Construct estimate E_{i + 1}[g(X, Y)]
        """
        log_weights = state.log_weights.at[i].get()
        particles = state.particles.at[i].get()
        proposed_particles = state.proposed_particles.at[i].get()
        g = self.criteria_function(particles, proposed_particles, state, i)

        def log_my_current_proposal(x, y):
            def _zero_proposal(x, y):
                return self.logbase_density_fn(y)

            def _log_my_current_proposal(x, y):
                return self.build_mh_proposal(
                    state,
                    self.log_tgt_fn(state.tempering_sequence.at[i - 1].get()),
                    self.log_likelihood_fn,
                    i)[0](x, y)

            return jax.lax.select(i > 0, _log_my_current_proposal(x, y), _zero_proposal(x, y))

        log_weights_current_proposal = jnp.vectorize(log_my_current_proposal, signature="(d),(d)->()")(particles,
                                                                                                       proposed_particles)

        def fun(param):
            _state = SMCStatebis(
                state.particles,
                state.proposed_particles,
                state.log_weights,
                state.mh_proposal_parameters.at[i].set(param),
                state.tempering_sequence,
                state.others
            )

            log_my_new_proposal, _, _ = self.build_mh_proposal(
                _state,
                self.log_tgt_fn(state.tempering_sequence.at[i].get()),
                self.log_likelihood_fn,
                i + 1
            )

            log_target_density_at_t_fn = self.vmapped_log_tgt_fn(state.tempering_sequence.at[i].get())
            log_target_density_at_t_fn_particles = log_target_density_at_t_fn(particles)
            log_current_tgt_density_new_proposed_particles = log_target_density_at_t_fn(
                proposed_particles)

            vmapped_log_my_new_proposal = jnp.vectorize(log_my_new_proposal, signature="(d),(d)->()")

            log_weights_proposal = vmapped_log_my_new_proposal(particles, proposed_particles)
            log_weights_proposal_inv = vmapped_log_my_new_proposal(proposed_particles, particles)

            _weights, _ = normalize_log_weights(log_weights + log_weights_proposal - log_weights_current_proposal)
            _weights = jnp.exp(_weights)

            log_acceptance_ratio = jax.lax.min(0.,
                                               log_current_tgt_density_new_proposed_particles - log_target_density_at_t_fn_particles + log_weights_proposal_inv - log_weights_proposal)
            acceptance_ratio = jnp.exp(log_acceptance_ratio)

            return jnp.sum(_weights * g * acceptance_ratio)

        return fun

    def sample(self, key: PRNGKey, num_parallel_chain: int, num_mcmc_steps: int,
               initial_mh_proposal_parameter: ArrayLike,
               tempering_sequence: ArrayLike,
               target_ess: Optional[float] = None,
               init_other: Optional[ArrayLike] = jnp.empty(1)) -> Tuple[
        ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        r"""

        Parameters
        ----------
        self
        key
        num_parallel_chain
        num_mcmc_steps
        initial_mh_proposal_parameter
        tempering_sequence
        target_ess
        init_other

        Returns
        -------

        """
        iteration = len(tempering_sequence) - 1
        diff_tempering_sequence = jnp.diff(tempering_sequence)
        diff_tempering_sequence = jnp.insert(diff_tempering_sequence, 0, tempering_sequence.at[0].get())

        GRID_CRITERIA = jnp.linspace(0.01, 8, 100)
        criteria = jnp.zeros((iteration + 1, GRID_CRITERIA.shape[-1]))

        P = num_mcmc_steps + 1
        num_particles = num_parallel_chain * P

        subkey = jax.random.fold_in(key, 0)
        subkeys = jax.random.split(subkey, (2 * num_parallel_chain, 2 * P))

        pair_init_particles = self.vmapped_base_measure_sampler(subkeys)
        dim = pair_init_particles.shape[-1]

        couple_particles = jnp.zeros((iteration + 1, 2, num_parallel_chain, P, dim))

        init_particles = pair_init_particles.at[:num_parallel_chain, :P].get()
        init_proposed_particles = pair_init_particles.at[num_parallel_chain:, P:].get()

        couple_particles = couple_particles.at[0, 0].set(init_particles)
        couple_particles = couple_particles.at[0, 1].set(init_proposed_particles)

        mh_proposal_parameters = jnp.zeros((iteration + 1, *initial_mh_proposal_parameter.shape))

        log_normalizations = jnp.zeros((iteration + 1,))
        log_weights = jnp.zeros((iteration + 1, num_parallel_chain, P))

        if target_ess:
            _log_weights = self.vmapped_log_likelihood_fn(init_particles)
            dlmbda = dichotomy(lambda dlmbda: log_ess(dlmbda, _log_weights) - jnp.log(target_ess), 0., 1.0, 1e-2, 10)
            dlmbda = jnp.clip(dlmbda, 0., 1.0)
            tempering_sequence = tempering_sequence.at[0].set(dlmbda)
            diff_tempering_sequence = diff_tempering_sequence.at[0].set(dlmbda)

        log_G0_fn = self.vmapped_log_weights_fn(diff_tempering_sequence.at[0].get())
        init_log_weights = log_G0_fn(init_particles)
        init_log_weights, log_normalization = normalize_log_weights(init_log_weights)

        log_normalizations = log_normalizations.at[0].set(log_normalization)
        log_weights = log_weights.at[0].set(init_log_weights)

        log_weights_couple = jnp.zeros(shape=log_weights.shape)

        others = jnp.zeros((iteration, *init_other.shape))
        others = others.at[0].set(init_other)

        state = SMCStatebis(
            couple_particles.at[:, 0].get(),
            couple_particles.at[:, 1].get(),
            log_weights,
            mh_proposal_parameters, tempering_sequence,
            others
        )

        to_optimize = self.estimate_expectation_criteria_fun(state, 0)
        criteria = criteria.at[0].set(jax.vmap(to_optimize)(GRID_CRITERIA))

        new_mh_proposal_parameter = self.optimisation(
            to_optimize,
            initial_mh_proposal_parameter
        )
        mh_proposal_parameters = mh_proposal_parameters.at[0].set(new_mh_proposal_parameter)

        new_state = SMCStatebis(
            couple_particles.at[:, 0].get(),
            couple_particles.at[:, 1].get(),
            log_weights,
            mh_proposal_parameters, tempering_sequence,
            others
        )

        _log_proposal, _, _ = (
            self.build_mh_proposal(new_state,
                                   self.log_tgt_fn(tempering_sequence.at[0].get()),
                                   self.log_likelihood_fn,
                                   1))
        log_proposal = jnp.vectorize(_log_proposal, signature="(d),(d)->()")

        log_weights_proposal = log_proposal(init_particles, init_proposed_particles)
        new_log_weights_couple = log_weights_proposal + log_weights.at[0].get()
        new_log_weights_couple, _ = normalize_log_weights(new_log_weights_couple)
        log_weights_couple = log_weights_couple.at[0].set(new_log_weights_couple)

        def make_inner_loop(i: int, couple_particles: ArrayLike,
                            log_weights: ArrayLike,
                            mh_proposal_parameters: ArrayLike,
                            tempering_sequence: ArrayLike,
                            others: ArrayLike):
            log_target_density_at_t_minus_one_fn = self.log_tgt_fn(tempering_sequence.at[i - 1].get())
            state = SMCStatebis(couple_particles.at[:, 0].get(), couple_particles.at[:, 1].get(),
                                log_weights, mh_proposal_parameters,
                                tempering_sequence,
                                others)
            log_proposal, proposal_sampler, _ = (
                self.build_mh_proposal(state,
                                       self.log_tgt_fn(tempering_sequence.at[i - 1].get()),
                                       self.log_likelihood_fn,
                                       i))

            @jax.vmap
            def inside_body_fn(key, couple_particle):
                # Resampling for the first iteration.
                subkey_0 = jax.random.fold_in(key, 0)
                first_proposal = proposal_sampler(subkey_0, couple_particle.at[0].get())
                couple_particle = couple_particle.at[1].set(first_proposal)

                _couple_particles = jnp.zeros((P, 2, dim))
                _couple_particles = _couple_particles.at[0].set(couple_particle)
                _acceptance_bools = jnp.zeros((num_mcmc_steps,), dtype=bool)

                def insinde_inside_body_fn(p, carry):
                    key, couple_particles_across_p, acceptance_bool_across_p = carry
                    couple_particle = couple_particles_across_p.at[p - 1].get()

                    particle = couple_particle.at[0].get()
                    proposed_particle = couple_particle.at[1].get()

                    subkey_p = jax.random.fold_in(key, p)
                    _, key = jax.random.split(key)

                    log_target_density_at_t_minus_one_fn_particle = log_target_density_at_t_minus_one_fn(particle)
                    log_current_tgt_density_new_proposed_particle = log_target_density_at_t_minus_one_fn(
                        proposed_particle)

                    accept_MH_boolean, _ = accept_reject_mh_step(
                        key,
                        log_current_tgt_density_new_proposed_particle,
                        log_target_density_at_t_minus_one_fn_particle,
                        log_proposal(proposed_particle, particle),
                        log_proposal(particle, proposed_particle)
                    )
                    acceptance_bool_across_p = acceptance_bool_across_p.at[p - 1].set(accept_MH_boolean)
                    new_particle = jax.lax.select(accept_MH_boolean, proposed_particle, particle)
                    new_proposed_particle = proposal_sampler(subkey_p, new_particle)
                    couple_particles_across_p = couple_particles_across_p.at[p, 0].set(new_particle)
                    couple_particles_across_p = couple_particles_across_p.at[p, 1].set(new_proposed_particle)
                    return key, couple_particles_across_p, acceptance_bool_across_p

                _, _couple_particles, _acceptance_bools = jax.lax.fori_loop(
                    1,
                    P,
                    insinde_inside_body_fn,
                    (
                        key,
                        _couple_particles,
                        _acceptance_bools
                    )
                )
                return _couple_particles, _acceptance_bools

            return inside_body_fn

        def body_fn(i, carry):
            couple_particles, log_weights_couple, log_weights, mh_proposal_parameters, acceptance_bools, criteria, tempering_sequence, diff_tempering_sequence, log_normalizations, others = carry
            subkey = jax.random.fold_in(key, i)
            ancestors = multinomial(subkey, jnp.exp(log_weights.at[i - 1].get().reshape(-1)), num_parallel_chain)
            resampled_couple_particles = couple_particles.at[i - 1].get().reshape((num_particles, 2, dim)).at[
                ancestors].get()
            particles = couple_particles.at[:, 0].get()
            proposed_particles = couple_particles.at[:, 1].get()
            if target_ess:
                _log_weights = self.vmapped_log_likelihood_fn(
                    particles.at[i - 1].get())  # do not use new_particles, this is wrong
                dlmbda = dichotomy(lambda dlmbda: log_ess(dlmbda, _log_weights) - jnp.log(target_ess), 0.,
                                   1.0 - tempering_sequence.at[i - 1].get(), 1e-2, 10)
                dlmbda = jnp.clip(dlmbda, 0., 1.0 - tempering_sequence.at[i - 1].get())
                tempering_sequence = tempering_sequence.at[i].set(tempering_sequence.at[i - 1].get() + dlmbda)
                diff_tempering_sequence = diff_tempering_sequence.at[i].set(dlmbda)

            inside_body_fn = make_inner_loop(i,
                                             couple_particles,
                                             log_weights,
                                             mh_proposal_parameters,
                                             tempering_sequence,
                                             others)
            # Current SMC state before iteration i.
            state = SMCStatebis(
                particles,
                proposed_particles,
                log_weights,
                mh_proposal_parameters,
                tempering_sequence, others
            )

            _, _, other = self.build_mh_proposal(state,
                                                 self.log_tgt_fn(tempering_sequence.at[i - 1].get()),
                                                 self.log_likelihood_fn,
                                                 i)
            others = others.at[i].set(other)

            keys = jax.random.split(subkey, num_parallel_chain)
            # Running the inner loop for iteration t
            new_couple_particles, new_acceptance_bools = inside_body_fn(keys, resampled_couple_particles)
            acceptance_bools = acceptance_bools.at[i - 1].set(new_acceptance_bools)
            new_particles = new_couple_particles.at[..., 0, :].get()
            new_proposed_particles = new_couple_particles.at[..., 1, :].get()
            couple_particles = couple_particles.at[i, 0].set(new_particles)
            couple_particles = couple_particles.at[i, 1].set(new_proposed_particles)
            log_Gi_fn = self.vmapped_log_weights_fn(diff_tempering_sequence.at[i].get())
            new_log_weights = log_Gi_fn(new_particles)
            new_log_weights, log_normalization = normalize_log_weights(new_log_weights)
            log_normalizations = log_normalizations.at[i].set(log_normalization)
            log_weights = log_weights.at[i].set(new_log_weights)

            new_state = SMCStatebis(
                couple_particles.at[:, 0].get(),
                couple_particles.at[:, 1].get(),
                log_weights,
                mh_proposal_parameters, tempering_sequence, others
            )

            to_optimize = self.estimate_expectation_criteria_fun(new_state, i)
            new_mh_proposal_parameter = self.optimisation(
                to_optimize,
                mh_proposal_parameters.at[i - 1].get()
            )
            criteria = criteria.at[i].set(jax.vmap(to_optimize)(GRID_CRITERIA))
            mh_proposal_parameters = mh_proposal_parameters.at[i].set(new_mh_proposal_parameter)

            new_state = SMCStatebis(
                new_state.particles,
                new_state.proposed_particles,
                new_state.log_weights, mh_proposal_parameters, new_state.tempering_sequence, new_state.others
            )

            _log_my_new_proposal, _, _ = self.build_mh_proposal(
                new_state,
                self.log_tgt_fn(tempering_sequence.at[i].get()),
                self.log_likelihood_fn,
                i + 1
            )

            _log_proposal, _, _ = (
                self.build_mh_proposal(state,
                                       self.log_tgt_fn(tempering_sequence.at[i - 1].get()),
                                       self.log_likelihood_fn,
                                       i))

            log_proposal = jnp.vectorize(_log_proposal, signature="(d),(d)->()")
            log_my_new_proposal = jnp.vectorize(_log_my_new_proposal, signature="(d),(d)->()")

            new_log_weights_proposal = \
                log_my_new_proposal(new_particles, new_proposed_particles) - \
                log_proposal(new_particles, new_proposed_particles)
            new_log_weights_couple = new_log_weights_proposal + new_log_weights
            new_log_weights_couple, _ = normalize_log_weights(new_log_weights_couple)
            log_weights_couple = log_weights_couple.at[i].set(new_log_weights_couple)

            return couple_particles, log_weights_couple, log_weights, mh_proposal_parameters, acceptance_bools, criteria, tempering_sequence, diff_tempering_sequence, log_normalizations, others

        acceptance_bools = jnp.zeros((iteration, num_parallel_chain, num_mcmc_steps), dtype=bool)

        couple_particles, log_weights_couple, log_weights, mh_proposal_parameters, acceptance_bools, criteria, tempering_sequence, diff_tempering_sequence, log_normalizations, others = \
            jax.lax.fori_loop(1, iteration + 1,
                              body_fn,
                              (
                                  couple_particles,
                                  log_weights_couple,
                                  log_weights,
                                  mh_proposal_parameters,
                                  acceptance_bools,
                                  criteria,
                                  tempering_sequence,
                                  diff_tempering_sequence,
                                  log_normalizations,
                                  others
                              ))
        return couple_particles, log_weights_couple, log_weights, mh_proposal_parameters, acceptance_bools, criteria, tempering_sequence, diff_tempering_sequence, log_normalizations, others
