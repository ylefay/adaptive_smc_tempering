from typing import Callable, Tuple

import jax.random
from blackjax.smc.resampling import multinomial
from jax import numpy as jnp
from jax.typing import ArrayLike

PRNGKey = jax.Array


class GenericAdaptiveWasteFreeTemperingSMC:
    """
    A class that implements an adaptive Tempering SMC algorithm with waste-free SMC using parametric random-walk Metropolis-Hastings proposals.
    """

    def __init__(self, logbase_density_fn: Callable[[ArrayLike], ArrayLike],
                 base_measure_sampler: Callable[[PRNGKey], ArrayLike],
                 log_likelihood_fn: Callable[[ArrayLike], ArrayLike],
                 build_RWMH_proposal: Callable[
                     [ArrayLike, ArrayLike, int], Tuple[Callable[[ArrayLike, ArrayLike], ArrayLike],
                     Callable[[PRNGKey, ArrayLike], ArrayLike]]]
                 ) -> None:
        self.logbase_density_fn = logbase_density_fn
        self.base_measure_sampler = base_measure_sampler
        self.log_likelihood_fn = log_likelihood_fn

        def log_weights_fn(dlmbda):
            def _log_weights_fn(x):
                return dlmbda * self.log_likelihood_fn(x)

            return _log_weights_fn

        self.log_weights_fn = log_weights_fn
        self.build_RWMH_proposal = build_RWMH_proposal

    def sample(self, key: PRNGKey, num_parallel_chain: int, num_mcmc_steps: int,
               initial_rwmh_proposal_parameter: ArrayLike, tempering_sequence: ArrayLike) -> Tuple[
        ArrayLike, ArrayLike, ArrayLike]:
        """

        Parameters
        ----------
        key: PRNGKey
            A JAX PRNGKey
        num_parallel_chain: int
            Number of particles
        num_mcmc_steps: int
            Number of MCMC steps
        initial_rwmh_proposal_parameter: ArrayLike
            Initial parameter for the first RWMH proposal
        Returns
        -------
        A JAX array of shape (T, num_parallel_chain, num_mcmc_steps + 1) containing the sampled particles, with T the number of iterations given by the length of logweights_fn
        """
        diff_tempering_sequence = jnp.diff(tempering_sequence)
        diff_tempering_sequence = jnp.insert(diff_tempering_sequence, 0, jnp.array([0.]))
        iteration = len(diff_tempering_sequence)

        def accept_reject_MH_step(key: PRNGKey, log_density_for_proposed, log_density_for_current,
                                  log_proposal_for_proposed, log_proposal_for_current) -> bool:
            logU = -jax.random.exponential(key)
            log_mh_ratio = jax.lax.min(0.,
                                       log_density_for_proposed - log_density_for_current + log_proposal_for_proposed - log_proposal_for_current)
            return logU <= log_mh_ratio

        num_particles = num_parallel_chain * (num_mcmc_steps + 1)

        subkey = jax.random.fold_in(key, -1)
        subkeys = jax.random.split(subkey, num_particles)
        init_particles = self.base_measure_sampler(subkeys)
        shape = init_particles.shape[1:]
        particles = jnp.zeros((iteration, num_parallel_chain, num_mcmc_steps + 1, *shape))
        particles = particles.at[0].set(init_particles.reshape((num_parallel_chain, num_mcmc_steps + 1, *shape)))

        rwmh_proposal_parameters = jnp.zeros((iteration, *initial_rwmh_proposal_parameter.shape))
        rwmh_proposal_parameters = rwmh_proposal_parameters.at[0].set(initial_rwmh_proposal_parameter)

        subkey = jax.random.fold_in(key, 0)
        subkeys = jax.random.split(subkey, num_particles)

        log_proposal, proposal_sampler = self.build_RWMH_proposal(initial_rwmh_proposal_parameter, particles, 0)

        init_proposed_particles = jax.vmap(proposal_sampler, in_axes=(0, 0))(subkeys, init_particles)
        proposed_particles = jnp.zeros((iteration - 1, num_parallel_chain, num_mcmc_steps, *shape))

        init_log_weights = jax.vmap(self.logbase_density_fn)(init_particles)
        init_log_weights = init_log_weights - jax.scipy.special.logsumexp(init_log_weights)
        init_log_weights = init_log_weights.reshape((num_parallel_chain, num_mcmc_steps + 1))

        log_weights = jnp.zeros((iteration, num_parallel_chain, num_mcmc_steps + 1))
        log_weights = log_weights.at[0].set(init_log_weights)

        log_g = jax.vmap(self.logbase_density_fn)(init_proposed_particles) - jax.vmap(self.logbase_density_fn)(
            init_particles)
        d = jnp.sum(jnp.square(init_proposed_particles - init_particles), axis=-1)

        new_rwmh_proposal_parameter = initial_rwmh_proposal_parameter  # need to implement a minization procedure
        rwmh_proposal_parameters = rwmh_proposal_parameters.at[1].set(new_rwmh_proposal_parameter)

        init_particles = init_particles.reshape((num_parallel_chain, num_mcmc_steps + 1, *shape))
        particles = particles.at[0].set(init_particles)

        def body_fn(i, carry):
            particles, log_weights, rwmh_proposal_parameters = carry
            subkey = jax.random.fold_in(key, i)
            ancestors = multinomial(subkey, jnp.exp(log_weights.at[i - 1].get().reshape(-1)), num_parallel_chain)
            resampled_particles = \
                particles.at[i - 1].get().reshape((num_parallel_chain * (num_mcmc_steps + 1), *shape)).at[
                    ancestors].get()

            @jax.vmap
            def inside_body_fn(key, particle):
                _proposed_particles = jnp.zeros((num_mcmc_steps + 1, *shape))
                _proposed_particles = _proposed_particles.at[0].set(particle)
                _particles = jnp.zeros((num_mcmc_steps + 1, *shape))
                _particles = _particles.at[0].set(particle)

                def inside_inside_body_fn(p, carry):
                    key, proposed_particles_across_p, particles_across_p, log_g, d, log_q = carry
                    particle = particles_across_p.at[p - 1].get()
                    subkey = jax.random.fold_in(key, p)
                    _, key = jax.random.split(key)
                    log_proposal, proposal_sampler = self.build_RWMH_proposal(rwmh_proposal_parameters.at[i].get(),
                                                                              particles, i)
                    new_proposed_particle = proposal_sampler(subkey, particle)  # to fix
                    proposed_particles_across_p = proposed_particles_across_p.at[p].set(new_proposed_particle)
                    new_log_g = self.log_weights_fn(diff_tempering_sequence.at[i].get())(
                        new_proposed_particle) - self.log_weights_fn(diff_tempering_sequence.at[i].get())(particle)
                    new_d = jnp.sum(jnp.square(particle - new_proposed_particle), axis=-1)
                    log_g = log_g.at[p - 1].set(new_log_g)
                    d = d.at[p - 1].set(new_d)
                    new_log_q = log_proposal(particle, new_proposed_particle)  # to fix
                    log_q = log_q.at[p - 1].set(new_log_q)
                    log_current_tgt_density_particle = self.log_weights_fn(tempering_sequence.at[i].get())(particle)
                    log_current_tgt_density_new_proposed_particle = self.log_weights_fn(tempering_sequence.at[i].get())(
                        new_proposed_particle)

                    accept_MH_boolean = accept_reject_MH_step(key,
                                                              log_current_tgt_density_new_proposed_particle,
                                                              log_current_tgt_density_particle,
                                                              log_proposal(new_proposed_particle, particle),
                                                              new_log_q
                                                              )
                    new_particle = jax.lax.select(accept_MH_boolean, new_proposed_particle, particle)

                    particles_across_p = particles_across_p.at[p].set(new_particle)
                    proposed_particles_across_p = proposed_particles_across_p.at[p].set(new_proposed_particle)
                    return key, proposed_particles_across_p, particles_across_p, log_g, d, log_q

                _, _proposed_particles, _particles, log_g, d, log_q = jax.lax.fori_loop(1, num_mcmc_steps + 1,
                                                                                        inside_inside_body_fn,
                                                                                        (
                                                                                            key, _proposed_particles,
                                                                                            _particles,
                                                                                            jnp.zeros(num_mcmc_steps),
                                                                                            jnp.zeros(num_mcmc_steps),
                                                                                            jnp.zeros(num_mcmc_steps)))
                return _particles, _proposed_particles, log_g, d, log_q

            keys = jax.random.split(subkey, num_parallel_chain)
            new_particles, new_proposed_particles, new_log_g, new_d, new_log_q = inside_body_fn(keys,
                                                                                                resampled_particles)

            particles = particles.at[i].set(new_particles)
            new_log_weights = jax.vmap(self.log_weights_fn(diff_tempering_sequence.at[i].get()))(
                new_particles.reshape(num_particles, *shape)).reshape(
                (num_parallel_chain, num_mcmc_steps + 1))  # to fix
            new_log_weights = new_log_weights - jax.scipy.special.logsumexp(new_log_weights)
            log_weights = log_weights.at[i].set(new_log_weights)
            new_weights = jnp.exp(new_log_weights)
            new_g = jnp.exp(new_log_g)

            @jax.jit
            def m_estimate_of_esjd(param):
                _new_particles = new_particles.at[:, 1:, :].get().reshape((num_particles - num_parallel_chain, *shape))
                _new_proposed_particles = new_proposed_particles.at[:, 1:, :].get().reshape(
                    (num_particles - num_parallel_chain, *shape))
                _log_proposal_fn, _ = self.build_RWMH_proposal(param, particles, i)
                log_proposal_fn = jax.vmap(_log_proposal_fn)  # to fix
                one_term_log_proposal = log_proposal_fn(
                    _new_particles, _new_proposed_particles
                )
                diff_log_proposal = log_proposal_fn(
                    _new_proposed_particles, _new_particles
                ) - one_term_log_proposal
                ratio_proposal = jnp.exp(diff_log_proposal).reshape((num_parallel_chain, num_mcmc_steps))
                to_sum = (new_weights.at[:, 1:].get() * new_d * jax.lax.max(1., new_g * ratio_proposal)).reshape(
                    -1) * jnp.exp(
                    one_term_log_proposal - new_log_q.reshape(-1))
                return -jnp.sum(to_sum)

            # new_rwmh_proposal_parameter = opt.minimize(m_estimate_of_esjd, rwmh_proposal_parameters.at[i].get(), method="BFGS").x

            new_rwmh_proposal_parameter = initial_rwmh_proposal_parameter  # need to implement a minimization procedure
            rwmh_proposal_parameters = rwmh_proposal_parameters.at[i + 1].set(new_rwmh_proposal_parameter)

            return particles, log_weights, rwmh_proposal_parameters

        particles, log_weights, rwmh_proposal_parameters = jax.lax.fori_loop(1, iteration, body_fn, (
            particles, log_weights, rwmh_proposal_parameters))
        return particles, log_weights, rwmh_proposal_parameters
