import jax
import jax.numpy as jnp


def particle_initialisation_logexp(key, num_particles, dim):
    log_scale_init = jnp.log(jax.random.exponential(key, shape=(num_particles, dim)))
    init_particles = [log_scale_init]
    return init_particles


def particle_initialisation_normal(key, num_particles, dim):
    init_particles = [jax.random.normal(key, shape=(num_particles, dim))]
    return init_particles
