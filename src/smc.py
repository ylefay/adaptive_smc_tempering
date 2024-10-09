import jax
import jax.numpy


def resample_fn(resampling_key, weights, num_resampled):
    """
    github copilot generated code
    Resample particles according to their weights.

    Parameters
    ----------
    resampling_key
        Key used to generate pseudo-random numbers.
    weights
        Weights of the particles.
    num_resampled
        The number of particles to resample.

    Returns
    -------
    resampling_idx
        An array of indices of the particles to resample.
    """
    num_particles = weights.shape[0]
    cumulative_weights = jax.numpy.cumsum(weights)
    cumulative_weights /= cumulative_weights[-1]
    resampling_idx = jax.random.choice(
        resampling_key, num_particles, shape=(num_resampled,), p=cumulative_weights
    )
    return resampling_idx
