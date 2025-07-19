import os
import pickle

import jax
import jax.lax
import jax.numpy as jnp


def save(res, config, output_path=""):
    """
    Saving in a PKL file the config dictionnary and the output of the SMC sampler.
    In addition, a plot of the means with error bars for each tempered distribution is saved in a PNG file.
    """
    # Extract directory from output_path
    directory = os.path.dirname(output_path)

    # Create directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_path, 'wb') as handle:
        pickle.dump({'config': config, 'res': res}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def acf(samples, max_order=20, diag=True):
    """
    Agarwal, M., & Vats, D. (2022). Globally Centered Autocovariances in MCMC.
    Journal of Computational and Graphical Statistics, 31(3), 629–638.
    https://doi.org/10.1080/10618600.2022.2037433

    Make ACF function for different iterations given set of samples (either particles or
    function evaluated particles) of shape (n_parallel_run, n_iterations, n_chain, n_length_of_chain, dim).
    It transforms the samples into a shape (n_iterations, n_parallel_run*n_chain, n_length_of_chain, dim)
    And compute the AC for order 1 to max_order, using the global mean as the stationary mean of the chain,
    and averaging over n_parallel_run*n_chain chains the computed auto-correlations.

    We iteratively set to 0 the entries of the chains at lag :k and -k:
    """
    n_iter = samples.shape[1]
    dim = samples.shape[-1]
    n_length_of_chain = samples.shape[3]

    samples = jnp.swapaxes(samples, 0, 1)
    samples = samples.reshape((samples.shape[0], -1, *samples.shape[3:]))

    global_mean = samples.mean(axis=[1, 2])
    global_mean = global_mean.reshape((samples.shape[0], 1, 1, samples.shape[3]))

    diff0 = samples - global_mean

    var0 = jnp.mean(diff0[..., jnp.newaxis] @ (diff0[..., jnp.newaxis, :]), axis=[2])

    def _acf(inps):
        k, samples_up_to_k, samples_from_k, carry = inps

        diff = samples_up_to_k
        diffk = samples_from_k

        diff = diff[..., jnp.newaxis, :]
        diffk = diffk[..., jnp.newaxis]

        prod = diffk @ (jnp.roll(diff, shift=k, axis=2))

        vark = jnp.mean(prod, axis=[2])  # biased
        corr = jnp.mean(vark / var0, axis=[-3])  # wrong

        carry = carry.at[k - 1].set(corr)
        return k + 1, samples_up_to_k.at[..., -k:, :].set(0), samples_from_k.at[..., :k, :].set(0), carry

    def cond(inps):
        k, *_ = inps
        return k < jax.lax.min(max_order + 1, n_length_of_chain)

    shifted_samples = samples - global_mean
    *_, acfs = jax.lax.while_loop(cond, _acf, (
        1, shifted_samples.at[..., -1:, :].set(0), shifted_samples.at[..., :1, :].set(0),
        jnp.zeros(shape=(max_order, n_iter, dim, dim))))

    if diag:
        acfs = acfs[..., jnp.arange(dim), jnp.arange(dim)]

    return acfs


def acf2(samples, max_order=20):
    """
    NOT OPTIMISED
    YIELDS EXACT SAME RESULT AS acf

    Agarwal, M., & Vats, D. (2022). Globally Centered Autocovariances in MCMC. Journal of Computational and Graphical Statistics, 31(3), 629–638. https://doi.org/10.1080/10618600.2022.2037433

    Make ACF function for different iterations given set of samples (either particles or
    function evaluated particles) of shape (n_parallel_run, n_iterations, n_chain, n_length_of_chain, dim).
    It transforms the samples into a shape (n_iterations, n_parallel_run*n_chain, n_length_of_chain, dim)
    And compute the AC for order 1 to max_order,
    using the global mean as the stationary mean of the chain, and
    averaging over n_parallel_run*n_chain chains the computed autocorrelations.
    """
    n_length_of_chain = samples.shape[3]

    samples = jnp.swapaxes(samples, 0, 1)
    samples = samples.reshape((samples.shape[0], -1, *samples.shape[3:]))

    global_mean = samples.mean(axis=[1, 2])
    global_mean = global_mean.reshape((samples.shape[0], 1, 1, samples.shape[3]))

    diff0 = samples - global_mean

    var0 = jnp.mean(diff0[..., jnp.newaxis] @ (diff0[..., jnp.newaxis, :]), axis=[2])

    def fcorr(k):
        diff = samples[:, :, :-k] - global_mean
        diffk = samples[:, :, k:] - global_mean
        prod = diff[..., jnp.newaxis] @ diffk[..., jnp.newaxis, :]
        vark = jnp.sum(prod, axis=[2]) / n_length_of_chain
        corr = jnp.mean(vark / var0, axis=[-3])
        return corr

    acf_result = jnp.array([fcorr(k) for k in range(1, max_order + 1)])
    return acf_result


def correct_acf(samples, max_order=20):
    """
    Agarwal, M., & Vats, D. (2022). Globally Centered Autocovariances in MCMC. Journal of Computational and Graphical Statistics, 31(3), 629–638. https://doi.org/10.1080/10618600.2022.2037433

    Make ACF function for different iterations given set of samples (either particles or
    function evaluated particles) of shape (n_parallel_run, n_iterations, n_chain, n_length_of_chain, dim).
    It transforms the samples into a shape (n_iterations, n_parallel_run*n_chain, n_length_of_chain, dim)
    And compute the AC for order 1 to max_order,
    using the global mean as the stationary mean of the chain, and
    averaging over n_parallel_run*n_chain chains the computed autocorrelations.
    """
    samples = jnp.swapaxes(samples, 0, 1)
    samples = samples.reshape(
        (samples.shape[0], samples.shape[1] * samples.shape[2], samples.shape[3], samples.shape[4]))
    global_mean = samples.mean(axis=[1, 2])
    global_mean = global_mean.reshape((samples.shape[0], 1, 1, samples.shape[3]))
    diff0 = samples - global_mean
    diff0 = diff0.reshape((diff0.shape[0], diff0.shape[1], diff0.shape[2], 1, diff0.shape[3]))
    var0 = jnp.mean(diff0 @ jnp.swapaxes(diff0, -1, -2), axis=[2])

    def fcorr(k):
        diff = samples[:, :, :-k] - global_mean
        diffk = samples[:, :, k:] - global_mean
        diff = diff.reshape((diff.shape[0], diff.shape[1], diff.shape[2], 1, diff.shape[3]))
        diffk = diffk.reshape((diffk.shape[0], diffk.shape[1], diffk.shape[2], diffk.shape[3], 1))
        prod = diff @ diffk
        vark = jnp.mean(prod, axis=[2])
        corr = jnp.mean(vark / var0, axis=[-3])
        return corr

    acf_result = jnp.array([fcorr(k) for k in range(1, max_order + 1)])
    return acf_result


if __name__ == "__main__":
    jax.config.update("jax_disable_jit", True)
    U = jax.random.PRNGKey(0)
    samples = jax.random.uniform(U, shape=(5, 2, 10, 100, 4))

    acf_jax = acf(samples, 5)
    acf_np = acf2(samples, 5)

    acf_correct = correct_acf(samples, 5)
    pass
