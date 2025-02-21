import pickle

import jax.numpy as jnp


def save(res, config, output_path=""):
    """
    Saving in a PKL file the config dictionnary and the output of the SMC sampler.
    In addition, a plot of the means with error bars for each tempered distribution is saved in a PNG file.
    """
    with open(output_path, 'wb') as handle:
        pickle.dump({'config': config, 'res': res}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def acf(samples, max_order=20):
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

    acf_result = jnp.array([fcorr(k) for k in range(1, max_order)])
    return acf_result
