import jax.numpy as jnp
import pandas as pd


def from_data_to_y(data: pd.DataFrame, grid_size: int) -> jnp.ndarray:
    """
    Code partially taken from A. Buchholz' repository:
    https://github.com/alexanderbuchholz/hsmc/blob/master/smc_sampler_functions/target_distributions_logcox.py
    Parameters
    ----------
    data: pd.DataFrame with two columns, x and y
    grid_size: size of the squared grid (i.e., N\times N cells)
    """
    grid = jnp.linspace(start=0, stop=1, num=grid_size + 1)

    data_x = jnp.asarray(data['data_x'])
    data_y = jnp.asarray(data['data_y'])

    logical_y = (data_x[:, None] > grid[:-1]) & (data_x[:, None] < grid[1:])  # shape: (n_data, N)

    logical_x = (data_y[:, None] > grid[:-1]) & (data_y[:, None] < grid[1:])

    data_counts = (logical_y[:, :, None] & logical_x[:, None, :]).sum(axis=0)
    data_counts = data_counts.ravel()  # size N*N

    return data_counts


def get_log_likelihood_fn(y, m):
    """
    See Section 4.5 from Adaptive tuning of hamiltonian monte carlo within sequential monte carlo
    """

    def log_likelihood_fn(x):
        return jnp.sum(y * x - m * jnp.exp(x))

    return log_likelihood_fn

def get_jac_log_likelihood_fn(y, m):
    def jac_log_likelihood_fn(x):
        return y - m * jnp.exp(x)
    return jac_log_likelihood_fn


def construct_target_and_prior(y):
    """
    The configuration from Section 4.5 of
    Adaptive tuning of hamiltonian monte carlo within sequential monte carlo,
    is hard-coded in this function.
    """
    sigmasq = 1.91
    beta = 1 / 33
    d = int(jnp.sqrt(y.shape[0]))
    mu = (jnp.log(126) - sigmasq / 2) * jnp.ones(d ** 2)
    m = 1 / d ** 2

    def delta_fn(j, jp, k, kp):
        return jnp.sqrt((j - jp) ** 2 + (k - kp) ** 2)

    indices = jnp.arange(1, d + 1)
    j = indices[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    jp = indices[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
    k = indices[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
    kp = indices[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

    delta = delta_fn(j, k, jp, kp).reshape(
        (d ** 2, d ** 2))
    Sigma = sigmasq * jnp.exp(-delta / (d * beta))

    precision = jnp.linalg.inv(Sigma)

    def log_prior(x):
        return -0.5 * (x - mu).T @ precision @ (x - mu)

    loglikelihood_fn = get_log_likelihood_fn(y, m)

    log_target_density = lambda x: loglikelihood_fn(x) + log_prior(x)
    return loglikelihood_fn, log_target_density, (log_prior, mu, Sigma)
