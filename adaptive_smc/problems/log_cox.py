import jax
import jax.numpy as jnp


def get_log_likelihood_fn(y, m):
    """
    See Section 4.5 from Adaptive tuning of hamiltonian monte carlo within sequential monte carlo
    """
    y = y.reshape(-1, 1)  # Ensure y is a column vector

    def log_likelihood_fn(x):
        return - jnp.sum(y * x - m * jnp.exp(x))

    return log_likelihood_fn


def make_buccholz_chopin_log_likelihood_fn(y):
    """
    See Section 4.5 from Adaptive tuning of hamiltonian monte carlo within sequential monte carlo
    by Buchholz and Chopin (2021).
    """
    d = y.shape[0] if y.ndims > 1 else int(jnp.sqrt(y.shape[0]))
    m = 1 / d ** 2

    def delta_fn(j, jp, k, kp):
        return jnp.sqrt((j - jp) ** 2 + (k - kp) ** 2)

    indices = jnp.arange(1, d + 1)
    j = indices[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    jp = indices[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
    k = indices[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
    kp = indices[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

    delta = delta_fn(j, jp, k, kp).reshape(
        (d ** 2, d ** 2))
    Sigma = sigmasq * jnp.exp(-delta / (d * beta))
    return get_log_likelihood_fn(y, m)


def construct_my_prior_and_target(y):
    """"We hardcode the config.
    """
    sigmasq = 1.91
    beta = 1 / 33
    d = y.shape[0] if y.ndims > 1 else int(jnp.sqrt(y.shape[0]))
    mu = (jnp.log(126) - sigmasq / 2) * jnp.ones(d)
    m = 1 / d ** 2

    def delta_fn(j, jp, k, kp):
        return jnp.sqrt((j - jp) ** 2 + (k - kp) ** 2)

    indices = jnp.arange(1, d + 1)
    j = indices[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    jp = indices[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
    k = indices[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
    kp = indices[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

    delta = delta_fn(j, jp, k, kp).reshape(
        (d ** 2, d ** 2))
    Sigma = sigmasq * jnp.exp(-delta / (d * beta))
    precision = jnp.linalg.inv(Sigma)

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, mu, Sigma)

    def logbase_density_fn(x):
        return -0.5 * (x - mu).T @ precision @ (x - mu)

    loglikelihood_fn = get_log_likelihood_fn(y, m)

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn


if __name__ == "__main__":
    import pandas as pd

    df_pines = pd.read_csv(open("./datasets/df_pines.csv", "r"))
