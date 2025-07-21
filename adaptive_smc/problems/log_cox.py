import jax.numpy as jnp


def get_log_likelihood_fn(y, m):
    """
    See Section 4.5 from Adaptive tuning of hamiltonian monte carlo within sequential monte carlo
    """
    y = y.reshape(-1, 1)  # Ensure y is a column vector

    def log_likelihood_fn(x):
        return - jnp.sum(y * x - m * jnp.exp(x))

    return log_likelihood_fn


def construct_target(y):
    """"We hardcode the config.
    Return the log likelihood and log target function
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

    def log_prior(x):
        return -0.5 * (x - mu).T @ precision @ (x - mu)

    loglikelihood_fn = get_log_likelihood_fn(y, m)

    log_target_density = lambda x: loglikelihood_fn(x) + log_prior(x)
    return loglikelihood_fn, log_target_density


if __name__ == "__main__":
    import pandas as pd

    df_pines = pd.read_csv(open("./datasets/df_pines.csv", "r"))
