from adaptive_smc.problems.log_gaussian_cox import from_data_to_y, get_log_likelihood_fn, construct_target_and_prior

import pandas as pd
import jax
from adaptive_smc.laplace import laplace_approximation


def construct_invariant_measure_and_target(config):
    grid_size = config['problem'].get('grid_size', 16)
    base_measure_type = config['problem'].get('base_measure_type', 'prior')


    df_pines = pd.read_csv(open("./df_pines.csv", "r"))
    Y = from_data_to_y(df_pines, grid_size)

    loglikelihood_fn, log_target_density, prior = construct_target_and_prior(y=Y)
    log_prior_function, my_prior_mean, my_prior_covariance = prior

    base_measure_mean = my_prior_mean
    base_measure_cov = my_prior_covariance

    if base_measure_type == "laplace":  # i.e., Laplace
        my_tgt_log_density = lambda x: loglikelihood_fn(x) + jax.scipy.stats.multivariate_normal.logpdf(x,
                                                                                                        mean=my_prior_mean,
                                                                                                        cov=my_prior_covariance)
        # Perform Laplace approximation to find the base measure mean and covariance
        _, m, C = laplace_approximation(my_tgt_log_density, my_prior_mean)
        base_measure_mean = m
        base_measure_cov = C
        _loglikelihood_fn = lambda x: my_tgt_log_density(x) - jax.scipy.stats.multivariate_normal.logpdf(x,
                                                                                                         mean=base_measure_mean,
                                                                                                         cov=base_measure_cov)

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, mean=base_measure_mean,
                                              cov=base_measure_cov)

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=base_measure_mean,
                                                          cov=base_measure_cov)

    _loglikelihood_fn = loglikelihood_fn

    return _loglikelihood_fn, base_measure_sampler, logbase_density_fn, base_measure_mean, base_measure_cov
