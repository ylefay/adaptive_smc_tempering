import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax import tempered_smc
from blackjax.smc.resampling import multinomial
from adaptive_smc_tempering.src.experiments.logistic import get_dataset
from adaptive_smc_tempering.src.logistic import get_log_likelihood
from adaptive_smc_tempering.src.proposals import build_crank_nicholson_kernel


class TemperedSMC():
    """Test posterior mean estimate."""

    def __init__(self, logprior_fn, loglikelihood_fn, dim):
        self.logprior_fn = logprior_fn
        self.loglikelihood_fn = loglikelihood_fn
        self.dim = dim
        self.resample_fn = multinomial
        self.mcmc_step_fn = build_crank_nicholson_kernel(0.1, jnp.eye(dim))

    def fixed_schedule_tempered_smc(self, key, init_particles):
        logprior_fn = self.logprior_fn
        loglikelihood_fn = self.loglikelihood_fn
        resample_fn = self.resample_fn
        mcmc_step_fn = self.mcmc_step_fn

        def mcmc_init_fn(position, logdensity_fn):
            return blackjax.mcmc.random_walk.init(position=position, logdensity_fn=logdensity_fn)

        num_tempering_steps = 10

        lambda_schedule = np.logspace(-5, 0, num_tempering_steps)

        kernel = tempered_smc.build_kernel(
            logprior_fn,
            loglikelihood_fn,
            mcmc_step_fn,
            mcmc_init_fn,
            resample_fn,
        )
        init_state = tempered_smc.init(init_particles)

        def body_fn(carry, lmbda):
            i, state = carry
            subkey = jax.random.fold_in(key, i)
            new_state, info = kernel(subkey, state, lmbda=lmbda, num_mcmc_steps=10, mcmc_parameters={})
            return (i + 1, new_state), (new_state, info)

        (_, result), _ = jax.lax.scan(body_fn, (0, init_state), lambda_schedule)


if __name__ == "__main__":
    OP_key = jax.random.PRNGKey(0)
    flipped_predictors = get_dataset()
    N, dim = flipped_predictors.shape

    _loglikelihood_fn = get_log_likelihood(flipped_predictors)
    loglikelihood_fn = lambda x: _loglikelihood_fn(x[0])

    num_particles = 200
    log_scale_init = np.log(np.random.exponential(1, num_particles * dim)).reshape(num_particles, dim)
    coeffs_init = 3 + 2 * np.random.randn(num_particles)


    def logprior_fn(x):
        return jax.scipy.stats.norm.logpdf(x[0], loc=jnp.zeros(dim), scale=jnp.ones(dim))


    init_particles = [log_scale_init]
    my_smc = TemperedSMC(logprior_fn, loglikelihood_fn, dim)
    my_smc.fixed_schedule_tempered_smc(OP_key, init_particles)
