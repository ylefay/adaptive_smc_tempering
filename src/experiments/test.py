import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from blackjax import tempered_smc
from blackjax.smc.resampling import multinomial

from src.experiments.logistic import get_dataset
from src.logistic import get_log_likelihood
from src.proposals import build_crank_nicholson_kernel


class TemperedSMC():
    """Test posterior mean estimate."""

    def __init__(self, logprior_fn, loglikelihood_fn, dim):
        self.logprior_fn = logprior_fn
        self.loglikelihood_fn = loglikelihood_fn
        self.dim = dim
        self.resample_fn = multinomial
        self.mcmc_step_fn = build_crank_nicholson_kernel(0.1, jnp.eye(dim))

    def fixed_schedule_tempered_smc(self, key, init_particles, lmbda_schedule):
        def mcmc_init_fn(position, logdensity_fn):
            return blackjax.mcmc.random_walk.init(position=position, logdensity_fn=logdensity_fn)

        kernel = tempered_smc.build_kernel(
            self.logprior_fn,
            self.loglikelihood_fn,
            self.mcmc_step_fn,
            mcmc_init_fn,
            self.resample_fn,
        )
        init_state = tempered_smc.init(init_particles)

        def body_fn(carry, lmbda):
            i, state = carry
            subkey = jax.random.fold_in(key, i)
            new_state, info = kernel(subkey, state, lmbda=lmbda, num_mcmc_steps=10, mcmc_parameters={})
            return (i + 1, new_state), (new_state, info)

        (_, result), chain = jax.lax.scan(body_fn, (0, init_state), lmbda_schedule)
        return result, chain


if __name__ == "__main__":
    OP_key = jax.random.PRNGKey(0)
    keys = jax.random.split(OP_key, 5)
    flipped_predictors = get_dataset()
    N, dim = flipped_predictors.shape

    _loglikelihood_fn = get_log_likelihood(flipped_predictors)
    loglikelihood_fn = lambda x: _loglikelihood_fn(x[0])

    num_particles = 200
    log_scale_init = np.log(np.random.exponential(1, num_particles * dim)).reshape(num_particles, dim)
    coeffs_init = np.random.randn(num_particles)


    def logprior_fn(x):
        return jax.scipy.stats.norm.logpdf(x[0], loc=jnp.zeros(dim), scale=jnp.ones(dim)).sum()


    num_tempering_steps = 1500
    lmbda_schedule = np.logspace(-5, 0, num_tempering_steps)

    init_particles = [log_scale_init]
    my_smc = TemperedSMC(logprior_fn, loglikelihood_fn, dim)
    res, chain = jax.vmap(my_smc.fixed_schedule_tempered_smc, in_axes=(0, None, None))(keys, init_particles,
                                                                                       lmbda_schedule)

    print(res[0][0].mean(axis=1))
    print(res[0][0].mean(axis=1).mean(axis=0))
    plt.plot(res[0][0].mean(axis=1).mean(axis=0))
    plt.savefig(f"{num_tempering_steps}_{num_particles}.png")
    plt.clf()
    for idx in range(len(res[0][0])):
        plt.plot(res[0][0][idx].mean(axis=0))
    plt.savefig(f"{num_tempering_steps}_{num_particles}_{len(res[0][0])}.png")
    
