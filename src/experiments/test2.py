"""Test the tempered SMC steps and routine"""
import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest

import blackjax
import blackjax.smc.resampling as resampling
import blackjax.smc.solver as solver
from blackjax import adaptive_tempered_smc, tempered_smc
from blackjax.smc import extend_params


def inference_loop(kernel, rng_key, initial_state):
    def cond(carry):
        _, state, *_ = carry
        return state.lmbda < 1

    def body(carry):
        i, state, curr_loglikelihood = carry
        subkey = jax.random.fold_in(rng_key, i)
        state, info = kernel(subkey, state)
        return i + 1, state, curr_loglikelihood + info.log_likelihood_increment

    total_iter, final_state, log_likelihood = jax.lax.while_loop(
        cond, body, (0, initial_state, 0.0)
    )

    return total_iter, final_state, log_likelihood


class TemperedSMCTest(SMCLinearRegressionTestCase):
    """Test posterior mean estimate."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_fixed_schedule_tempered_smc(self):
        (
            init_particles,
            logprior_fn,
            loglikelihood_fn,
        ) = self.particles_prior_loglikelihood()

        num_tempering_steps = 10

        lambda_schedule = np.logspace(-5, 0, num_tempering_steps)
        hmc_init = blackjax.hmc.init
        hmc_kernel = blackjax.hmc.build_kernel()
        hmc_parameters = extend_params(
            {
                "step_size": 10e-2,
                "inverse_mass_matrix": jnp.eye(2),
                "num_integration_steps": 50,
            },
        )

        tempering = tempered_smc(
            logprior_fn,
            loglikelihood_fn,
            hmc_kernel,
            hmc_init,
            hmc_parameters,
            resampling.systematic,
            10,
        )
        init_state = tempering.init(init_particles)
        smc_kernel = self.variant(tempering.step)

        def body_fn(carry, lmbda):
            i, state = carry
            subkey = jax.random.fold_in(self.key, i)
            new_state, info = smc_kernel(subkey, state, lmbda)
            return (i + 1, new_state), (new_state, info)

        (_, result), _ = jax.lax.scan(body_fn, (0, init_state), lambda_schedule)
        self.assert_linear_regression_test_case(result)


def normal_logdensity_fn(x, chol_cov):
    """minus log-density of a centered multivariate normal distribution"""
    dim = chol_cov.shape[0]
    y = jax.scipy.linalg.solve_triangular(chol_cov, x, lower=True)
    normalizing_constant = (
        np.sum(np.log(np.abs(np.diag(chol_cov)))) + dim * np.log(2 * np.pi) / 2.0
    )
    norm_y = jnp.sum(y * y, -1)
    return -(0.5 * norm_y + normalizing_constant)


if __name__ == "__main__":
    absltest.main()