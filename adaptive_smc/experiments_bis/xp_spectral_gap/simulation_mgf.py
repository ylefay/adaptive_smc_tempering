import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from math import floor

def make_sim(sampling, p):
    def sim(key, length):
        key_init, key = jax.random.split(key)
        init_state = sampling(key_init)

        def fun(current_state, input_key):
            key1, key2 = jax.random.split(input_key, 2)
            ind = jax.random.bernoulli(key1, p)
            new_state = jax.lax.cond(ind, lambda _: sampling(key2), lambda _: current_state, None)
            return new_state, new_state

        my_iter_keys = jax.random.split(key, length)
        _, all_states = jax.lax.scan(fun, init_state, my_iter_keys)
        return all_states

    return sim


def mgf(key, my_test_fun, n_samples, length, sampling, p):
    keys = jax.random.split(key, n_samples)
    my_simulator = make_sim(sampling, p)
    my_vmapped_simulator = jax.vmap(lambda key: my_simulator(key, length))

    my_simulations = my_vmapped_simulator(keys)
    evaluated_simulations = my_test_fun(my_simulations)
    cum_sums = jnp.cumsum(evaluated_simulations, axis=-1)

    def _mgf(u):
        return jnp.mean(jnp.exp(u[:, None, None] * cum_sums),
                        axis=1)

    return jax.jit(_mgf), my_simulations, evaluated_simulations


def sampling_gaussian(key):
    return jax.random.normal(key)


my_test_fun = lambda x: -jnp.abs(x)
my_u_axis = jnp.linspace(0, 1, 500)
OP_key = jax.random.PRNGKey(0)
n_samples = 1500
length = 250

import seaborn as sns


# -jnp.minimum(jnp.abs(x), 1.)
def variance_proxy_vs_n():
    uidx = 20
    n_range = jnp.arange(1, length + 1)
    for p in jnp.linspace(0.1, 0.9, 9):
        my_mgf, my_sims, evaluated_simulations = mgf(OP_key, my_test_fun, n_samples, length, sampling_gaussian, p)
        my_test_fun_sum = evaluated_simulations.cumsum(axis=-1).mean(axis=0)
        res = my_mgf(my_u_axis)
        plt.semilogy(n_range,
                     res[uidx] * jnp.exp(-(my_test_fun_sum * my_u_axis[uidx])), label=f'p={p:.2f}')

    plt.xlabel("n (length of the chain)")
    plt.ylabel("MGF evaluted at some theta, (adjusted for mean, log scale)")
    plt.title("variance proxy (up to constant) vs reset probability p")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig("proxy")
    plt.show()


def mgf_quadratic_in_u(p=0.1):
    # Checking if quadratic in theta
    my_mgf, _, evaluated_simulations = mgf(OP_key, my_test_fun, n_samples, length, sampling_gaussian, p)
    my_test_fun_mean = evaluated_simulations.mean()
    res = my_mgf(my_u_axis)
    plt.semilogy(my_u_axis, res[:, -1] * jnp.exp(-(my_test_fun_mean * length * my_u_axis)))
    plt.title(rf"centered MGFs (log scale) as a function of theta, p={p}")
    plt.legend()
    plt.savefig("centered_mgf_as_a_function_of_u")


def tail_hist(ps):
    if isinstance(ps, float):
        ps = [ps]
    for p in ps:
        my_mgf, my_sims, evaluated_simulations = mgf(OP_key, my_test_fun, n_samples, length, sampling_gaussian, p)
        my_test_fun_sum = my_test_fun(my_sims).mean(axis=-1)

        # --- Remove left outliers ---
        q_low = jnp.quantile(my_test_fun_sum, 0.005)  # remove bottom 0.5%
        trimmed = my_test_fun_sum[my_test_fun_sum >= q_low]

        plt.hist(trimmed, alpha=0.2, label=f'p={p:.2f}', bins=100)
    ref_value = (2 / 3) * evaluated_simulations.mean()
    plt.axvline(
        x=ref_value,  # negative since test_fun = -|x|
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="2/3 of the mean"
    )
    plt.title(f"histogram of ergodic means for {n_samples} chains ")
    plt.legend()
    plt.savefig('histograms')
    plt.show()


def tail_prob(chains, t):
    n_range = jnp.arange(1, chains.shape[1] + 1, 1)
    my_test_fun_cum_mean = chains.cumsum(axis=-1) / n_range
    tail = jnp.mean(my_test_fun_cum_mean >= t, axis=0)
    return tail


def tail_as_n(beta=1 / 3):
    n_range = jnp.arange(1, length + 1)
    for p in jnp.linspace(0.05, 1., 20):
        my_mgf, my_sims, evaluated_simulations = mgf(OP_key, my_test_fun, n_samples, length, sampling_gaussian, p)
        mean = evaluated_simulations.mean()
        tails = tail_prob(evaluated_simulations, (1 - beta) * mean)
        plt.plot(n_range, jnp.log(tails) / p, alpha=0.2, label=f'p={p:.2f}')
    plt.title(f"Tail Probability Evolution (β={beta:.2f})", fontsize=16)
    plt.xlabel("Chain length", fontsize=14)
    plt.ylabel("Tail Probability (log scale) ", fontsize=14)
    plt.legend(frameon=True, fontsize=12, title="Parameters")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("tail_prob")
    plt.show()


import numpy as np


def tail_as_t(ps=0.5, scale_with_p=False):
    """
    Plot log tail probabilities as a function of relative deviation t,
    and fit a quadratic curve to check sub-Gaussian behavior.
    """
    if isinstance(ps, float):
        ps = [ps]

    plt.figure(figsize=(8, 6))
    v_tail_prob = partial(jax.vmap, in_axes=(None, 0))(tail_prob)

    for p in ps:
        # --- Compute MGF + simulations ---
        my_mgf, my_sims, evaluated_simulations = mgf(
            OP_key, my_test_fun, n_samples, floor(length/p) if scale_with_p else length, sampling_gaussian, p
        )

        # Mean of f(X)
        mean = evaluated_simulations.mean()

        # Relative deviations t ∈ [0, 1]
        t = jnp.linspace(0, 1, 200)
        tail_probs = v_tail_prob(evaluated_simulations, (1 - t) * mean)[..., -1]

        # Avoid log(0)
        safe_tail = jnp.clip(tail_probs, 1e-300, 1.0)
        log_tail = np.array(jnp.log(safe_tail))
        first_idx_inf = jnp.argwhere(log_tail == -jnp.inf)[0,0]
        log_tail = log_tail[:first_idx_inf]
        t = t[:first_idx_inf]
        # --- Fit quadratic log P ≈ a t^2 + b t + c ---
        coeffs = np.polyfit(np.array(t), log_tail, 2)
        a, b, c = coeffs

        # --- Generate smooth fitted curve ---
        fit_curve = np.polyval(coeffs, np.array(t))

        # --- Plot original log-tail data ---
        plt.plot(
            t,
            log_tail,
            lw=2.0,
            alpha=0.8,
            label=rf"$p={p:.2f}$, fit: $a={a:.2f}$"
        )

        # --- Plot quadratic fit overlay (dashed) ---
        plt.plot(
            t,
            fit_curve,
            "--",
            color="gray",
            alpha=0.6,
            lw=1.5
        )

    # --- Aesthetics ---
    plt.title(r"Tail Probability $\log P(\cdot)$ vs relative deviation $t$", fontsize=16, pad=10)
    plt.xlabel(r"Relative deviation $t$", fontsize=14)
    plt.ylabel(r"$ \log P(\geq \mu_1(1-t))$", fontsize=14)
    plt.legend(
        title="Reset probability $p$ (with quadratic fit)",
        fontsize=11,
        title_fontsize=12,
        frameon=True,
        loc="best"
    )
    plt.grid(True, which="both", ls="--", lw=0.6, alpha=0.6)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"tails_as_t_fit_scale{scale_with_p}.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    tail_as_t([0.01,0.1,0.25,0.5,1.0], True)
