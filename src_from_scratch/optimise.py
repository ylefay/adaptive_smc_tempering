from typing import Callable

import jax.numpy as jnp
from jax.typing import ArrayLike

def make_optimize_within_a_grid(minmax, interval, n_steps):
    min, max = minmax
    a, b = interval
    def optimize_within_a_grid(func: Callable[[ArrayLike], ArrayLike], x: ArrayLike) -> ArrayLike:
        grid = x + jnp.linspace(a, b, n_steps)
        grid = jnp.minimum(jnp.maximum(grid, min), max)
        fun_applied_to_grid = func(grid)
        return grid.at[jnp.argmax(fun_applied_to_grid, keepdims=True).at[0].get()].get()
    return optimize_within_a_grid

