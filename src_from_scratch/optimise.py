from typing import Callable, Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike


def make_optimize_within_a_grid(minmax: Tuple[float, float], interval: Tuple[float, float], n_steps: int) -> Callable[
    [Callable[[ArrayLike], ArrayLike], ArrayLike], ArrayLike]:
    """
    Constructing a maximisation procedure of a function over a unidimensional grid.
    The grid is centered around the initialisation point of the maximisation procedure, is made of
    n_steps regularly distributed points. All the points outside the interval defined by the minmax tuple are flattened.
    """
    min, max = minmax
    a, b = interval

    def optimize_within_a_grid(func: Callable[[ArrayLike], ArrayLike], x: ArrayLike) -> ArrayLike:
        grid = x + jnp.linspace(a, b, n_steps)
        grid = jnp.minimum(jnp.maximum(grid, min), max)
        fun_applied_to_grid = func(grid)
        return grid.at[jnp.argmax(fun_applied_to_grid, keepdims=True).at[0].get()].get()

    return optimize_within_a_grid
