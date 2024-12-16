from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxopt import ScipyRootFinding

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
        fun_applied_to_grid = jax.vmap(func)(grid)
        return grid.at[jnp.argmax(fun_applied_to_grid, keepdims=True).at[0].get()].get()

    return optimize_within_a_grid


def make_bisection(minmax, interval, n_iter=100):
    r"""
    Bisection method for derivable one-dimensional function f: \mathbb{R} -> \mathbb{R}.
    """
    _min, _max = minmax
    ap, bp = interval

    def bisection_procedure(func: Callable[[ArrayLike], ArrayLike], x: ArrayLike):
        grad_f = lambda x: jax.grad(func)(x)

        a, b = interval
        a = jax.lax.max(a, _min)
        b = jax.lax.min(b, _max)
        c = (a + b) / 2

        def iter_fun(_, ab_tuple):
            a, b = ab_tuple
            b = (grad_f(c) * grad_f(a) < 0) * c + (1 - (grad_f(c) * grad_f(a) < 0)) * b
            a = (grad_f(c) * grad_f(a) > 0) * c + (1 - (grad_f(c) * grad_f(a) > 0)) * a
            ab_tuple = (a, b)
            return ab_tuple

        a, b = jax.lax.fori_loop(0, n_iter, iter_fun, (a, b))

        return (a + b) / 2

    return bisection_procedure


def make_bisection_2(minmax, interval, n_iter=100):
    r"""
    Bisection method for derivable one-dimensional function f: \mathbb{R} -> \mathbb{R}.
    """
    _min, _max = minmax
    ap, bp = interval

    def bisection_procedure(func: Callable[[ArrayLike], ArrayLike], x: ArrayLike):
        grad_f = lambda x: jax.grad(func)(x)

        a, b = jax.lax.max(x + ap, _min), jax.lax.min(x + bp, _max)
        c = (a + b) / 2

        def iter_fun(_, ab_tuple):
            a, b = ab_tuple
            b = (grad_f(c) * grad_f(a) < 0) * c + (1 - (grad_f(c) * grad_f(a) < 0)) * b
            a = (grad_f(c) * grad_f(a) > 0) * c + (1 - (grad_f(c) * grad_f(a) > 0)) * a
            ab_tuple = (a, b)
            return ab_tuple

        a, b = jax.lax.fori_loop(0, n_iter, iter_fun, (a, b))

        return (a + b) / 2

    return bisection_procedure

def make_ScipyRootFinding(lmbda, interval):
    a, b = interval
    def solve(func, x):
        grad_f = jax.grad(func)
        grad_grad_f = jax.grad(lambda x: grad_f(x).at[0].get())
        def iter_fun(i, inps):
            x, lmbda = inps
            _grad_f = grad_f(x)
            ggrad_f = grad_grad_f(x)
            step = _grad_f / ggrad_f * (_grad_f / ggrad_f > 1e-4) * (ggrad_f != 0.)
            proposed_x = x - lmbda * step
            proposed_x = jnp.isnan(x) * x + (1 - jnp.isnan(x)) * proposed_x
            return proposed_x, lmbda/2
        x, _ = jax.lax.fori_loop(0, 5, iter_fun, (x, lmbda))
        return jax.lax.max(jax.lax.min(x, b), a)
    return solve