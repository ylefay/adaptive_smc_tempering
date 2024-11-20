import blackjax.smc.tempered as _tempered

from functools import partial
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

import blackjax.smc as smc
from blackjax.base import SamplingAlgorithm
from blackjax.smc.base import SMCState
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["TemperedSMCState", "init"]

TemperedSMCState = _tempered.TemperedSMCState
init = _tempered.init

