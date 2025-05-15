import jax.numpy as jnp
from typing import Tuple


def final_period_utility(wealth: float, params,) -> Tuple[float, float]:
    b_scale = params["b_scale"]
    xi = params["xi"]
    
    bequest = b_scale * ((wealth ** (1 - xi)) - 1) / (1 - xi)

    return bequest



def marginal_final(wealth: float, params):
    xi = params["xi"]
    return wealth ** (-xi)


final_period_utility = {
    "utility": final_period_utility,
    "marginal_utility": marginal_final,
}