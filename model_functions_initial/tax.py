import jax.numpy as jnp

def tax_liability(labor_income, th1, th2, rate2, rate3):
    """
    Piecewise tax liability with
      – 0% på indkomst [0, th1]
      – rate2 på indkomst (th1, th2]
      – rate3 på indkomst > th2
    th1, th2, rate2, rate3 kan hentes fra options eller params.
    """
    # indkomst i de tre stykker
    inc1 = jnp.minimum(labor_income, th1)
    inc2 = jnp.minimum(jnp.maximum(labor_income - th1, 0.0), th2 - th1)
    inc3 = jnp.maximum(labor_income - th2, 0.0)

    # marginalsatser
    rate1 = 0.0  # ingen skat under th1

    return rate1 * inc1 + rate2 * inc2 + rate3 * inc3