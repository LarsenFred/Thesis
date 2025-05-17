import jax.numpy as jnp


def flow_util(consumption, choice, params, period, options, lagged_choice):
    # Utility parameter
    rho = params["rho"]

    # Disutility parameters
    age = (options["start_age"] + period).astype(float)
    gamma = params["gamma"][choice - 1]  # remove the first element

    ##############################################
    ########## Disutility of working #############
    ##############################################
    working = choice > 0  # 1 if choice is working, 0 if not
    # -------  Zero disutility from working if unemployed

    # Age component of disutility
    age_linear = jnp.where(age > 50, params["kappa1"] * (age - 50), 0.0)
    age_quadratic = jnp.where(age > 50, params["kappa2"] * (age - 50) ** 2, 0.0)

    # + age_linear

    # estimate total disutility
    disutil_0 = working * (1.0 + age_linear + age_quadratic) * gamma

    # No disutility from working, if you dont work.
    disutil = jnp.where(working, disutil_0, 0.0)  # if working == 0, disutility = 0

    ##############################################
    ############# Transaction costs  #############
    ##############################################

    # transition cost when going from working to unemployed, and vice versa
    trans_cost = jnp.where(
        choice != lagged_choice, params["phi"], 0.0
    )  # from changing choice

    # Utility for agents that are alive.
    u = (
        ((consumption ** (1 - rho)) - 1) / (1 - rho) - disutil - trans_cost
    )  # working*gamma*hours*(1+(kappa1*age)*age_1+(kappa2*age_2)**2*age_2) #jax.lax.select(working, gamma, 0) - if a NaN included

    # Utility for agents that are dead. no utility from consumption, only utility from bequest.
    # u_dead = jnp.where(first_time_dead, bequest, 0.0) # if first time dead, utility is -inf

    # u = jnp.where(survival == 1, u_alive, u_dead) # if survival == 1, utility is alive utility, else dead utility

    return u


def marginal_utility(consumption, params):
    rho = params["rho"]
    u_prime = consumption ** (-rho)
    return u_prime


def inverse_marginal_utility(marginal_utility, params):
    rho = params["rho"]
    return marginal_utility ** (-1 / rho)


utility_functions = {
    "utility": flow_util,
    "inverse_marginal_utility": inverse_marginal_utility,
    "marginal_utility": marginal_utility,
}
