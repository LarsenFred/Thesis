import jax.numpy as jnp
from typing import Dict
import numpy as np

def next_period_experience(period, lagged_choice, experience, options):
    """Calculate next period's experience based on current period's experience and last period labor choice."""
    experience = experience.astype(float)
    period = period.astype(float)

    # grab hours and max_hours as
    hours = options["hours"][lagged_choice]
    max_hours = jnp.array(options["max_hours"])
    init_exp = jnp.array(options["max_init_experience"])

    # t+init_experience
    max_experience_period = period + init_exp

    # E_{t+1} = ((t * E_t) + h_t / max_hours) / (t+1)
    next_exp = (
        (max_experience_period - 1.0) * experience + (hours / max_hours)
    ) / max_experience_period

    return next_exp


# def state_specific_choice_set(period, lagged_choice, options):
#     """Determine the feasible choice set for the current state."""

#     age = (options["start_age"] + period).astype(float)

#     # Retirement is absorbing
#     if (lagged_choice == 0) and (age > options["retirement_age"]):
#         return [0]
#     # If period equal or larger max ret age you have to choose retirement
#     elif period >= options["max_ret_period"]:
#         return [0]
#     # If above minimum retirement period, retirement is possible
#     else:
#         return options["choices"]
    

def get_state_specific_feasible_choice_set(
    lagged_choice: int,
    options: Dict,
) -> np.ndarray:
    """Select state-specific feasible choice set such that retirement is absorbing.

    Will be a user defined function later.

    This is very basic in Ishkakov et al (2017).

    Args:
        state (np.ndarray): Array of shape (n_state_variables,) defining the agent's
            state. In Ishkakov, an agent's state is defined by her (i) age (i.e. the
            current period) and (ii) her lagged labor market choice.
            Hence n_state_variables = 2.
        map_state_to_state_space_index (np.ndarray): Indexer array that maps
            a period-specific state vector to the respective index positions in the
            state space.
            The shape of this object is quite complicated. For each state variable it
            has the number of potential states as rows, i.e.
            (n_potential_states_state_var_1, n_potential_states_state_var_2, ....).

    Returns:
        choice_set (np.ndarray): 1d array of length (n_feasible_choices,) with the
            agent's (restricted) feasible choice set in the given state.

    """
    # lagged_choice is a state variable
    n_choices = options["n_choices"]

    # n_choices = len(options["state_space"]["choices"])

    # Once the agent choses retirement, she can only choose retirement thereafter.
    # Hence, retirement is an absorbing state.
    if lagged_choice == 0 and options["start_age"] > options["retirement_age"]:
        feasible_choice_set = np.array([0])
    else:
        feasible_choice_set = options["choices"]

    return feasible_choice_set


def sparsity_condition(period, lagged_choice, survival, options):
    """Determine the sparsity condition for the current state."""
    final_period = options["n_periods"] - 1

    if survival == 0:
        return {
            "period": final_period,
            "lagged_choice": lagged_choice,
            "survival": survival,
        }
    else:
        return {
            "period": period,
            "lagged_choice": lagged_choice,
            "survival": survival,
        }


def create_state_space_function_dict():
    """Create dictionary with state space functions.

    Returns:
        state_space_functions (dict): Dictionary with state space functions.

    """
    return {
        "next_period_experience": next_period_experience,
        "sparsity_condition": sparsity_condition,
        "state_specific_choice_set": get_state_specific_feasible_choice_set,
    }
