import jax.numpy as jnp


def next_period_experience(period, lagged_choice, experience, options):
    """Calculate next period's experience based on current period's experience and last period labor choice.
    
    """
    experience = experience.astype(float)
    period     = period.astype(float)

    # grab hours and max_hours as 
    hours      = options["hours"][lagged_choice]
    max_hours  = jnp.array(options["max_hours"])
    init_exp   = jnp.array(options["max_init_experience"])

    # t+init_experience
    max_experience_period = period + init_exp

    # E_{t+1} = ((t * E_t) + h_t / max_hours) / (t+1)
    next_exp = (
        (max_experience_period - 1.0) * experience
        + (hours/max_hours)
    ) / max_experience_period

    return next_exp 


def state_specific_choice_set(period, lagged_choice, options):
    """Determine the feasible choice set for the current state.

    """

    age = (options["start_age"] + period).astype(float)

    # Retirement is absorbing
    if (lagged_choice == 0) and (age > options["retirement_age"]):
        return [0]
    # If period equal or larger max ret age you have to choose retirement
    elif period >= options["max_ret_period"]:
        return [0]
    # If above minimum retirement period, retirement is possible
    else: 
            return options["choices"]
    
def sparsity_condition(
    period, lagged_choice, survival, options
):
    """Determine the sparsity condition for the current state.

    """
    last_period = (options["n_periods"] - 1)

    if survival == 0:
        return {
            "period": last_period,
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
            #"sparsity_condition": sparsity_condition,
            #"state_specific_choice_set": state_specific_choice_set,
    }