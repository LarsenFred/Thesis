# msm_objective.py

import numpy as np
from scipy.optimize import minimize
import jax.numpy as jnp

# import whatever simulation helpers you already have
from dcegm.pre_processing.setup_model import setup_model
from dcegm.sim_interface import get_sol_and_sim_func_for_model
from dcegm.simulation.sim_utils import create_simulation_df
from model_functions_initial.compute_moments import compute_simulation_moments


def crit_func_scipy(
    theta_array: np.ndarray,
    params: dict,
    start_age: int,
    hours_map: dict,
    empirical_moms_df,
    keep_cols: list,
) -> float:
    """
    High-level MSM objective function.

    Args:
      theta_array: array of structural parameters to estimate.
      params:      your model‐params dict (will get mutated here).
      start_age:   the age at period 0.
      hours_map:   mapping choice→hours for simulation‐to‐moments.
      empirical_moms_df: DataFrame of empirical moments (indexed by age).
      keep_cols:   list of moment‐names to include in the objective.

    Returns:
      crit_val: weighted sum of squared moment deviations.
    """

    # 1) overwrite the fixed stuff
    params["interest_rate"] = 0.01
    params["beta0"], params["beta1"], params["beta2"] = -6.8961, 0.0272, -0.0002

    # 2) unpack theta_array however you like:
    params["sigma"] = theta_array[0]
    params["lambda"] = theta_array[1]
    params["beta"] = theta_array[2]
    params["rho"] = theta_array[3]
    params["gamma"] = jnp.array(theta_array[4:8])
    params["kappa2"] = theta_array[8]
    params["phi"]    = theta_array[9]
    params["b_scale"]= theta_array[10]
    params["xi"]     = theta_array[11]
    params["eta_edu1"]= theta_array[12]

    # 3) run your sim
    sim_out = sim_func_aux(params)
    df_sim = create_simulation_df(sim_out["sim_dict"])

    # 4) build simulated‐moments DataFrame
    sim_moms = compute_simulation_moments(df_sim, start_age, hours_map)

    # 5) drop any nuisance cols and select the ones you care about
    for c in ("pens",):
        if c in sim_moms:      sim_moms = sim_moms.drop(columns=[c])
        if c in empirical_moms_df: empirical_moms_df = empirical_moms_df.drop(columns=[c])
    sim_moms = sim_moms[keep_cols]
    emp_moms = empirical_moms_df[keep_cols]

    # 6) now build the weighted‐sos
    sim_vals = sim_moms.to_numpy()
    emp_vals = emp_moms.to_numpy()
    diff = sim_vals - emp_vals

    emp_var = np.nanvar(emp_vals, axis=0, ddof=1)
    weights = 1.0 / (emp_var + 1e-6)

    crit = 0.0
    for i, w in enumerate(weights):
        crit += w * np.nansum(diff[:, i]**2)

    return float(crit)


def estimate_msm(
    initial_guess: np.ndarray,
    params, start_age, hours_map,
    empirical_moms_df, keep_cols,
    **minimize_kwargs
):
    """
    Wrapper to actually call scipy.minimize on your crit_func_scipy.
    """
    obj = lambda θ: crit_func_scipy(
        θ, params, start_age, hours_map, empirical_moms_df, keep_cols
    )
    return minimize(obj, initial_guess, **minimize_kwargs)