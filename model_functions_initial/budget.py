#=====================================
#       Budget function 
#=====================================

import jax.numpy as jnp
import numpy as np

#import oap regression function from first stage estimation
# import sys
# sys.path.insert(0,"/Users/frederiklarsen/dcegm/Speciale")



def budget_dcegm(
    lagged_choice,
    savings_end_of_previous_period,
    income_shock_previous_period,
    params,
    options,
    period,
    survival,
    experience,
):
    # Interest on savings
    interest_factor = 1 + params["interest_rate"]
    # Age
    age = (options["start_age"] + period).astype(float) # as float to avoid int64, which complicates wage function.
    # Working hours. indexed on lagged choice, since income today is last period's choice
    hours = options["hours"][lagged_choice]

    # Survival
    alive = (survival == 1)
    death = 1 - alive  # death probability

    # if survival == 0, then choice = 0

    # ====================================================================
    # ————---------------------- Experience ------------------------------
    # ====================================================================    

    # Count total experience as current periods experience times period
    initial_experience = options["max_init_experience"] = 5
    acc_exp = initial_experience + (period * experience)




    # ====================================================================
    # ————------------------------- Wage —————----------------------------
    # ====================================================================

    
    # Wage function, with wage parameters estimated above.
    wage_0 = jnp.exp(params["beta0"] +
        params["beta1"] * age
        + params["beta2"] * age**2
        + income_shock_previous_period
    )
    labor_income = wage_0 * hours

    # ====================================================================
    # ————--------------------- Tax-function —————------------------------
    # ====================================================================

    # Simpple tax-function)
    tax_rate = jnp.where(
        labor_income <= options["inc_threshold"],
        options["tax_base_rate"],
        options["tax_top_rate"],
    )
    net_labor = labor_income * (1 - tax_rate)



    # ====================================================================
    # ————------------------- Old Age Pension —————-----------------------
    # ====================================================================



    # 1) grab your knots
    k1 = options["supp_threshold"]
    k2 = options["oap_threshold"]



    # 2) extract the four coefficients from your fitted statsmodels OLS
    b0, b1, b2, b3 = np.loadtxt("/Users/frederiklarsen/dcegm/Speciale/first_step/oap_params.txt")    # [(Intercept), inc, (inc-k1)+, (inc-k2)+]

    # 3) define a vectorized “predict pension” function
    def predict_oap(labor_income):
        L1 = jnp.maximum(0, labor_income - k1)
        L2 = jnp.maximum(0, labor_income - k2)
        return b0 + b1*labor_income + b2*L1 + b3*L2

    oap_estimate = predict_oap(labor_income)*0.6*(age >= options["retirement_age"]) # 0.4 is the tax rate


    # 1) Determine if agent is retired
    retirement_age = jnp.where(age >= options["retirement_age"], 1, 0)

    # 2) Calculated income above supplement threshold
    income_over_supp = jnp.maximum(0.0, labor_income - options["supp_threshold"])

    # 3)Reduction in supplement
    supp_reduction = options["supp_reduction_rate"] * income_over_supp
    supplement = jnp.maximum(0.0, options["oap_max_supplement"] - supp_reduction)

    # calculate old age pension - full oap until income reaches threshold and then oap is reduced with with oap_reduction_rate pr unit of income above threshold
    income_over_oap = jnp.maximum(0.0, labor_income - options["oap_threshold"])
    oap_reduction = options["oap_reduction_rate"] * income_over_oap
    
    oap = jnp.maximum(0.0, options["oap_base_amount"] - oap_reduction)

    # 4) Samlet årlig pension (grundbeløb + supplement)
    period_pension = oap_estimate

    # 5) Skaler til måned (eller periode­længde) — her antager vi måned
    # period_pension = jnp.where(
    #     retirement_age,
    #     annual_pension,
    #     0.0
    # )

    # Period_pension after tax
    # period_pension = period_pension * (1 - tax_rate)

    # ====================================================================
    # ————---------------- Labor Market Pensions ——-----------------------
    # ====================================================================


    # 1) Calculate labor market pension
    lmpens = params["eta_edu1"] * experience
    # 2) Calculate labor market pension after tax
    lmpens = lmpens * (1 - 0.4)
    lumpsum = jnp.where((age == 67), lmpens, 0.0)



    # ====================================================================
    # ————---------------------- Resources -—————-------------------------
    # ====================================================================

    unemployment_benefit = 1.47912

    # Total resource available for consumption
    resource = jnp.where(lagged_choice > 0,(
        interest_factor * savings_end_of_previous_period
        + net_labor
        + period_pension
        + lumpsum
    ),(interest_factor * savings_end_of_previous_period
       + (unemployment_benefit*0.6)))



    #resource = jnp.where(alive, resource_raw, 0.0) # if alive, resource is resource_raw, else 0.0

    # resource = jnp.where(
    # alive,
    # jnp.maximum(resource_raw, 0.5),
    # 0.0,
    # )

    aux_dict = {
        "wage": wage_0,
        "net_labor": net_labor,
        "period_pension": period_pension,
        "lumpsum": lumpsum,
        "experience": experience,
        "acc_exp": acc_exp,
        "labor_income": labor_income,
        "resource": resource,
    }

    # max(resource, 0.5) to avoid negative consumption - 0 if agent is dead.
    return resource, aux_dict
    