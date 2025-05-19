#=====================================
#       Budget function 
#=====================================

import jax.numpy as jnp
import numpy as np

#import oap regression function from first stage estimation
# import sys
# sys.path.insert(0,"/Users/frederiklarsen/dcegm/Speciale")



def budget_dcegm_counter_oap(
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
    death = survival == 0  # death probability

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

    # 2) inline-tax logic
    th1 = options["tax_threshold1"]
    th2 = options["tax_threshold2"]
    r2  = options["tax_base_rate"]   # e.g. 0.38
    r3  = options["tax_top_rate"]   # e.g. 0.50

    inc1 = jnp.minimum(labor_income, th1)
    inc2 = jnp.minimum(jnp.maximum(labor_income - th1, 0.0), th2 - th1)
    inc3 = jnp.maximum(labor_income - th2, 0.0)

    tax_labor = r2 * inc2 + r3 * inc3      # 0% on inc1
    net_labor = labor_income - tax_labor



    # ====================================================================
    # ————------------------- Old Age Pension —————-----------------------
    # ====================================================================

    #==============================================================
    # For the counterfactual, every agent above the retirement age
    # can receive full OAP regardless of income. 



    # # 1) grab your knots
    # k1 = options["supp_threshold"]
    # k2 = options["oap_threshold"]



    # # 2) extract the four coefficients from your fitted statsmodels OLS
    # b0, b1, b2, b3 = np.loadtxt("/Users/frederiklarsen/dcegm/Speciale/first_step/oap_params.txt")    # [(Intercept), inc, (inc-k1)+, (inc-k2)+]

    # # 3) define a vectorized “predict pension” function
    # def predict_oap(labor_income):
    #     L1 = jnp.maximum(0, labor_income - k1)
    #     L2 = jnp.maximum(0, labor_income - k2)
    #     return b0 + b1*labor_income + b2*L1 + b3*L2

    # oap_estimate = predict_oap(labor_income)*0.6*(age >= options["retirement_age"]) # 0.4 is the tax rate


    # # 4) Samlet årlig pension (grundbeløb + supplement)
    # period_pension = oap_estimate

    # Period_pension after tax
    period_pension = jnp.where((age >= options["retirement_age"]),(options["oap_base_amount"]+options["oap_max_supplement"]) * 0.63, 0)

    # ====================================================================
    # ————---------------- Labor Market Pensions ——-----------------------
    # ====================================================================


    # 1) Calculate labor market pension
    lmpens = params["eta_edu"] * experience
    # 2) Calculate labor market pension after tax
    lmpens = lmpens * (1 - 0.4)
    lumpsum = jnp.where((age == 67), lmpens, 0.0)



       # ====================================================================
    # ————---------------------- Resources -—————-------------------------
    # ====================================================================

    unemployment_benefit = 1

    # Total resource available for consumption
    resource = jnp.where(
        lagged_choice > 0,
        (
            interest_factor * savings_end_of_previous_period
            + net_labor
            + period_pension
            + lumpsum
        ),
        (
            jnp.maximum(unemployment_benefit * 0.62 * (age < options["retirement_age"]), savings_end_of_previous_period*interest_factor)+ period_pension * (lagged_choice == 0)
        ),
    )

    resource = jnp.where(
        death, savings_end_of_previous_period, resource
    )  # if dead, resource is 0.0

    # resource = jnp.where(alive, resource_raw, 0.0) # if alive, resource is resource_raw, else 0.0

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
    