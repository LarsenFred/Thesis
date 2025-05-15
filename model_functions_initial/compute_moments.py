import pandas as pd
import numpy as np


def compute_simulation_moments(df_sim, start_age, hours_map):

    """
    Konverter et simuleret DataFrame til et moments DataFrame,
    men kun for agenter med survival == 1.
    """
    # 1) Reset index og beregn 'age'
    df = df_sim.reset_index()
    df["age"] = df["period"] + start_age

    # 2) Filtrer kun de levende
    df_alive = df[df["survival"] == 1].copy()

    # 3) Map 'choice' til 'hours_value'
    df_alive["hours_value"] = df_alive["choice"].map(hours_map)

    # 4) Gruppér på age (kun levende)
    grouped = df_alive.groupby("age")

    # 5) Sandsynlighed for at arbejde (choice != 0)
    prob_work = grouped["choice"].apply(lambda x: (x != 0).mean())

    # 6) Fraktion i hver hours‐cluster
    hours_cluster = {
        f"hours_{cat}": grouped["choice"]
            .apply(lambda x, cat=cat: (x == cat).mean())
        for cat in sorted(hours_map.keys())
    }

    # 7) Gennemsnitlig formue i periodens start
    avg_wealth = grouped["wealth_beginning_of_period"].mean()

    # 8) Transitioner work->work og nowork->nowork
    work_work = grouped.apply(
        lambda g: ((g["lagged_choice"] != 0) & (g["choice"] != 0)).mean(),
        include_groups=False
    )

    nowork_nowork = grouped.apply(
        lambda g: ((g["lagged_choice"] == 0) & (g["choice"] == 0)).mean(),
        include_groups=False
    )

    # 9) Betingede momenter for arbejdende
    avg_wage  = grouped["wage"].mean()
    avg_hours = df_alive[df_alive["choice"] != 0].groupby("age")["hours_value"].mean()
    var_wage  = grouped["wage"].var()
    skew_wage = grouped["wage"].skew()

    net_labor = df_alive.groupby("age")["net_labor"].mean()
    survival  = grouped["survival"].mean()  # =1 for alle her

    # 10) Pens hvis det findes, ellers 0
    if "lumpsum" in df_alive:
        pens = grouped["lumpsum"].mean()
    else:
        pens = pd.Series(0.0, index=grouped.groups.keys())

    # Consumption
    avg_consumption = df_alive.groupby("age")["consumption"].mean()

    #nr of individuals at each age
    n_individuals = df_alive.groupby("age")["agent"].count()

    # avg accumulated experience
    avg_experience = grouped["acc_exp"].mean()

    # 11) Sammensæt resultat‐DataFrame
    moments_df = pd.DataFrame({
        "age":            grouped["age"].first(),
        "prob_work":      prob_work,
        **hours_cluster,
        "avg_wealth":     avg_wealth,
        "work_work":      work_work,
        "nowork_nowork":  nowork_nowork,
        "avg_wage":       avg_wage,
        "avg_hours":      avg_hours,
        "var_wage":       var_wage,
        "skew_wage":      skew_wage,
        "pens":           pens,
        "net_labor":      net_labor,
        "survival":       survival,
        "avg_consumption": avg_consumption,
        "n_individuals": n_individuals,
        "avg_experience": avg_experience,
        "avg_labor_income": df_alive[df_alive["choice"] != 0].groupby("age")["labor_income"].mean(),
    })

    # 12) Ryd op: sortér, nulstil index og sæt NaN til 0
    moments_df = moments_df.sort_index().reset_index(drop=True)
    moments_df.fillna(0.0, inplace=True)

    return moments_df




def compute_simulation_moments_with_ci(df_sim, start_age, hours_map):
    """
    Like your compute_simulation_moments but also returns
    95% CIs for each moment at each age, INCLUDING avg_labor_income.
    """
    df = df_sim.reset_index()
    df["age"] = df["period"] + start_age

    # only those alive
    df = df[df["survival"] == 1].copy()

    # map to hours
    df["hours_value"] = df["choice"].map(hours_map)

    g = df.groupby("age")
    N = g.size().rename("N")  # number of agents at each age

    # 1) prob work
    p_work = g["choice"].apply(lambda x: (x != 0).mean()).rename("prob_work")
    var_pw = p_work * (1 - p_work)  # binomial variance

    # 2) hours clusters
    hours_p = {}
    var_h    = {}
    for cat in sorted(hours_map):
        col = f"hours_{cat}"
        ph  = g["choice"].apply(lambda x, cat=cat: (x == cat).mean())
        hours_p[col] = ph
        var_h[col]   = ph * (1 - ph)

    # 3) continuous moments: 
    #    avg_wealth, avg_wage, avg_hours, avg_consumption, avg_experience, AND avg_labor_income
    cont_moms = {}
    cont_vars = {}
    for name, ser in [
        ("avg_wealth",         df["wealth_beginning_of_period"]),
        ("avg_wage",           df["wage"]),
        ("avg_hours",          df[df["choice"] != 0]["hours_value"]),
        ("avg_consumption",    df["consumption"]),
        ("avg_experience",     df["acc_exp"]),
        ("avg_labor_income",   df["labor_income"]),       # ← new!
    ]:
        mean_ser = ser.groupby(df["age"]).mean()
        var_ser  = ser.groupby(df["age"]).var()
        cont_moms[name] = mean_ser
        cont_vars[name] = var_ser

    # 4) transitions etc. (all binary)
    ww     = g.apply(lambda x: ((x["lagged_choice"] != 0) & (x["choice"] != 0)).mean()).rename("work_work")
    nw     = g.apply(lambda x: ((x["lagged_choice"] == 0) & (x["choice"] == 0)).mean()).rename("nowork_nowork")
    var_ww = ww * (1 - ww)
    var_nw = nw * (1 - nw)

    # 5) put it all in a DataFrame
    df_out = pd.DataFrame({
        "N": N,
        "prob_work":     p_work,
        "work_work":     ww,
        "nowork_nowork": nw,
        **hours_p,
        **cont_moms
    })

    # 6) compute SE and 95% CI
    z = 1.96

    # binary ones
    df_out["prob_work_se"]      = np.sqrt(var_pw / df_out["N"])
    df_out["prob_work_lower"]   = df_out["prob_work"] - z * df_out["prob_work_se"]
    df_out["prob_work_upper"]   = df_out["prob_work"] + z * df_out["prob_work_se"]

    df_out["work_work_se"]      = np.sqrt(var_ww / df_out["N"])
    df_out["work_work_lower"]   = df_out["work_work"] - z * df_out["work_work_se"]
    df_out["work_work_upper"]   = df_out["work_work"] + z * df_out["work_work_se"]

    df_out["nowork_nowork_se"]    = np.sqrt(var_nw / df_out["N"])
    df_out["nowork_nowork_lower"] = df_out["nowork_nowork"] - z * df_out["nowork_nowork_se"]
    df_out["nowork_nowork_upper"] = df_out["nowork_nowork"] + z * df_out["nowork_nowork_se"]

    # hours clusters CIs
    for cat in sorted(hours_map):
        col    = f"hours_{cat}"
        varcol = var_h[col]
        se     = np.sqrt(varcol / df_out["N"])
        df_out[f"{col}_se"]    = se
        df_out[f"{col}_lower"] = df_out[col] - z * se
        df_out[f"{col}_upper"] = df_out[col] + z * se

    # continuous ones including avg_labor_income
    for name in cont_moms:
        var_ser = cont_vars[name]
        se      = np.sqrt(var_ser / df_out["N"])
        df_out[f"{name}_se"]    = se
        df_out[f"{name}_lower"] = df_out[name] - z * se
        df_out[f"{name}_upper"] = df_out[name] + z * se

    return df_out.reset_index()