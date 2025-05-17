import os
import matplotlib.pyplot as plt
import numpy as np


def plot_empirical_vs_simulated_with_ci(
    edu,
    moments_sim,
    out_base_dir,  # e.g. "/…/Figurer"
    out_subfolder,  # e.g. "edu1", "edu2", "edu3"
):
    """
    Plot empirical vs simulated moments with 95% CI and save to
    {out_base_dir}/{out_subfolder}/{var}_over_age_{out_subfolder}.png
    """

    out_dir = os.path.join(out_base_dir, out_subfolder)
    os.makedirs(out_dir, exist_ok=True)

    ylims = {
        "prob_work": (0, 1),
        "hours_0": (0, 1),
        "hours_1": (0, 1),
        "hours_2": (0, 1),
        "hours_3": (0, 1),
        "hours_4": (0, 1),
        "work_work": (0, 1),
        "nowork_nowork": (0, 1),
        "avg_wealth": (0, 30),
        "avg_experience": (0, 50),
    }
    default_ylim = (None, None)
    exclude = ["age", "var_wage", "pens", "skew_wage"]

    for var in edu.columns:
        if var in exclude:
            continue

        fig, ax = plt.subplots(figsize=(4, 4))

        ax.plot(edu["age"], edu[var], "-", label="Empirical", color="C0")
        ax.plot(
            moments_sim["age"], moments_sim[var], "--", label="Simulated", color="C1"
        )

        low_col, high_col = f"{var}_lower", f"{var}_upper"
        if low_col in moments_sim and high_col in moments_sim:
            ax.fill_between(
                moments_sim["age"],
                moments_sim[low_col],
                moments_sim[high_col],
                color="C1",
                alpha=0.2,
                label="95% CI",
            )

        ax.grid()
        y0, y1 = ylims.get(var, default_ylim)
        if y0 is not None or y1 is not None:
            ax.set_ylim(y0, y1)
        ax.legend()
        plt.tight_layout()

        # save
        filename = f"{var}_over_age_{out_subfolder}.png"
        out_path = os.path.join(out_dir, filename)
        fig.savefig(out_path, dpi=100)
        print(f"Saved {out_path}")

        # **show before close**
        plt.show()

        # now you can close
        plt.close(fig)


# # 1) Define a grid of annual labor incomes from 0 to 100 000 DKK
# income = np.linspace(0, 600_000, 500)

# # 2) Calculate the supplement component:
# #    income above the supplement threshold is taxed away at the supplement reduction rate
# supp_threshold = options["model_params"]["supp_threshold"] * 100_000  # convert model units → DKK
# supp_reduction_rate = options["model_params"]["supp_reduction_rate"]
# income_over_supp = np.maximum(0.0, income - supp_threshold)
# supplement = np.maximum(
#     0.0,
#     options["model_params"]["oap_max_supplement"] * 100_000  # max supplement in DKK
#     - supp_reduction_rate * income_over_supp
# )

# # 3) Calculate the base pension component:
# #    income above the OAP threshold reduces the base pension at the OAP reduction rate
# oap_threshold = options["model_params"]["oap_threshold"] * 100_000
# oap_reduction_rate = options["model_params"]["oap_reduction_rate"]
# income_over_oap = np.maximum(0.0, income - oap_threshold)
# base_pension = np.maximum(
#     0.0,
#     options["model_params"]["oap_base_amount"] * 100_000  # base amount in DKK
#     - oap_reduction_rate * income_over_oap
# )

# # 4) Total annual pension is the sum of base pension and supplement
# annual_pension = base_pension + supplement

# # 5) Plot the piecewise-linear pension schedule
# plt.figure(figsize=(8,5))
# plt.plot(income, annual_pension, lw=2, label="Total Pension")
# # mark the two thresholds
# plt.axvline(supp_threshold, color="blue", linestyle="--", label="Supplement Threshold")
# plt.axvline(oap_threshold, color="green", linestyle="--", label="Base Pension Threshold")
# plt.xlabel("Annual Labor Income (DKK)")
# plt.ylabel("Annual Old-Age Pension (DKK)")
# plt.title("Old-Age Pension Schedule as a Function of Income")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
