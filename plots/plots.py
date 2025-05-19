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

    # 1) define per‐variable y‐limits and scales
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
        "avg_hours": (0, 1800),
    }
    default_ylim = (None, None)
    exclude = ["age", "var_wage", "pens", "skew_wage"]

    for var in edu.columns:
        if var in exclude:
            continue

        # 2) pick up scale factor

        fig, ax = plt.subplots(figsize=(4, 4))

        # 3) plot empirical & simulated (scaled)
        ax.plot(
            edu["age"],
            edu[var],
            "-",
            label="Empirical",
            color="C0"
        )
        ax.plot(
            moments_sim["age"],
            moments_sim[var],
            "--",
            label="Simulated",
            color="C1"
        )

        # 4) CI fill (also scaled)
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

        # 5) axis labels reflect scaling
        ax.legend()
        plt.tight_layout()

        # 6) save and show
        filename = f"{var}_over_age_{out_subfolder}.png"
        out_path = os.path.join(out_dir, filename)
        fig.savefig(out_path, dpi=100)
        print(f"Saved {out_path}")
        plt.show()
        plt.close(fig)

