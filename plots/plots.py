import os
import matplotlib.pyplot as plt

def plot_empirical_vs_simulated_with_ci(
    edu,
    moments_sim,
    out_base_dir,
    out_subfolder,
    var_labels=None,
    var_scales=None,
    ylims=None,
    figsize=(4,4),
    dpi=100
):
    # avoid ambiguous truth‐value errors
    if var_labels is None: var_labels = {}
    if var_scales is None: var_scales = {}
    if ylims      is None: ylims      = {}

    out_dir = os.path.join(out_base_dir, out_subfolder)
    os.makedirs(out_dir, exist_ok=True)

    # only plot vars present in both
    plot_vars = [v for v in edu.columns if v != "age" and v in moments_sim.columns]

    for var in plot_vars:
        pretty = var_labels.get(var, var)
        scale  = var_scales.get(var, 1.0)
        ymin, ymax = ylims.get(var, (None, None))

        # extract & scale
        x_emp = edu["age"]
        y_emp = edu[var] * scale
        x_sim = moments_sim["age"]
        y_sim = moments_sim[var] * scale

        # CI
        low_col, high_col = f"{var}_lower", f"{var}_upper"
        has_ci = low_col in moments_sim and high_col in moments_sim
        if has_ci:
            y_low = moments_sim[low_col] * scale
            y_high = moments_sim[high_col] * scale

        # plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_emp, y_emp, "-", label="Empirical", color="C0")
        ax.plot(x_sim, y_sim, "--", label="Simulated", color="C1")
        if has_ci:
            ax.fill_between(x_sim, y_low, y_high, color="C1", alpha=0.2, label="95% CI")

        # formatting
        ax.set_title(pretty)
        ax.set_xlabel("Age")
        ax.grid(True)
        ax.set_ylim(ymin, ymax)
        ax.legend()
        plt.tight_layout()

        # save
        fn = f"{var}_over_age_{out_subfolder}.png"
        path = os.path.join(out_dir, fn)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {path}")
        plt.show()