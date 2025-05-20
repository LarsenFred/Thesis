import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_counterfactual_diff(df_base, df_cf, metrics, age_col='age'):
    """
    Compute absolute and percent differences between baseline and counterfactual moments.

    Parameters
    ----------
    df_base : pd.DataFrame
        Baseline moments DataFrame. Must contain an 'age' column (or named by age_col).
    df_cf : pd.DataFrame
        Counterfactual moments DataFrame. Must contain same age_col and metrics columns.
    metrics : list of str
        Column names of the moments you want to compare, e.g. ['avg_hours','avg_wealth','avg_consumption'].
    age_col : str, optional
        Name of the age column. Default is 'age'.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed 0.. with columns:
          - age
          - for each m in metrics:
              m_diff  : absolute difference = cf[m] - base[m]
              m_pct   : percent change = 100*(cf[m] - base[m]) / base[m]
    """
    # 1) set age as index to align
    base = df_base.set_index(age_col)
    cf   = df_cf.set_index(age_col)

    # 2) intersect ages
    common = base.index.intersection(cf.index)
    base = base.loc[common]
    cf   = cf.loc[common]

    # 3) build output DataFrame
    out = pd.DataFrame({age_col: common})
    out = out.reset_index(drop=True)

    # 4) compute for each metric
    for m in metrics:
        b = base[m]
        c = cf[m]
        diff = c - b
        # avoid division by zero in pct
        pct = diff.div(b.replace({0: pd.NA})) * 100
        out[f'{m}_diff'] = diff.values
        out[f'{m}_pct']  = pct.values

    return out

def plot_counterfactual_diff(df_diff, metrics, age_col='age'):
    """
    For each metric, plots level‐change (solid) and %‐change (dashed) on twin y‐axes.
    """
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4*n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        ages = df_diff[age_col]
        ax.plot(ages, df_diff[f'{m}_diff'], label=f'{m} Δ level', lw=2)
        ax.set_ylabel(f'{m} Δ')
        ax2 = ax.twinx()
        ax2.plot(ages, df_diff[f'{m}_pct'], '--', label=f'{m} % change', lw=1.5, color='gray')
        ax2.set_ylabel(f'{m} %')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    axes[-1].set_xlabel('Age')
    plt.tight_layout()
    plt.show()



def plot_cf_diff_separate(df_diff, metrics, age_col='age', fig_size=6):
    """
    For each metric in `metrics`, draw a separate square plot (fig_size × fig_size)
    showing:
      - absolute change: solid line + grid
      - % change: dashed line on a twin‐y axis
    """
    for m in metrics:
        ages = df_diff[age_col]
        diff = df_diff[f'{m}_diff']
        pct  = df_diff[f'{m}_pct']
        
        # 1) square figure
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # 2) absolute‐change curve
        ax.plot(ages, diff, '-', lw=2, label=f'{m} Δ level')
        ax.set_xlabel('Age')
        ax.set_ylabel(f'{m} Δ')
        ax.grid(True)
        
        # 3) twin‐axis for % change
        ax2 = ax.twinx()
        ax2.plot(ages, pct, '--', lw=1.5, color='gray', label=f'{m} % change')
        ax2.set_ylabel(f'{m} %')
        
        # 4) legends & title
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title(f'Counterfactual change in {m}')
        
        plt.tight_layout()
        plt.show()



def compute_diff_by_edu(df_init, df_cf, metrics, edu_col="edu", age_col="age"):
    diff_dfs = []
    edus = df_init[edu_col].unique()

    for edu in edus:
        base = df_init[df_init[edu_col] == edu].set_index(age_col)
        cf   = df_cf [df_cf [edu_col] == edu].set_index(age_col)
        ages = base.index.intersection(cf.index)

        df_diff = pd.DataFrame({edu_col: edu, age_col: ages})
        for m in metrics:
            b = base.loc[ages, m]
            c = cf.loc[ages, m]
            diff = c - b
            pct  = diff.div(b.replace({0: np.nan})) * 100
            df_diff[f"{m}_diff"] = diff.values
            df_diff[f"{m}_pct"]  = pct.values

        diff_dfs.append(df_diff)

    return pd.concat(diff_dfs, ignore_index=True) 

def plot_metrics_individual(
    df_diff,
    metrics,
    edus=None,
    edu_labels=None,
    metric_labels=None,
    diff_suffix="_diff",
    pct_suffix="_pct",
    figsize=(10, 5),
    legend_y_offset=-0.15,
):
    if edus is None:
        edus = df_diff['edu'].unique()
    if edu_labels is None:
        edu_labels = {edu: edu for edu in edus}
    if metric_labels is None:
        metric_labels = {}

    for m in metrics:
        pretty_metric = metric_labels.get(m, m)
        fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=figsize)

        for edu in edus:
            label = edu_labels.get(edu, edu)
            sub = df_diff[df_diff['edu'] == edu]
            ax_abs.plot(sub['age'], sub[f'{m}{diff_suffix}'], lw=2, label=label)
            ax_pct.plot(sub['age'], sub[f'{m}{pct_suffix}'],  lw=2, label=label)

        ax_abs.set_title(f"{pretty_metric} – Absolute Change")
        ax_abs.set_xlabel("Age")
        ax_abs.set_ylabel("Δ level")
        ax_abs.grid(True)

        ax_pct.set_title(f"{pretty_metric} – Percent Change")
        ax_pct.set_xlabel("Age")
        ax_pct.set_ylabel("% change")
        ax_pct.grid(True)

        handles, labels = ax_abs.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y_offset),
            ncol=len(edus),
            frameon=False,
            fontsize=9.17,
        )
        plt.subplots_adjust(bottom=0.25, wspace=0.3)
        plt.tight_layout()
        plt.show()



def plot_separate_metric_panels(
    df_diff,
    metrics,
    edus=None,
    edu_labels=None,
    metric_labels=None,
    diff_suffix="_diff",
    pct_suffix="_pct",
    figsize=(8,4),
    legend_y_offset=-0.2
):
    """
    For each metric in `metrics`, makes two *separate* figures:
      1) Absolute change
      2) Percent change
    Legend is drawn without a frame, centered below the axis.
    """
    if edus is None:
        edus = df_diff['edu'].unique()
    if edu_labels is None:
        edu_labels = {e:e for e in edus}
    if metric_labels is None:
        metric_labels = {}

    for m in metrics:
        pretty = metric_labels.get(m, m)

        # --- Absolute Change Figure ---
        fig, ax = plt.subplots(figsize=figsize)
        for e in edus:
            lbl = edu_labels.get(e, e)
            sub = df_diff[df_diff['edu']==e]
            ax.plot(sub['age'], sub[f'{m}{diff_suffix}'], lw=2, label=lbl)

        ax.set_title(f"{pretty} – Absolute Change")
        ax.set_xlabel("Age")
        ax.set_ylabel("Δ level")
        ax.grid(True)
        # frameless legend below x-axis
        ax.legend(
            title="Education",
            loc='upper center',
            bbox_to_anchor=(0.5, legend_y_offset),
            ncol=len(edus),
            frameon=False
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.show()
        plt.close(fig)

        # --- Percent Change Figure ---
        fig, ax = plt.subplots(figsize=figsize)
        for e in edus:
            lbl = edu_labels.get(e, e)
            sub = df_diff[df_diff['edu']==e]
            ax.plot(sub['age'], sub[f'{m}{pct_suffix}'], lw=2, label=lbl)

        ax.set_title(f"{pretty} – Percent Change")
        ax.set_xlabel("Age")
        ax.set_ylabel("% change")
        ax.grid(True)
        ax.legend(
            title="Education",
            loc='upper center',
            bbox_to_anchor=(0.5, legend_y_offset),
            ncol=len(edus),
            frameon=False
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.show()
        plt.close(fig)