import pandas as pd
import matplotlib.pyplot as plt

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

