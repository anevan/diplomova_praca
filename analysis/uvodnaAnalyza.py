import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def cv_table(df):
    numeric_df = df.select_dtypes(include='number')
    # MAD = Median Absolute Deviation more robust than Mean Absolute deviation
    mad_per_attribute = numeric_df.apply(lambda x: np.median(np.abs(x - np.median(x))))
    median_per_attribute = numeric_df.median()
    # Coefficient of Variation = CV
    cv_per_attribute = mad_per_attribute / median_per_attribute
    variability_table = pd.DataFrame({
        'Attribute': numeric_df.columns,
        'MAD': mad_per_attribute.values,
        'Median': median_per_attribute.values,
        'CV (MAD/median)': cv_per_attribute.values
    })
    variability_table = (variability_table.sort_values(by='CV (MAD/median)', ascending=False).reset_index(drop=True))
    print(variability_table)

def plot_histograms(df, bins=30):
    numeric_cols = df.select_dtypes(include=['int', 'float'])
    col_names = numeric_cols.columns.tolist()

    n = len(col_names)
    if n == 0:
        print("No numeric columns to plot.")
        return

    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), constrained_layout=True)
    axes = axes.flatten()

    for i, col in enumerate(col_names):
        data = numeric_cols[col].dropna()
        mean_val = data.mean()
        # Histogram
        sns.histplot(
            data,
            bins=bins,
            ax=axes[i],
            kde=False,
            stat="density",
            color="lightskyblue",
        )
        # Density plot
        sns.kdeplot(
            data=data,
            ax=axes[i],
            color="red",
            linewidth=1.5
        )

        # vertical line for mean
        axes[i].axvline(mean_val, color='green', linestyle='--', linewidth=2)
        # axes[i].text(mean_val, axes[i].get_ylim()[1]*0.9, f"Mean: {mean_val:.2f}",
        #              color='green', rotation=90, va='top', ha='right')

        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Scale the window relative to screen size
    manager = plt.get_current_fig_manager()
    try:
        screen_w = manager.window.screen().geometry().width()
        screen_h = manager.window.screen().geometry().height()
        win_w = int(screen_w * 0.5)
        win_h = int(screen_h * 0.7)
        manager.window.resize(win_w, win_h)
    except Exception:
        pass

    plt.show()
    plt.pause(0.2)
