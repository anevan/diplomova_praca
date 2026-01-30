import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# datacamp.com/tutorial/correlation, scribbr.com/statistics/correlation-coefficient/
# Pearson when data is continuous and normally distributed with a linear relationship.
# Spearman for ordinal data or when the relationship is monotonic but not necessarily linear. (Any distribution.)
# Kendall is best for small datasets with many ties or when you want a more robust non-parametric measure. (Any distribution.)

def compute_correlation_matrix(df, method='spearman'):
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation.")
    return numeric_df.corr(method=method)

# If normal, the histogram will look bell-shaped, roughly symmetric around the mean
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


# Straight-ish trend → linear
# Curved or complex shape → non-linear
def plot_scatter_plots(df):
    numeric_cols = df.select_dtypes(include=['int', 'float'])
    if numeric_cols.shape[1] == 0:
        print("No numeric columns to plot.")
        return

    sns.set_theme(style="ticks")
    g = sns.pairplot(
        numeric_cols,
        kind="scatter",
        diag_kind="hist",
        plot_kws={'s': 20},
    )

    max_label_len = max(len(str(col)) for col in numeric_cols.columns)
    x_labelpad = 5 + max_label_len * 3
    for ax in g.axes.flatten():
        # rotate x axis labels
        ax.set_xlabel(ax.get_xlabel(), rotation=90)
        # rotate y axis labels
        ax.set_ylabel(ax.get_ylabel(), rotation=0)
        # set y labels alignment
        ax.yaxis.get_label().set_horizontalalignment('right')
        # ax.yaxis.labelpad = 10 + len(ax.get_ylabel()) * 5
        ax.yaxis.labelpad = x_labelpad
        ax.xaxis.labelpad = 20

    # LOWESS regression line chart
    for i, row_var in enumerate(numeric_cols.columns):
        for j, col_var in enumerate(numeric_cols.columns):
            if i != j:
                ax = g.axes[i, j]
                sns.regplot(
                    x=numeric_cols[col_var],
                    y=numeric_cols[row_var],
                    scatter=False,
                    lowess=True,
                    ax=ax,
                    color="red",
                    line_kws={'linewidth': 1.5}
                )

    max_label_length = max(len(str(col)) for col in numeric_cols.columns)
    # dynamic_bottom = 0.1 + max_label_length *  0.009
    dynamic_left = 0.05 + max_label_length * 0.005
    # plt.subplots_adjust(left=dynamic_left,bottom=dynamic_bottom, hspace=0.3, wspace=0.3)
    plt.subplots_adjust(left=dynamic_left, hspace=0.3, wspace=0.3)
    manager = plt.get_current_fig_manager()
    try:
        screen_w = manager.window.screen().geometry().width()
        screen_h = manager.window.screen().geometry().height()
        win_w = int(screen_w * 0.9)
        win_h = int(screen_h * 0.9)
        manager.window.resize(win_w, win_h)
    except Exception:
        pass
    plt.show()
    plt.pause(0.2)
