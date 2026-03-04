import math
import os
import seaborn as sns
import matplotlib.pyplot as plt


def save_heatmap(matrix, file, method=None, sigma=None, alpha=None):
    # Determine if it's a pruned or full correlation matrix
    is_pruned = sigma is not None or alpha is not None

    # Build the visualization directory path
    parts = ["outputs", file]
    if method:
        parts.append(method.lower())
    if is_pruned and alpha is not None:
        parts.append(f"alpha_{alpha:.2f}")
    output_dir = os.path.join(*parts)
    os.makedirs(output_dir, exist_ok=True)

    # File name: correlation_matrix.png or pruned_correlation_matrix.png
    filename = "pruned_correlation_matrix.png" if is_pruned else "correlation_matrix.png"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"Heatmap already exists: {output_path}")
        return

    # Generate plot title
    title = f"Pruned Correlation Matrix of {file} dataset" if is_pruned else f"Correlation Matrix of {file} dataset"
    if is_pruned and sigma is not None and alpha is not None:
        title += f" (σ = {sigma:.3f}, α = {alpha:.3f})"
    if method:
        title += f" [{method.capitalize()}]"

    # font sizes
    n = matrix.shape[0]  # number of attributes
    figsize = (max(10, n * 1), max(8, n * 1))
    if n <= 50:
        annot_fontsize = max(6, min(15, 50 / math.log(n + 1)))
    else:
        annot_fontsize = min(22, 12 + (n - 50) * 0.06)
    tick_fontsize = max(6, min(12, 40 / math.log(n + 1)))
    title_fontsize = cbar_labelsize = max(20, int(18 + (n - 9) * 0.8)) # linearly scaling
    cbar_tickfontsize = max(16, int(14 + (n - 9) * 0.7))
    cbar_ticklength = 4 + (n - 9) * 0.04
    cbar_tickwidth = 1 + (n - 9) * 0.03

    # heatmap
    plt.figure(figsize=figsize)
    sns.set_context("notebook", font_scale=1.5)
    ax = sns.heatmap(
        matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={"size": annot_fontsize},
        linewidths=0.5,
        vmin=-1,
        vmax=1
    )

    # colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label("Correlation coefficient", fontsize=cbar_labelsize)
    cbar.ax.tick_params(labelsize=cbar_tickfontsize, length=cbar_ticklength, width=cbar_tickwidth)

    plt.xticks(rotation=45, ha="right", fontsize=tick_fontsize)
    plt.yticks(rotation=0, fontsize=tick_fontsize)
    plt.title(title, fontsize=title_fontsize, pad=max(20, n * 0.5))
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved to: {output_path}")