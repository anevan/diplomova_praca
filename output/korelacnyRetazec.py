import datetime
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns


def plot_correlation_chain_graph(matrix, path, score=None, corr_method=None, alpha=None, sigma=None,
                                 path_finding_method=None):
    if not path or len(path) < 2:
        print("Invalid path provided for visualization.")
        return None, None, None

    symbol_map = {
        "spearman": "ρ",
        "pearson": "r",
        "kendall": "τ",
    }

    corr_symbol = symbol_map.get(corr_method, "r") # r set as default

    g = nx.Graph()
    for node in path:
        g.add_node(node)
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        weight = matrix.loc[a, b]
        g.add_edge(a, b, weight=weight, label=f"{corr_symbol} = {weight:.2f}")

    max_label_len = max(len(str(node)) for node in path)
    font_char_width = 0.25
    label_width = max_label_len * font_char_width
    base_spacing = 5.0
    horizontal_spacing = max(base_spacing, label_width)

    y_offset = 10
    node_count = len(path)
    padding_nodes = 1
    fig_width = (node_count + 2 * padding_nodes) * horizontal_spacing
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    pos = {node: ((i + padding_nodes) * horizontal_spacing, y_offset) for i, node in enumerate(path)}

    nx.draw_networkx_edges(g, pos, width=2, edge_color="gray", ax=ax)
    edge_labels = nx.get_edge_attributes(g, "label")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color="black", font_size=20,
                                 label_pos=0.5, rotate=False, ax=ax)

    for node, (x, y) in pos.items():
        ax.annotate(
            node,
            xy=(x, y),
            xytext=(0, 0),
            textcoords='offset points',
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white", linewidth=1.5),
            fontsize=20,
            fontweight='bold'
        )

    details = []
    if corr_method:
        details.append(f"Correlation coefficient: {corr_method}")
    if alpha is not None:
        details.append(f"α = {alpha:.2f}")
    if sigma is not None:
        details.append(f"σ = {sigma:.4f}")
    if path_finding_method:
        details.append(f"Path finding method: {path_finding_method}")
    # if score is not None:
    #     details.append(f"Correlation sum: {score:.2f}")

    title = " | ".join(details) if details else "Correlation Chain"
    ax.set_title(title, fontsize=20)
    ax.axis("off")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.5, bottom=0.15)

    return fig, ax, pos


def add_error_metrics_to_plot(fig, ax, pos, path, error_metrics, palette):
    g_edges = list(zip(path[:-1], path[1:]))
    smape_loess, smape_svr, smape_cart = [], [], []

    for (a, b) in g_edges:
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2

        key = (a, b) if (a, b) in error_metrics else (b, a)
        metrics = error_metrics.get(key)
        if metrics:
            rmse = metrics.get("rmse", [float("nan")] * 3)
            mae = metrics.get("mae", [float("nan")] * 3)
            smape = metrics.get("smape", [None, None, None])

            smape_loess.append(smape[0])
            smape_svr.append(smape[1])
            smape_cart.append(smape[2])

            text = (
                f"RMSE  [{rmse[0]:.2f}, {rmse[1]:.2f}, {rmse[2]:.2f}]\n"
                f"MAE   [{mae[0]:.2f}, {mae[1]:.2f}, {mae[2]:.2f}]\n"
                f"sMAPE [{smape[0]:.1f}%, {smape[1]:.1f}%, {smape[2]:.1f}%]"
            )
            ax.text(
                mid_x - 1.5, mid_y - 0.15,
                text,
                ha='left', va='top',
                fontsize=20,
                color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
            )

    # Dynamic width calculation based on number of edges
    num_edges = len(g_edges)
    width_per_edge = max(0.03, min(0.15, 0.3 / num_edges))
    # Total width, clamped
    inset_width = max(0.7, max(0.35, num_edges * width_per_edge))
    inset_left = 0.5 - inset_width / 2

    # inset plot for sMAPE
    # sMAPE above 100?
    above_100 = any(
        val > 100
        for metrics in error_metrics.values()
        for val in metrics['smape']
    )
    inset_bottom, inset_height = (-0.3, 0.3) if above_100 else (-0.2, 0.2)
    # Create the inset axis
    inset_ax = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])

    # Set inset axis limits and ticks based on sMAPE values
    ymax = 200 if above_100 else 100
    inset_ax.set_ylim(0, ymax)
    inset_ax.set_yticks(range(0, ymax + 1, 20))

    inset_ax.tick_params(axis='y', labelsize=14)
    inset_ax.tick_params(axis='x', labelsize=14)
    # Format inset axis
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['right'].set_visible(False)
    x = list(range(1, len(g_edges) + 1))

    # bright colors
    color_palette = sns.color_palette("PiYG", 10)
    blue_palette = sns.color_palette("Blues")
    # pastel colors
    pastel_palette = sns.color_palette("pastel")
    pastel_palette1 = sns.color_palette("Pastel1")

    palette_map = {
        "meh": [None, None, None],  # default matplotlib colors
        "pastel-red": [pastel_palette[0], pastel_palette1[0], pastel_palette[2]],  # blue, red, green
        "pastel-orange": [pastel_palette[0], pastel_palette[1], pastel_palette[2]],  # blue, orange, green
        "bright": [color_palette[1], blue_palette[5], color_palette[9]]  # pink, blue, green
    }
    colors = palette_map.get(palette, [None, None, None])

    inset_ax.plot(x, smape_loess, marker='o', label='LOESS', color=colors[0])
    inset_ax.plot(x, smape_svr, marker='o', label='SVR', color=colors[1])
    inset_ax.plot(x, smape_cart, marker='o', label='CART', color=colors[2])

    inset_ax.set_xticks(x)
    inset_ax.set_xticklabels([f"{a}→{b}" for (a, b) in g_edges], rotation=45, ha='right', fontsize=16)
    inset_ax.set_ylabel("sMAPE (%)", fontsize=16)
    inset_ax.set_title("Prediction Error (sMAPE per Edge)", fontsize=20, pad=20)
    inset_ax.grid(True, linestyle='--', alpha=0.6)
    inset_ax.legend(
        fontsize=12,
        loc='center left',  # anchor legend's left center point
        bbox_to_anchor=(1, 0.5),  # slightly outside right edge, vertically centered
        borderaxespad=0
    )


def save_correlation_chains(matrix, paths, file_name, method, alpha, sigma, path_finding_method,
                            start_node, end_node, error_metrics=None, palette=None):
    out_dir = os.path.join(
        "outputs",
        file_name,
        method,
        f"alpha_{alpha:.2f}",
        path_finding_method,
        f"{start_node}_to_{end_node}"
    )
    os.makedirs(out_dir, exist_ok=True)

    if not paths:
        print("No paths provided (None or empty). Skipping saving.")
        return None

    # Convert single path format into consistent list-of-tuples format
    if isinstance(paths, list) and isinstance(paths[0], str):
        paths = [(paths, None)]

    # Filter invalid paths
    paths = [
        (path, score)
        for path, score in paths
        if path and isinstance(path, (list, tuple)) and len(path) > 1
    ]

    if not paths:
        print("All provided paths were empty, None, or too short. Skipping saving.")
        return None

    for idx, (path, score) in enumerate(paths, start=1):
        fig, ax, pos = plot_correlation_chain_graph(
            matrix, path, score,
            corr_method=method,
            alpha=alpha,
            sigma=sigma,
            path_finding_method=path_finding_method
        )

        if fig is None:
            continue

        # Add error metrics
        if error_metrics:
            add_error_metrics_to_plot(fig, ax, pos, path, error_metrics, palette)

        # Save the figure
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(out_dir, f"correlation_chain_{idx}_{timestamp}.png")
        fig.savefig(file_path, bbox_inches='tight', pad_inches=0.4)
        plt.close(fig)
        print(f"Saved: {file_path}")
        return timestamp
    return None
