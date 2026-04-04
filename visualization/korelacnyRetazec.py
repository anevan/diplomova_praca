from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns


def plot_correlation_chain_graph(matrix, path, score=None, error_metrics=None, corr_method=None, alpha=None, sigma=None,
                                 path_finding_method=None):
    if not path or len(path) < 2:
        print("Invalid path provided for visualization.")
        return None, None, None

    symbol_map = { "spearman": "ρ", "pearson": "r", "kendall": "τ"}
    corr_symbol = symbol_map.get(corr_method, "r") # r set as default

    # CREATE THE UNDIRECTED GRAPH - correlation chain
    g = nx.Graph()
    for node in path:
        g.add_node(node)
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        weight = matrix.loc[a, b]
        color = "red" if weight > 0 else "blue"
        g.add_edge(a, b, weight=weight, label=f"{corr_symbol} = {weight:.2f}", color=color)

    # ADJUST EDGE LENGTHS based on labels and metrics
    # max_label_len = max(len(str(node)) for node in path)
    max_label_len = max(
        max(len(word) for word in str(node).split())
        for node in path
    )

    # increase edge width based on rmse and mae
    font_char_width = 0.3
    label_width = 0
    threshold = 9  # extra_padding if total integer digits exceed this num
    padding_per_digit = 0.4
    extra_padding = 0
    if error_metrics:
        # rmse/mae array with the largest total integer digits
        max_digits_sum = 0
        for metrics in error_metrics.values():
            for key in ('rmse', 'mae'):
                values = metrics.get(key, [])
                if values:
                    digits_sum = sum(len(str(int(abs(val)))) for val in values if isinstance(val, (int, float)))
                    if digits_sum > max_digits_sum:
                        max_digits_sum = digits_sum
        # Add extra padding only if it exceeds the threshold
        if max_digits_sum > threshold:
            extra_digits = max_digits_sum - threshold
            extra_padding = padding_per_digit * extra_digits
    label_width += extra_padding
    #print(f"Label width - error metrics: {label_width}")

    # increase edge width based on attribute name
    has_multiple_words = any(len(str(node).split()) > 1 for node in path)
    if has_multiple_words:
        label_width += max_label_len * font_char_width
    #print(f"Label width - multiple words: {label_width}")

    # increase edge width based on smape
    threshold_low = 5
    threshold_high = 8
    # maximum sum of integer digits in any smape array
    max_smape_digits_sum = max(
        sum(len(str(int(abs(val)))) for val in metrics.get('smape', []) if isinstance(val, (int, float)))
        for metrics in error_metrics.values()
    )
    if max_smape_digits_sum > threshold_low and extra_padding == 0 and has_multiple_words == 0:
        if max_smape_digits_sum < threshold_high:
            label_width += 2
        else:  # >= threshold_high
            label_width += 3
    #print(f"Label width - smape threshold: {label_width}")

    # increase edge width based on attribute name (length)
    label_width += max_label_len * font_char_width
    #print(f"Label width - final: {label_width}")
    base_spacing = 4.0
    if label_width < 5 < len(path):
        if label_width <= base_spacing/1.5:
            label_width += base_spacing
        else:
            label_width += base_spacing/2
    #print(f"Label width - small width: {label_width}")

    # final edge width
    horizontal_spacing = max(base_spacing, label_width)
    #print(f"Edge width: {horizontal_spacing}")

    # Vertical offset for node labels
    y_offset = 12 # moves metrics text
    node_count = len(path)
    # width of the figure based on nodes and edge widths
    fig_width = (node_count + 1) * horizontal_spacing
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    pos = {node: ((i + 1) * horizontal_spacing, y_offset) for i, node in enumerate(path)}

    # DRAW EDGES
    nx.draw_networkx_edges(g, pos, width=1.5, edge_color="black", ax=ax)
    edge_labels = nx.get_edge_attributes(g, "label")
    # EDGE LABEL COLOR ADDED
    for (x, y), label in edge_labels.items():
        label_color = g[x][y]['color']  # red if positive, blue if negative
        nx.draw_networkx_edge_labels(
            g,
            pos,
            edge_labels={(x, y): label},
            font_color=label_color,
            font_size=20,
            label_pos=0.5,
            rotate=False,
            ax=ax
        )
    # ADD (MULTI-LINE) NODES USING matplotlib's ax.annotate() instead of nx.draw_networkx_nodes()
    for node, (x, y) in pos.items():
        # Split node label into multiple lines if it has spaces
        node_label = "\n".join(str(node).split(" "))
        ax.annotate(
            node_label,
            xy=(x, y),
            xytext=(0, 0),
            textcoords='offset points',
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white", linewidth=1.5),
            fontsize=20,
        )

    # main title
    details = []
    if corr_method:
        details.append(f"Correlation coefficient: {corr_method}")
    if alpha is not None:
        details.append(f"α = {alpha:.2f}")
    if sigma is not None:
        details.append(f"σ = {sigma:.4f}")
    if path_finding_method:
        details.append(f"Path-finding algorithm: {path_finding_method}")
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
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2 # midpoint for placing text

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

            # Place the metrics text at edge midpoint
            ax.text(
                mid_x, mid_y - 0.4,
                text,
                ha='center', va='center',
                multialignment='left',
                fontsize=20,
                color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
            )

    # Inset sMAPE line chart
    width_per_edge = max(0.03, min(0.15, 0.3 / len(g_edges)))
    inset_width = max(0.7, max(0.35, len(g_edges) * width_per_edge))
    inset_left = 0.5 - inset_width / 2

    # sMAPE above 100? for y-axis scaling
    above_100 = any(
        val > 100
        for metrics in error_metrics.values()
        for val in metrics['smape']
    )
    inset_bottom, inset_height = (-0.3, 0.3) #if above_100 else (-0.2, 0.2)
    # Create inset axis
    inset_ax = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])
    ymax = 200 if above_100 else 100
    inset_ax.set_ylim(0, ymax)
    inset_ax.set_yticks(range(0, ymax + 1, 20))
    # Configure ticks
    inset_ax.tick_params(axis='y', labelsize=14)
    inset_ax.tick_params(axis='x', labelsize=14)
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['right'].set_visible(False)

    # x positions for edges
    x = list(range(1, len(g_edges) + 1))

    # color palettes
    color_palette = sns.color_palette("PiYG", 10)
    blue_palette = sns.color_palette("Blues")
    pastel_palette = sns.color_palette("pastel")
    pastel_palette1 = sns.color_palette("Pastel1")

    palette_map = {
        "meh": [None, None, None],  # default matplotlib colors
        "pastel-red": [pastel_palette[0], pastel_palette1[0], pastel_palette[2]],  # blue, red, green
        "pastel-orange": [pastel_palette[0], pastel_palette[1], pastel_palette[2]],  # blue, orange, green
        "bright": [color_palette[1], blue_palette[5], color_palette[9]]  # pink, blue, green
    }
    colors = palette_map.get(palette, [None, None, None])
    # Plot sMAPE for each model
    inset_ax.plot(x, smape_loess, marker='o', label='LOESS', color=colors[0])
    inset_ax.plot(x, smape_svr, marker='o', label='SVR', color=colors[1])
    inset_ax.plot(x, smape_cart, marker='o', label='CART', color=colors[2])
    # Configure inset axis
    inset_ax.set_xlim(1 - 0.05, len(g_edges) + 0.05)
    inset_ax.set_xticks(x)
    inset_ax.set_xticklabels([f"{a}\n↓\n{b}" for (a, b) in g_edges], rotation=45, ha='center', fontsize=16)
    inset_ax.set_ylabel("sMAPE (%)", fontsize=16)
    inset_ax.set_title("Prediction Error (sMAPE per Edge)", fontsize=20, pad=30)
    inset_ax.grid(True, linestyle='--', alpha=0.6)
    inset_ax.legend(
        fontsize=12,
        loc='center left',
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
            matrix, path, score, error_metrics,
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
