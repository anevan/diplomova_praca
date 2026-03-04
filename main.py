import os
from pprint import pprint
import pandas as pd
import time
from loader import load_csv
from analysis.regresnaAnalyza import multi_model_chained_predict, print_error_metrics
from cli import (get_correlation_method, get_alpha, get_user_input_columns,  get_frac, get_max_depth, get_plot_palette,
                 get_svr_C, get_svr_epsilon, get_svr_gamma, get_min_samples_leaf, get_min_samples_split)
from analysis.filtrovanie import zero_diagonal, apply_sigma_mask, modify_pruned_matrix
from analysis.identifikaciaRetazcov import run_selected_path_finding_method
from analysis.korelacnaAnalyza import compute_correlation_matrix
from analysis.analyzaKorelacnychRetazcov import analyze_correlation_chains
from visualization.korelacnaMatica import save_heatmap
from visualization.korelacnyRetazec import save_correlation_chains
from analysis.uvodnaAnalyza import cv_table, plot_histograms

# Ensure pandas doesn't truncate the visualization
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)  # allow wide console visualization
pd.set_option('display.max_colwidth', None)  # no truncation of cell contents

# Prevent scientific notation
pd.set_option('display.float_format', lambda x: f'{x:.4f}')


def main():
    # Load dataset
    df, file = load_csv()

    # Optional analyses
    run_cv= input(
        "Do you want to print the coefficient of variation (CV) of each numeric attribute? (y/n):").strip().lower()
    if run_cv == 'y':
        cv_table(df)
    run_hist = input("Do you want to plot histograms? (y/n): ").strip().lower()
    if run_hist == 'y':
        plot_histograms(df)

    # Correlation analysis
    corr_method = get_correlation_method()
    matrix = compute_correlation_matrix(df, corr_method)
    print("Correlation matrix computed. Preview:")
    print(matrix.round(4))

    # Save correlation heatmap
    save_heatmap(matrix, file, corr_method)

    # Prune correlation matrix
    zero_matrix = zero_diagonal(matrix)
    # print(f"zero_matrix:\n{zero_matrix}")
    while True:
        alpha = get_alpha()
        print(f"Using alpha: {alpha}")

        pruned_matrix, sigma = apply_sigma_mask(zero_matrix, alpha=alpha)
        modified_pruned_matrix = modify_pruned_matrix(pruned_matrix)

        print(f"[Sigma Threshold: {sigma:.4f}]")
        print("[Pruned Correlation Matrix (values ≥ σ)]")
        print(modified_pruned_matrix.round(4))
        if modified_pruned_matrix.empty:
            print("Pruned matrix is empty. Try a smaller alpha.")
        else:
            break

    # Save pruned matrix heatmap
    save_heatmap(modified_pruned_matrix, file, corr_method, sigma, alpha)

    # User inputs for further analysis
    columns = list(modified_pruned_matrix.columns)
    # Selected start/end attributes
    in_col, out_col = get_user_input_columns(columns)
    # Regression hyperparameters
    frac = get_frac()
    c = get_svr_C()
    epsilon = get_svr_epsilon()
    gamma = get_svr_gamma()
    max_depth = get_max_depth()
    min_samples_split = get_min_samples_split()
    min_samples_leaf = get_min_samples_leaf()
    palette = get_plot_palette()
    # palette = 'meh'

    # Loop over all graph algorithms
    all_graph_algorithms = {
        "greedy": "greedy (chamtivé hľadanie)",
        "greedy+dfs": "greed+DFS (chamtivé hľadanie do hĺbky)",
        "dfs": "DFS (prehľadávanie do hĺbky)",
        "a_star": "A* (A* algoritmus)"
    }
    all_paths = []

    for algorithm_key, display_name in all_graph_algorithms.items():
        print(f"\n=== Using {display_name} graph algorithm ===")
        start = time.time()
        paths, scores = run_selected_path_finding_method(
            method=algorithm_key,
            matrix=pruned_matrix.abs().round(4),
            start=in_col,
            end=out_col)
        end = time.time()
        print("Graph algorithm execution completed in", end - start, "s.")

        if not paths or not isinstance(paths, list) or len(paths) == 0:
            print(f"No path found with {display_name} method.")
            continue

        # Select path with the highest corr sum
        if len(paths) > 1 and scores:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_path = paths[best_idx]
            best_score = scores[best_idx]
        else:
            best_path = paths[0]
            best_score = scores[0] if scores else sum(abs(matrix.loc[a, b]) for a, b in zip(paths[0], paths[0][1:]))

        # Regression analysis
        start = time.time()
        _, error_metrics = multi_model_chained_predict(
            df,
            path=best_path,
            frac=frac,
            c=c,
            epsilon=epsilon,
            gamma=gamma,
            max_depth=max_depth,
            min_samples_leaf = min_samples_leaf,
            min_samples_split = min_samples_split)
        end = time.time()
        print("Chained prediction execution completed in", end - start, "s.")
        # print_error_metrics(error_metrics)

        # Save correlation chain visualization
        timestamp = save_correlation_chains(
            matrix=matrix,
            paths=[(best_path, best_score)],
            file_name=file,
            method=corr_method,
            alpha=alpha,
            sigma=sigma,
            path_finding_method=algorithm_key,
            start_node=in_col,
            end_node=out_col,
            error_metrics=error_metrics,
            palette=palette
        )

        # Save log file
        save_folder = os.path.join(
            "outputs",
            file,
            corr_method,
            f"alpha_{alpha:.2f}",
            algorithm_key,
            f"{in_col}_to_{out_col}")
        os.makedirs(save_folder, exist_ok=True)
        new_filename = f"log_{timestamp}.txt"
        full_path = os.path.join(save_folder, new_filename)
        with open(full_path, "w") as f:
            f.write(f"Selected CSV file/dataset: {file}.csv\n")
            f.write(f"Selected correlation coefficient: {corr_method}\n")
            f.write(f"Used for pruning\n\tAlpha: {alpha}\n\tSigma: {sigma}\n")
            f.write(f"Source attribute: {in_col}\nTarget attribute: {out_col}\n")
            f.write(f"Graph algorithm: {algorithm_key}\n")
            f.write(f"Identified path: {best_path}\nPath's correlation sum: {best_score}\n")
            f.write(f"LOESS fraction value: {frac}\n")
            f.write(f"SVR C parameter: {c}\n")
            f.write(f"SVR epsilon parameter: {epsilon}\n")
            f.write(f"SVR gamma parameter: {gamma}\n")
            f.write(f"Max depth (CART): {max_depth}\n")
            f.write(f"Minimum samples needed to make a split (CART): {min_samples_split}\n")
            f.write(f"Minimum samples in a leaf (CART): {min_samples_leaf}\n")
            f.write("Error metrics\n")
            pprint(error_metrics, stream=f)
        print(f"Log file was saved to: {full_path}\n")

        # all correlation chains
        all_paths.append({
            "method": algorithm_key,
            "best_path": best_path,
        })

    analyze_correlation_chains(all_paths,
                               matrix,
                               df,
                               file,
                               frac=frac,
                               c=c,
                               epsilon=epsilon,
                               gamma=gamma,
                               max_depth=max_depth,
                               min_samples_leaf=min_samples_leaf,
                               min_samples_split=min_samples_split
                               )

if __name__ == "__main__":
    main()
