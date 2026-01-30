import os
from pprint import pprint

from pyarrow import timestamp

from data.loader import load_csv
from processing.regresnaAnalyza import multi_model_chain_predict, print_error_metrics
from ui.cli import (get_correlation_method, get_alpha, get_user_input_columns, get_path_finding_method, get_frac, \
    get_max_depth, get_plot_palette, get_svr_C, get_svr_epsilon, get_svr_gamma, get_min_samples_leaf, get_min_samples_split)
from processing.orezavanie import zero_diagonal, apply_sigma_mask, modify_pruned_matrix
from processing.identifikaciaRetazcov import run_selected_path_finding_method
from processing.korelacnaAnalyza import compute_correlation_matrix, plot_histograms, plot_scatter_plots
from output.korelacnaMatica import save_heatmap
from output.korelacnyRetazec import save_correlation_chains
import pandas as pd
import numpy as np
import time

# Ensure pandas doesn't truncate the output
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)  # allow wide console output
pd.set_option('display.max_colwidth', None)  # no truncation of cell contents

# Prevent scientific notation
pd.set_option('display.float_format', lambda x: f'{x:.4f}')


def main():
    df, file = load_csv()

    run_scatter = input(
        "Do you want to print the coefficient of variation (CV) of each numeric attribute? (y/n):").strip().lower()
    if run_scatter == 'y':
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

        # variability_table = variability_table.sort_values(by='CV (σ/mean)', ascending=False).reset_index(drop=True)
        variability_table = (variability_table.sort_values(by='CV (MAD/median)', ascending=False).reset_index(drop=True))
        print(variability_table)

    run_hist = input("Do you want to plot histograms? (y/n): ").strip().lower()
    if run_hist == 'y':
        plot_histograms(df)

    # run_scatter = input("Do you want to plot scatter plots?\n"
    #                     "WARNING: This may be slow for bigger datasets. (y/n):").strip().lower()
    # if run_scatter == 'y':
    #     plot_scatter_plots(df)

    # Correlation analysis
    corr_method = get_correlation_method()
    matrix = compute_correlation_matrix(df, corr_method)
    print("Correlation matrix computed. Preview:")
    print(matrix.round(4))
    save_heatmap(matrix, file, corr_method)
    ###
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
            print("\nPruned matrix is empty with this alpha.")
            print("Please try a smaller alpha (lower α → lower sigma → more values kept).")
        else:
            break
    save_heatmap(modified_pruned_matrix, file, corr_method, sigma, alpha)

    # Correlation chains
    columns = list(modified_pruned_matrix.columns)
    in_col, out_col = get_user_input_columns(columns)
    pathf_method = get_path_finding_method()
    # Calculate path(s) and score(s)
    start = time.time()
    paths, scores = run_selected_path_finding_method(
        method=pathf_method,
        matrix=pruned_matrix.abs().round(4),
        start=in_col,
        end=out_col)
    end = time.time()
    print("Execution time:", end - start, "seconds")
    ###
    # If no paths were returned
    if not paths or not isinstance(paths, list) or len(paths) == 0:
        print(f"No path found from '{in_col}' to '{out_col}' — nodes may be disconnected.")
        return
    ###
    # If multiple paths exist, pick the one with the highest score
    if len(paths) > 1 and scores:
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_path = paths[best_idx]
        best_score = scores[best_idx]
    else:
        best_path = paths[0]
        best_score = scores[0] if scores else sum(abs(matrix.loc[a, b]) for a, b in zip(paths[0], paths[0][1:]))

    # Regression analysis
    frac = get_frac()
    c = get_svr_C()
    epsilon = get_svr_epsilon()
    gamma = get_svr_gamma()
    max_depth = get_max_depth()
    min_samples_leaf = get_min_samples_leaf()
    min_samples_split = get_min_samples_split()
    palette = get_plot_palette()
    # palette = 'meh'
    # Run regression on the best path
    start = time.time()
    df_with_predictions, error_metrics= multi_model_chain_predict(df,
                                                                   path=best_path,
                                                                   frac=frac,
                                                                   c=c,
                                                                   epsilon=epsilon,
                                                                   gamma=gamma,
                                                                   max_depth=max_depth,
                                                                   min_samples_leaf = min_samples_leaf,
                                                                   min_samples_split = min_samples_split)
    end = time.time()
    print("Execution time:", end - start, "seconds")
    print_error_metrics(error_metrics)

    ###
    # Save visualization of the best chain with outputs from regression analysis
    timestamp = save_correlation_chains(
        matrix=matrix,
        paths=[(best_path, best_score)],
        file_name=file,
        method=corr_method,
        alpha=alpha,
        sigma=sigma,
        path_finding_method=pathf_method,
        start_node=in_col,
        end_node=out_col,
        error_metrics=error_metrics,
        palette=palette
    )

    # LOG FILE
    save_folder = os.path.join(
        "outputs",
        file,
        corr_method,
        f"alpha_{alpha:.2f}",
        pathf_method,
        f"{in_col}_to_{out_col}")
    os.makedirs(save_folder, exist_ok=True)
    filename = f"log_{timestamp}.txt"
    full_path = os.path.join(save_folder, filename)
    with open(full_path, "w") as f:
        f.write(f"Selected CSV file/dataset: {file}.csv\n")
        f.write(f"Selected correlation coefficient: {corr_method}\n")
        f.write(f"Used for pruning\n\tAlpha: {alpha}\n\tSigma: {sigma}\n")
        f.write(f"Source attribute: {in_col}\nTarget attribute: {out_col}\n")
        f.write(f"Path finding method: {pathf_method}\n")
        f.write(f"Identified path: {best_path}\nPath's score (correlation sum): {best_score}\n")
        f.write(f"LOESS fraction value: {frac}\n")
        f.write(f"SVR C parameter: {c}\n")
        f.write(f"SVR epsilon parameter: {epsilon}\n")
        f.write(f"SVR gamma parameter: {gamma}\n")
        f.write(f"Max depth (CART): {max_depth}\n")
        f.write(f"Minimum samples in a leaf (CART): {min_samples_leaf}\n")
        f.write(f"Minimum samples needed to make a split (CART): {min_samples_split}\n")
        f.write("Error metrics\n")
        pprint(error_metrics, stream=f)
    print(f"Log file was saved to: {full_path}")

if __name__ == "__main__":
    main()
