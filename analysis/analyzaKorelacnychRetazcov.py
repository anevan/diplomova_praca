import os
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from analysis.regregresneModely import predict_loess,predict_cart, predict_svr
from analysis.metriky import smape as sMAPE

def analyze_correlation_chains(
        corr_chains,
        matrix,
        df,
        dataset_name,
        frac,
        c,
        epsilon,
        gamma,
        max_depth,
        min_samples_leaf,
        min_samples_split):

    rows = []

    for chain_info in corr_chains:
        pathf_method = chain_info["method"]
        correlation_chain = chain_info["best_path"]
        avrg_last_smape = chain_info["avrg_last_smape"]
        first = correlation_chain[0]
        last = correlation_chain[-1]

        # priama korelácia
        priama_korelacia = abs(matrix.loc[first, last])

        # nepriama korelácia (priemer korelácii)
        pairs = list(zip(correlation_chain[:-1], correlation_chain[1:]))
        edge_corrs = [abs(matrix.loc[a, b]) for a, b in pairs]
        nepriama_korelacia = np.mean(edge_corrs)

        # sMAPE cez tri modely
        x = df[[first]].values
        y = df[last].values
        loess_preds = predict_loess(x, y, frac)
        svr_preds = predict_svr(x, y, c, epsilon, gamma)
        cart_preds = predict_cart(
            x, y,
            max_depth,
            min_samples_split,
            min_samples_leaf
        )
        smape_mean = np.mean([
            sMAPE(y, loess_preds),
            sMAPE(y, svr_preds),
            sMAPE(y, cart_preds)
        ])

        # nový riadok
        rows.append({
            "dataset": dataset_name,
            "grafovy_algoritmus": pathf_method,
            "pociatocny_cielovy": f"{first}_{last}",
            "priama_korelacia": priama_korelacia,
            "nepriama_korelacia": nepriama_korelacia,
            "sMAPE": smape_mean,
            "sMAPE(y)": avrg_last_smape
        })

    df_corr_chain_analysis = pd.DataFrame(rows)
    filename = "correlation_chain_analysis.csv"
    print(f"Add new rows to {filename} to analyse relationship between correlation in correlation chains and sMAPE.")
    print(df_corr_chain_analysis)
    # Ask user to append to CSV
    answer = input("\nDo you want to append these rows to the CSV file? (y/n): ").strip().lower()
    if answer == "y":
        while True:
            try:
                if os.path.exists(filename):
                    df_corr_chain_analysis.to_csv(filename, mode='a', header=False, index=False)
                else:
                    df_corr_chain_analysis.to_csv(filename, mode='w', header=True, index=False)
                print(f"Rows successfully saved to: {filename}")
                break
            except PermissionError:
                input(f"Permission denied for '{filename}'. Close the file if it's open, then press Enter to retry...")
    else:
        print("Rows were not saved.")

    # Optional correlation analysis on CSV
    if os.path.exists(filename):
        df_full = pd.read_csv(filename)
        print(f"\nCurrent content of {filename}:")
        print(df_full)

        do_corr = input("\nDo you want to compute correlation analysis on the CSV file? (y/n): ").strip().lower()

        if do_corr == "y":
            #  GLOBAL CORRELATION (all datasets together)
            if len(df_full) >= 2:
                direct_corr = df_full["priama_korelacia"].values
                indirect_corr = df_full["nepriama_korelacia"].values
                smape = df_full["sMAPE"].values
                smape_y = df_full["sMAPE(y)"].values

                # Direct correlation vs sMAPE
                if np.all(direct_corr == direct_corr[0]) or np.all(smape == smape[0]):
                    print("Cannot compute correlation between priama_korelacia and sMAPE: one of the inputs is constant.")
                else:
                    corr_direct, p_direct = pearsonr(direct_corr, smape)
                    print(f"Direct correlation vs sMAPE: r = {corr_direct:.4f}, p-value = {p_direct:.16f}")

                # Indirect correlation vs sMAPE
                if np.all(indirect_corr == indirect_corr[0]) or np.all(smape_y == smape_y[0]):
                    print("Cannot compute correlation between nepriama_korelacia and sMAPE(y): one of the inputs is constant.")
                else:
                    corr_indirect, p_indirect = pearsonr(indirect_corr, smape_y)
                    print(f"Indirect correlation vs sMAPE(y): r = {corr_indirect:.4f}, p-value = {p_indirect:.16f}")
            else:
                print("\nCannot compute correlation: CSV has fewer than 2 rows.")

            # PER-DATASET CORRELATION
            print("\nPER-DATASET CORRELATION")

            grouped = df_full.groupby("dataset")

            for dataset_name, df_dataset in grouped:
                print(f"Dataset: {dataset_name}")

                if len(df_dataset) < 2:
                    print("  Not enough rows to compute correlation.")
                    continue

                direct_corr = df_dataset["priama_korelacia"].values
                indirect_corr = df_dataset["nepriama_korelacia"].values
                smape = df_dataset["sMAPE"].values
                smape_y = df_dataset["sMAPE(y)"].values

                # Direct correlation
                if np.all(direct_corr == direct_corr[0]) or np.all(smape == smape[0]):
                    print("  Cannot compute direct correlation: one input is constant.")
                else:
                    corr_direct, p_direct = pearsonr(direct_corr, smape)
                    print(f"  Direct correlation vs sMAPE: r = {corr_direct:.4f}, p-value = {p_direct:.16f}")

                # Indirect correlation
                if np.all(indirect_corr == indirect_corr[0]) or np.all(smape_y == smape_y[0]):
                    print("  Cannot compute indirect correlation: one input is constant.")
                else:
                    corr_indirect, p_indirect = pearsonr(indirect_corr, smape_y)
                    print(f"  Indirect correlation vs sMAPE(y): r = {corr_indirect:.4f}, p-value = {p_indirect:.16f}")
        else:
            print("Analysis of the relationship between correlation in correlation chains and sMAPE was skipped.")
    else:
        print(f"\nCreate {filename} for further analysis.")
    return