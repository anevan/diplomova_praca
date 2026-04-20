import numpy as np
import pandas as pd

def compute_correlation_matrix(df, method='spearman'):
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation.")
    return numeric_df.corr(method=method)


def zero_diagonal(matrix: pd.DataFrame) -> pd.DataFrame:
    mat = matrix.copy()
    arr = mat.to_numpy().copy() # writable numpy array
    np.fill_diagonal(arr, 0)
    return pd.DataFrame(arr, index=mat.index, columns=mat.columns)

def apply_sigma_mask(matrix, alpha):
    abs_matrix = matrix.abs()
    # print(f"abs_matrix:\n{abs_matrix}")
    # abs_matrix.max() max value in each column
    # abs_matrix.max().max() the single largest absolute correlation in the matrix
    sigma = (abs_matrix.max().max() + abs_matrix.mean().mean()) / 2 + alpha
    # pruned = abs_matrix.where(abs_matrix >= sigma, 0)
    mask = abs_matrix >= sigma
    pruned = matrix.where(mask, 0)
    return pruned, sigma


def modify_pruned_matrix(pruned_matrix):
    # Drop rows and columns that are entirely zero
    non_zero_rows = (pruned_matrix != 0).any(axis=1)
    non_zero_cols = (pruned_matrix != 0).any(axis=0)
    display_pruned = pruned_matrix.loc[non_zero_rows, non_zero_cols].copy()
    arr = display_pruned.to_numpy(copy=True)
    np.fill_diagonal(arr, 1)
    # transforming modified array to DataFrame
    display_pruned.iloc[:, :] = arr
    return display_pruned
