import numpy as np


def zero_diagonal(matrix):
    mat = matrix.copy()
    np.fill_diagonal(mat.values, 0)
    return mat


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
    display_pruned = pruned_matrix.loc[non_zero_rows, non_zero_cols]
    np.fill_diagonal(display_pruned.values, 1)
    return display_pruned
