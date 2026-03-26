import pandas as pd
from unittest.mock import patch
import pytest
import matplotlib

matplotlib.use('Agg') # to prevent plots from opening during tests

@pytest.mark.parametrize("side_effects", [
    [
        'n',  # cv
        'n',  # histograms
        'pearson',  # correlation method
        '0.1',  # alpha
        'A','C',  # source/target
        '0.3',  # frac
        '1.0',  # C
        '0.1',  # epsilon
        'scale',  # gamma
        '5',  # max_depth
        '2',  # min_samples_split
        '1',  # min_samples_leaf
        '2',  # palette
        'y', # DFS: list all chains
        'n', # append CSV
        'y' # compute correlation
    ],
    [
        'y',  # cv
        'y',  # histograms
        'spearman',  # correlation method
        '1',  # alpha
        'B', 'C',  # source / target attributes
        '0.5',  # LOESS fraction
        '-1',  # SVR C
        '0.05',  # SVR epsilon
        'auto',  # SVR gamma
        '3',  # max depth (CART)
        '2',  # min samples to split (CART)
        '1',  # min samples in leaf (CART)
        '4',  # plot palette
        'n',  # DFS: list all chains
        'n',  # append CSV
        'n'   # compute correlation
    ],
    [
        'n',  # cv
        'n',  # histograms
        'kendall',  # correlation method
        '0.05',  # alpha
        'X', # incorrect source
        'C', # correct source
        'D',  # incorrect target
        'A', # correct target
        '2',  # LOESS fraction
        '-1',  # SVR C
        '-1',  # SVR epsilon
        '-1',  # SVR gamma
        '-1',  # max depth (CART)
        '-1',  # min samples to split (CART)
        '-1',  # min samples in leaf (CART)
        'bright',  # plot palette
        'n',  # DFS: list all chains
        'n',  # append CSV
        'n'  # compute correlation
    ]
])
def test_main_pipeline(side_effects):
    import main
    main.load_csv = lambda: (pd.DataFrame({
        "A":[1,2,3],
        "B":[2,3,4],
        "C":[3,2,1]
    }), "test_file")

    with patch('builtins.input', side_effect=side_effects):
        main.main()