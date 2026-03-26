import pandas as pd
from unittest.mock import patch

@patch('builtins.input', side_effect=[
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
    'pastel-red',  # palette
    'y', # DFS: list all chains
    'n', # append CSV
    'y' # compute correlation
])
def test_main_pipeline(mock_input):
    import main
    main.load_csv = lambda: (pd.DataFrame({
        "A":[1,2,3],
        "B":[2,3,4],
        "C":[3,2,1]
    }), "test_file")
    main.main()