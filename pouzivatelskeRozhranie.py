def get_user_input_columns(columns):
    print("\nFor the feed-backward construction of correlation chains in multidimensional datasets, "
          "\nselect the source and target attributes from the available attributes:")
    print(columns)
    while True:
        x = input("Select the source attribute (počiatočný atribút): ").strip()
        if x not in columns:
            print(f"'{x}' is not a valid attribute name. Please choose from the list.")
        else:
            print(f"Source attribute: {x}")
            break
    while True:
        y = input("Select the target attribute (cieľový atribút): ").strip()
        if y not in columns:
            print(f"'{y}' is not a valid attribute name. Please choose from the list.")
        else:
            print(f"Target attribute: {y}")
            break
    return x, y


def get_correlation_method():
    methods = ['pearson', 'spearman', 'kendall']
    default = 'spearman'
    method = input(f"Choose correlation method ({', '.join(methods)}) [default={default}]: ").strip().lower()

    if method == '':
        return default
    elif method in methods:
        return method
    else:
        print(f"Invalid method. Using default: {default}")
        return default


def get_alpha(default=0.1):
    try:
        alpha_input = input(f"\nEnter alpha value for pruning (α ∈ ⟨0, 0.3⟩) [default={default}]: ").strip()
        if alpha_input == '':
            print(f"No input provided. Using default: {default}")
            return default
        alpha = float(alpha_input)
        if not (0 <= alpha <= 0.3):
            raise ValueError("Alpha must be between 0 and 0.3")
        return alpha
    except ValueError as e:
        print(f"Invalid input: {e}. Using default value: {default}")
        return default


def get_frac(default=0.3):
    print("\nRegression analysis will be performed on the identified correlation chains")
    print("Please specify fraction value used for LOESS smoothing (controls smoothing degree).")
    try:
        frac_input = input(f"Enter fraction value (e.g. 0.3 for 30%) [default={default}]: ").strip()
        if frac_input == '':
            print(f"No input provided. Using default: {default}")
            return default
        frac = float(frac_input)
        if not (0 < frac < 1):
            raise ValueError("Fraction must be between 0 and 1.")
        print(f"Using frac: {frac}")
        return frac
    except ValueError as e:
        print(f"Invalid input: {e}. Using default value: {default}")
        return default


def get_max_depth(default=5):
    print("Please specify max depth for CART (controls decision tree complexity).")
    try:
        max_depth_input = input(f"Enter max depth (positive integer) [default={default}]: ").strip()
        if max_depth_input == '':
            print(f"No input provided. Using default: {default}")
            return default
        max_depth = int(max_depth_input)
        if max_depth <= 0:
            raise ValueError("Max depth must be a positive integer.")
        print(f"Using max_depth: {max_depth}")
        return max_depth
    except ValueError as e:
        print(f"Invalid input: {e}. Using default value: {default}")
        return default

def get_min_samples_split(default=2):
    print("Please specify minimum samples needed to make a split (CART).")
    try:
        split_value = input(f"Enter min_samples_split [default={default}]: ").strip()
        if split_value == "":
            print(f"No input provided. Using default: {default}")
            return default
        split_value = int(split_value)
        if split_value < 2:
            raise ValueError("Must be ≥ 2.")
        print(f"Using min_samples_split: {split_value}")
        return split_value
    except ValueError as e:
        print(f"Invalid input: {e}. Using default: {default}")
        return default

def get_min_samples_leaf(default=1):
    print("Please specify minimum samples in a leaf (CART).")

    try:
        leaf_value = input(f"Enter min_samples_leaf [default={default}]: ").strip()
        if leaf_value == "":
            print(f"No input provided. Using default: {default}")
            return default
        leaf_value = int(leaf_value)
        if leaf_value <= 0:
            raise ValueError("Must be ≥ 1.")
        print(f"Using min_samples_leaf: {leaf_value}")
        return leaf_value
    except ValueError as e:
        print(f"Invalid input: {e}. Using default: {default}")
        return default


def get_svr_C(default=1.0):
    print("Please specify the regularization parameter C for SVR.")
    try:
        C_input = input(f"Enter C (positive number) [default={default}]: ").strip()
        if C_input == '':
            print(f"No input provided. Using default C: {default}")
            return default
        C_value = float(C_input)
        if C_value <= 0:
            raise ValueError("C must be a positive number.")
        print(f"Using C: {C_value}")
        return C_value
    except ValueError as e:
        print(f"Invalid input: {e}. Using default value: {default}")
        return default


def get_svr_epsilon(default=0.1):
    print("Please specify the epsilon parameter for SVR.")
    try:
        epsilon_input = input(f"Enter epsilon (positive number) [default={default}]: ").strip()
        if epsilon_input == '':
            print(f"No input provided. Using default epsilon: {default}")
            return default
        epsilon_value = float(epsilon_input)
        if epsilon_value <= 0:
            raise ValueError("Epsilon must be a positive number.")
        print(f"Using epsilon: {epsilon_value}")
        return epsilon_value
    except ValueError as e:
        print(f"Invalid input: {e}. Using default value: {default}")
        return default


def get_svr_gamma(default='scale'):
    print("Please specify the gamma parameter for SVR (with RBF kernel).")
    gamma_input = input(f"Enter gamma (positive number, 'auto' or 'scale') [default={default}]: ").strip()
    if gamma_input == '':
        print(f"No input provided. Using default gamma: {default}")
        return default
    if gamma_input.lower() in ['scale', 'auto']:
        print(f"Using gamma: {gamma_input.lower()}")
        return gamma_input.lower()
    try:
        gamma_value = float(gamma_input)
        if gamma_value <= 0:
            raise ValueError("Gamma must be a positive number, 'auto' or 'scale'.")
        print(f"Using gamma: {gamma_value}")
        return gamma_value
    except ValueError as e:
        print(f"Invalid input: {e}. Using default: {default}")
        return default

def get_plot_palette():
    palette_options = {
        1: 'meh',
        2: 'pastel-red',
        3: 'pastel-orange',
        4: 'bright'
    }
    default_key = 1
    prompt = (
        f"Available color palettes for the prediction error graph:\n"
        f"  1  default\n"
        f"  2  pastel-red\n"
        f"  3  pastel-orange\n"
        f"  4  bright\n"
        f"Enter a number to select a color palette from the list above [default={default_key}]: "
    )

    try:
        user_input = input(prompt).strip()
        key = int(user_input) if user_input else default_key
    except ValueError:
        key = default_key

    return palette_options.get(key, palette_options[default_key])
