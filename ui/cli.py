def get_user_input_columns(columns):
    print("\nFeed backward construction of correlation chains in multidimensional datasets")
    print("Available attributes:", columns)

    while True:
        x = input("Select source attribute: ").strip()
        if x not in columns:
            print(f"'{x}' is not a valid attribute name. Please choose from the list.")
        else:
            break

    while True:
        y = input("Select target attribute: ").strip()
        if y not in columns:
            print(f"'{y}' is not a valid attribute name. Please choose from the list.")
        else:
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
            return default
        alpha = float(alpha_input)
        if not (0 <= alpha <= 0.3):
            raise ValueError("Alpha must be between 0 and 0.3")
        return alpha
    except ValueError as e:
        print(f"Invalid input: {e}. Using default value: {default}")
        return default


def get_path_finding_method():
    methods = {
        'greedy': 'greedy',
        'greedy+dfs': 'greedy+dfs',
        'dfs': 'dfs',
        'a*': 'a_star'
    }
    default = 'greedy'
    prompt = f"Choose path finding method ({', '.join(methods.keys())}) [default={default}]: "
    method = input(prompt).strip().lower().replace(' ', '')

    return methods.get(method, default)


def get_frac(default=0.3):
    print("Regression analysis will be performed.")
    print("Please specify fraction value used for LOESS smoothing (controls smoothing degree).")
    try:
        frac_input = input(f"Enter fraction value (e.g., 0.3 for 30%) [default={default}]: ").strip()
        if frac_input == '':
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
            return default
        max_depth = int(max_depth_input)
        if max_depth <= 0:
            raise ValueError("Max depth must be a positive integer.")
        print(f"Using max_depth: {max_depth}")
        return max_depth
    except ValueError as e:
        print(f"Invalid input: {e}. Using default value: {default}")
        return default

def get_min_samples_split(default=5):
    print("Please specify minimum samples needed to make a split (CART).")
    try:
        split_value = input(f"Enter min_samples_split [default={default}]: ").strip()
        if split_value == "":
            return default
        split_value = int(split_value)
        if split_value < 2:
            raise ValueError("Must be ≥ 2.")
        print(f"Using min_samples_split: {split_value}")
        return split_value
    except ValueError as e:
        print(f"Invalid input: {e}. Using default: {default}")
        return default

def get_min_samples_leaf(default=3):
    print("Please specify minimum samples in a leaf (CART).")

    try:
        leaf_value = input(f"Enter min_samples_leaf [default={default}]: ").strip()
        if leaf_value == "":
            return default
        leaf_value = int(leaf_value)
        if leaf_value <= 0:
            raise ValueError("Must be ≥ 1.")
        print(f"Using min_samples_leaf: {leaf_value}")
        return leaf_value
    except ValueError as e:
        print(f"Invalid input: {e}. Using default: {default}")
        return default


def get_svr_C(default=5.0):
    print("Please specify the regularization parameter C for SVR (controls model flexibility).")
    try:
        C_input = input(f"Enter C (positive number) [default={default}]: ").strip()
        if C_input == '':
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
    print("Please specify the epsilon parameter for SVR (defines the insensitive margin).")
    try:
        epsilon_input = input(f"Enter epsilon (positive number) [default={default}]: ").strip()
        if epsilon_input == '':
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
    gamma_input = input(f"Enter gamma for SVR [default={default}]: ").strip()
    if gamma_input == '':
        return default
    if gamma_input.lower() in ['scale', 'auto']:
        return gamma_input.lower()
    try:
        gamma_value = float(gamma_input)
        if gamma_value <= 0:
            raise ValueError("Gamma must be a positive number.")
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
