import matplotlib.colors as mcolors
import pandas as pd
from sklearn.model_selection import train_test_split


def load_datasets(window, threshold):
    """
    Load and preprocess the dataset for a given time window and MEPI threshold.

    Parameters:
        window (int): The size of the time window (e.g., number of consecutive years).
        threshold (float): The threshold to binarize the MEPI column.

    Returns:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels (binary MEPI).
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels (binary MEPI).
    """
    # Load the dataset for the specified window size
    data = pd.read_csv(f'filtered_data/Hildabalin_{window}.csv')
    # Ensure data is sorted by user ID and wave for consistency in processing
    data.sort_values(by=['id', 'wave'], inplace=True)

    # Binarize the 'mepi' column based on the given threshold
    # If the MEPI value is greater than or equal to the threshold, set it to 1 (high risk)
    # Otherwise, set it to 0 (low risk)
    data['mepi'] = data['mepi'].apply(lambda x: 1 if x >= threshold else 0)

    # Split the data into training and testing sets based on unique user IDs
    user_ids = data['id'].unique()  # Extract unique user IDs
    train_ids, test_ids = train_test_split(user_ids, test_size=0.2, random_state=42)  # 80-20 split

    # Prepare the training dataset
    X_train = data[data['id'].isin(train_ids)].copy()  # Select rows where user ID is in the training set
    y_train = X_train['mepi']  # Extract the target variable (binary MEPI)
    X_train = X_train.drop(columns=['id', 'mepi', 'wave'])  # Drop unnecessary columns from features

    # Prepare the testing dataset
    X_test = data[data['id'].isin(test_ids)].copy()  # Select rows where user ID is in the test set
    y_test = X_test['mepi']  # Extract the target variable (binary MEPI)
    X_test = X_test.drop(columns=['id', 'mepi', 'wave'])  # Drop unnecessary columns from features

    # Return the processed training and testing datasets
    return X_train, y_train, X_test, y_test


def has_consecutive_years(waves, min_years):
    """
    Check if a list of years contains at least a specified number of consecutive years.

    Parameters:
        waves (list): List of years.
        min_years (int): Minimum number of consecutive years required.

    Returns:
        bool: True if the condition is met, False otherwise.
    """
    consecutive_count = 1
    max_consecutive = 0
    for i in range(1, len(waves)):
        if waves[i] == waves[i - 1] + 1:
            consecutive_count += 1
        else:
            max_consecutive = max(max_consecutive, consecutive_count)
            consecutive_count = 1
    max_consecutive = max(max_consecutive, consecutive_count)
    return max_consecutive >= min_years


def split_array(arr, chunk_size):
    """
    Split an array into chunks of a specified size.

    Parameters:
        arr (list or array): The array to split.
        chunk_size (int): The size of each chunk.

    Returns:
        list: List of array chunks.
    """
    return [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]


def is_continuous(arr):
    """
    Check if an array contains consecutive numbers.

    Parameters:
        arr (list or array): The array to check.

    Returns:
        bool: True if the array contains consecutive numbers, False otherwise.
    """
    return all(x == y + 1 for x, y in zip(arr, arr[1:]))


def rename_variables(variable_names, window_size):
    """
    Renames a list of variables based on specified naming conventions and a window size.

    Args:
        variable_names (list): List of original variable names.
        window_size (int): The total number of time steps to calculate T, T-1, T-2, etc.

    Returns:
        list: List of renamed variables based on the specified rules.
    """
    # Define variable patterns and their corresponding base names
    patterns = {
        "lnndincome_heu_t": "Household Income",
        "employed_t": "Employment Status",
        "energyprice_t": "Energy Price",
        "lnndincome_heu_change_t": "Household Income Change",
        "hhsizeequiv_t": "Household Size",
        "parttimerate_t": "Part-Time Employment Rate",
        "partipatotrate_t": "Labor Force Participation Rate",
        "childrendic_t": "Children Presence at Household",
        "energy_cost_burden_t": "Energy Cost Burden",
        "education_employment_interaction_t": "Education-Employment Interaction",
        "married_income_interaction_t": "Married-Income Interaction",
        "GSPpc_t": "GSP Per Capita",
        "GSPpc_growth_t": "GSP Per Capita Growth",
        "GSPpcgrowth_t": "GSP Per Capita Growth",
        "age_t": "Age",
        "age_group_26-35_t": "Age Group 26-35",
        "age_group_36-45_t": "Age Group 36-45",
        "age_group_46-55_t": "Age Group 46-55",
        "age_group_56-65_t": "Age Group 56-65",
        "age_group_65+_t": "Age Group 65+",
        "age_income_interaction_t": "Age-Income Interaction",
        "disability_t": "Poor Health",
        "divorced_t": "Divorced Status",
        "dum_hhstate2_t": "Household Region 2",
        "dum_hhstate3_t": "Household Region 3",
        "dum_hhstate4_t": "Household Region 4",
        "dum_hhstate5_t": "Household Region 5",
        "dum_hhstate6_t": "Household Region 6",
        "dum_hhstate7_t": "Household Region 7",
        "dum_hhstate8_t": "Household Region 8",
        "dum_rem2_t": "Place of Residence 2",
        "dum_rem3_t": "Place of Residence 3",
        "dum_rem4_t": "Place of Residence 4",
        "dum_rem5_t": "Place of Residence 5",
        "economic_growth_income_interaction_t": "GSP Growth-Income Interaction",
        "educated_unemployed_t": "Years of Education-Unemployed Interaction",
        "energyprice_change_t": "Energy Price Change",
        "financial_vulnerability_t": "Financial Vulnerability",
        "income_per_capita_t": "Income Per Capita",
        "ln_yearseduc_t": "Years of Education",
        "married_t": "Married Status",
        "participratetot_t": "Total Labor Force Participation Rate",
        "participratetot_change_t": "Labor Force Participation Rate Change",
        "unemp_t": "Unemployment Status",
        "unemployment_burden_t": "Unemployment Burden",
        "unempratetot_t": "Unemployment Rate",
        "unempratetot_change_t": "Unemployment Rate Change",
        "widowed_t": "Widowed Status",
        "GSPpc_growth_change_t": "GSP Per Capita Growth Change",
    }

    renamed = []

    for var in variable_names:
        matched = False
        for pattern, base_name in patterns.items():
            if var.startswith(pattern):
                t = int(var.split("_t")[-1])  # Extract time step
                if t == window_size:
                    renamed.append(f"{base_name} at T")
                else:
                    renamed.append(f"{base_name} at T-{window_size - t}")
                matched = True
                break

        if not matched:
            raise ValueError(f"Variable {var} not recognized.")

    return renamed


def get_group_a_features():
    return ['age_group_26-35', 'age_group_36-45', 'age_group_46-55', 'age_group_56-65', 'age_group_65+', 'childrendic', 'disability', 'divorced', 'employed', 'energyprice', 'GSPpc', 'GSPpcgrowth', 'hhsizeequiv',
            'ln_yearseduc', 'lnndincome_heu', 'lnndincome_heu_change', 'married', 'participratetot', 'parttimerate', 'unemp', 'unempratetot', 'widowed']


def get_group_b_features():
    return ['age', 'dum_hhstate2', 'dum_hhstate3', 'dum_hhstate4', 'dum_hhstate5', 'dum_hhstate6', 'dum_hhstate7', 'dum_hhstate8', 'dum_rem2', 'dum_rem3', 'dum_rem4', 'dum_rem5', 'energyprice_change', 'GSPpc_growth_change',
            'participratetot_change', 'partipatotrate', 'unempratetot_change']


def is_group_feature(feature, group):
    # remote the _txx part
    feature = feature.split('_t')[0]
    if group == 'A':
        return feature in get_group_a_features()
    elif group == 'B':
        return feature in get_group_b_features()
    else:
        raise ValueError(f"Group {group} not recognized.")


def get_feature_group(feature):
    # remote the _txx part
    feature = feature.split('_t')[0]
    if feature in get_group_a_features():
        return 'A'
    elif feature in get_group_b_features():
        return 'B'
    else:
        raise ValueError(f"Feature {feature} not recognized.")


def darken_color(color, factor=0.7):
    """
    Darkens a given color by a specified factor.
    :param color: Original color in any Matplotlib-compatible format.
    :param factor: Factor by which to darken the color (0-1, where 1 is the original color).
    :return: Darkened color.
    """
    rgb = mcolors.to_rgb(color)  # Convert color to RGB
    dark_rgb = tuple(c * factor for c in rgb)  # Scale each channel
    return dark_rgb
