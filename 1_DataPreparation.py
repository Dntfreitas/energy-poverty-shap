import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import has_consecutive_years, split_array, is_continuous

# Define constants for column and file path management
USER_ID_COL = 'id'  # Column to group data by user ID
PATH_PREFIX = ''  # Prefix for file paths

# Load and preprocess the dataset
data = pd.read_csv(f'{PATH_PREFIX}dataset/Hildabalin.csv')
data.sort_values(by=[USER_ID_COL, 'wave'], inplace=True)

# Drop unnecessary columns and handle missing data
data.drop(columns=['age2'], inplace=True)  # 'age2' is a derived column, not needed
data.dropna(inplace=True)  # Remove rows with missing values

# Feature engineering: Create new features for analysis

# 1. Age group categorization
bins = [0, 25, 35, 45, 55, 65, 100]  # Define age bins
labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']  # Labels for age groups
data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)

# 2. Create dummy variables for age groups
data = pd.get_dummies(data, columns=['age_group'], drop_first=True)

# 3. Calculate year-on-year changes for selected columns
data['GSPpc_growth_change'] = data.groupby(USER_ID_COL)['GSPpc'].pct_change().fillna(0)
data['unempratetot_change'] = data.groupby(USER_ID_COL)['unempratetot'].diff().fillna(0)
data['participratetot_change'] = data.groupby(USER_ID_COL)['participratetot'].diff().fillna(0)
data['energyprice_change'] = data.groupby(USER_ID_COL)['energyprice'].diff().fillna(0)
data['lnndincome_heu_change'] = data.groupby(USER_ID_COL)['lnndincome_heu'].diff().fillna(0)

# 4. Adjust `mepi` values for analysis
data['mepi'] = 1 - data['mepi']


# Define utility functions for processing data

def process_min_year(min_year, data, user_id_col, output_dir, include_contemporaneous=False):
    """
    Filter data for users with at least `min_year` consecutive years and process subsets.

    Parameters:
        min_year (int): Minimum number of consecutive years required.
        data (DataFrame): Input dataset.
        user_id_col (str): Column name for user IDs.
        output_dir (str): Directory to save processed data.
        include_contemporaneous (bool): Whether to include contemporaneous data in the analysis.

    Returns:
        None
    """

    # 1) Filter users with enough consecutive waves
    filtered = (
        data
        .groupby(user_id_col)
        .filter(lambda grp: has_consecutive_years(grp['wave'].tolist(), min_years=min_year))
    )

    result_subsets = []

    for user_id, grp in tqdm(filtered.groupby(user_id_col)):
        waves = sorted(grp['wave'].tolist(), reverse=True)
        for wave_chunk in split_array(np.array(waves), min_year):
            if len(wave_chunk) == min_year and is_continuous(wave_chunk):
                subset = grp[grp['wave'].isin(wave_chunk)].copy()
                lag_feats = subset.columns.difference([user_id_col, 'mepi', 'wave'])

                # 2) Create a DataFrame of all lagged columns at once
                lagged = {}
                n_lags = min_year if include_contemporaneous else (min_year - 1)
                for feat in lag_feats:
                    shifted = subset.groupby(user_id_col)[feat]
                    for lag in range(n_lags):
                        t_name = min_year - lag if include_contemporaneous else min_year - 1 - lag
                        col_name = f"{feat}_t{t_name}"
                        lagged[col_name] = shifted.shift(lag)
                lagged_df = pd.DataFrame(lagged, index=subset.index)

                # 3) Drop original features and concat the new lags
                subset = pd.concat([subset.drop(columns=lag_feats), lagged_df], axis=1)

                # 4) Handle contemporaneous vs. penultimate row
                if include_contemporaneous:
                    out = subset.iloc[[-1]]  # keep last
                else:
                    last_mepi = subset['mepi'].iat[-1]
                    out = subset.iloc[[-2]].copy()
                    out['mepi'] = last_mepi

                result_subsets.append(out)

    # Combine and write once
    all_out = pd.concat(result_subsets, ignore_index=True)
    fn = f"{output_dir}/Hildabalin_{min_year}.csv"
    all_out.to_csv(fn, index=False)
    print(f"File saved: {fn}")


# Main processing logic

# List of minimum years to process
MIN_YEARS = [2, 4, 8, 12, 14]
# Directory to save output files
OUTPUT_DIR = "filtered_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_DIR_CONTEMPORANEOUS = "filtered_data_contemporaneous"
os.makedirs(OUTPUT_DIR_CONTEMPORANEOUS, exist_ok=True)

# Process each `min_year` concurrently using a thread pool
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_min_year, min_year, data.copy(), USER_ID_COL, OUTPUT_DIR, False) for min_year in MIN_YEARS] + \
              [executor.submit(process_min_year, min_year, data.copy(), USER_ID_COL, OUTPUT_DIR_CONTEMPORANEOUS, True) for min_year in MIN_YEARS]
    for future in as_completed(futures):
        try:
            future.result()  # Wait for each task to complete
        except Exception as e:
            print(f"Error occurred: {e}")
