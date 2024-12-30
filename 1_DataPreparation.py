import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

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

def process_min_year(min_year, data, user_id_col, output_dir):
    """
    Filter data for users with at least `min_year` consecutive years and process subsets.

    Parameters:
        min_year (int): Minimum number of consecutive years required.
        data (DataFrame): Input dataset.
        user_id_col (str): Column name for user IDs.
        output_dir (str): Directory to save processed data.

    Returns:
        None
    """

    print(f"Processing data for {min_year} consecutive years...")

    filtered_data = data.groupby(user_id_col).filter(
        lambda x: has_consecutive_years(x['wave'].tolist(), min_years=min_year))

    subsets = pd.DataFrame([])
    for user_id, group in filtered_data.groupby(user_id_col):
        waves = np.array(sorted(group['wave'].tolist()))[::-1]  # Sort waves in descending order
        wave_chunks = split_array(waves, min_year)  # Split waves into chunks
        for wave_chunk in wave_chunks:
            if len(wave_chunk) == min_year and is_continuous(wave_chunk):
                subset = group[group['wave'].isin(wave_chunk)].copy()
                lag_features = subset.columns.difference(['id', 'mepi', 'wave'])
                for feature in lag_features:
                    for lag in range(min_year - 1):
                        subset[f"{feature}_t{min_year - 1 - lag}"] = subset.groupby("id")[feature].shift(lag)
                subset.drop(columns=lag_features, inplace=True)
                last_mepi = subset['mepi'].iloc[-1]  # Get the final MEPI value
                subset = subset.iloc[-2:-1]  # Retain the penultimate row for analysis
                subset['mepi'] = last_mepi  # Assign the last MEPI value to the retained row
                subsets = pd.concat([subsets, subset], ignore_index=True)

    # Save the processed subsets to a CSV file
    file_path = f'{output_dir}/Hildabalin_{min_year}.csv'
    subsets.to_csv(file_path, index=False)
    print(f'File saved: {file_path}')


# Main processing logic

# List of minimum years to process
MIN_YEARS = [2, 4, 8, 12, 14]
# Directory to save output files
OUTPUT_DIR = "filtered_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each `min_year` concurrently using a thread pool
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_min_year, min_year, data, USER_ID_COL, OUTPUT_DIR) for min_year in MIN_YEARS]
    for future in as_completed(futures):
        try:
            future.result()  # Wait for each task to complete
        except Exception as e:
            print(f"Error occurred: {e}")
