import ast

import pandas as pd

# Load results from different classifier CSV files and tag each dataset with the respective model name
balancedBaggingClassifier = pd.read_csv('results/BalancedBaggingClassifier.csv')
balancedBaggingClassifier['model'] = 'balancedBaggingClassifier'

easyEnsembleClassifier = pd.read_csv('results/EasyEnsembleClassifier.csv')
easyEnsembleClassifier['model'] = 'easyEnsembleClassifier'

rUSBoostClassifier = pd.read_csv('results/RUSBoostClassifier.csv')
rUSBoostClassifier['model'] = 'RUSBoostClassifier'

# Combine all the classifier results into a single DataFrame for unified processing
results = pd.concat([balancedBaggingClassifier, easyEnsembleClassifier, rUSBoostClassifier])

# Extract 'threshold' and 'window' parameters from the 'parameters' column using regex
results['param_dict'] = results['parameters'].apply(ast.literal_eval)

for key in ['contemporaneous', 'threshold', 'window']:
    results[key] = results['param_dict'].apply(lambda d: d[key])

# Convert extracted parameters to appropriate numeric types
results['contemporaneous'] = results['contemporaneous'].astype(bool)
results['threshold'] = results['threshold'].astype(float)
results['window'] = results['window'].astype(int)

# Define grouping columns and the metrics of interest
groups_cols = ['model', 'threshold', 'window', 'contemporaneous']  # Grouping based on model, threshold, window, and contemporaneous
results_cols = [
    'mean_test_recall', 'std_test_recall', 'mean_test_precision', 'std_test_precision',
    'mean_test_accuracy', 'std_test_accuracy', 'mean_test_balanced_accuracy', 'std_test_balanced_accuracy',
    'mean_test_roc_auc', 'std_test_roc_auc', 'mean_test_f1', 'std_test_f1', 'mean_test_f1_macro',
    'std_test_f1_macro', 'mean_test_f1_micro', 'std_test_f1_micro', 'mean_test_f1_weighted',
    'std_test_f1_weighted', 'mean_test_precision_macro', 'std_test_precision_macro',
    'mean_test_precision_micro', 'std_test_precision_micro', 'mean_test_precision_weighted',
    'std_test_precision_weighted', 'mean_test_recall_macro', 'std_test_recall_macro',
    'mean_test_recall_micro', 'std_test_recall_micro', 'mean_test_recall_weighted',
    'std_test_recall_weighted', 'sensitivity_0', 'sensitivity_1'
]

# Save the results to a CSV file for future reference
results.to_csv('results/all_results.csv')

# Define the primary column to sort by (for selecting the best models)
sort_col = 'mean_test_balanced_accuracy'  # Sorting based on balanced accuracy (higher is better)

# Reindex the DataFrame and sort by the selected metric
results = results.reset_index(drop=True)
results.sort_values(by=sort_col, inplace=True)

# Identify the best models for each combination of model, threshold, and window
best_models = results.loc[results.groupby(groups_cols)[sort_col].idxmax().values]

# Further group and calculate maximum metrics for the best models
best_models = best_models.groupby(groups_cols).max()[results_cols + ['model_name']]

# Calculate and display mean and standard deviation of sensitivity for each class
mean_sensitivity_0 = best_models['sensitivity_0'].mean()
std_sensitivity_0 = best_models['sensitivity_0'].std()
mean_sensitivity_1 = best_models['sensitivity_1'].mean()
std_sensitivity_1 = best_models['sensitivity_1'].std()

# Save the DataFrame of best models to an Excel file for future reference
best_models.to_csv('results/best_models.csv')
