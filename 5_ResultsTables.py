import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

from utils import load_datasets


def evaluate_model(model_name, threshold, window):
    """
    Evaluate a model on test data and calculate performance metrics.

    Parameters:
        model_name (str): Name of the model file to load.
        threshold (float): Threshold value for the dataset.
        window (int): Window size for the dataset.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    # Load the dataset with the specified parameters
    X_train, _, X_test, y_test = load_datasets(window=window, threshold=threshold)

    # Scale the features using RobustScaler
    scaler = RobustScaler()
    scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_test = y_test.values.ravel()

    # Load the pre-trained model
    model_path = f'models/{model_name}.joblib'
    model = joblib.load(model_path)

    # Generate predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate confusion matrix components
    TP = sum((y_test == 1) & (y_pred == 1))
    TN = sum((y_test == 0) & (y_pred == 0))
    FP = sum((y_test == 0) & (y_pred == 1))
    FN = sum((y_test == 1) & (y_pred == 0))

    # Compute sensitivity and specificity
    sensitivity = (TP / (TP + FN) * 100) if (TP + FN) > 0 else 0
    specificity = (TN / (TN + FP) * 100) if (TN + FP) > 0 else 0

    # Calculate additional metrics
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred) * 100
    roc_auc = roc_auc_score(y_test, y_pred) * 100

    return {
        'model': model_name,
        'threshold': threshold,
        'window': window,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'balanced_accuracy': balanced_accuracy
    }


def main():
    """
    Main function to evaluate models and save results in Markdown and Excel formats.
    """

    # Create the directory to store results
    os.makedirs('results/tables', exist_ok=True)

    # Load the list of best models
    models_data = pd.read_csv('results/best_models.csv')

    # Dictionary to store evaluation results
    models_results = {}

    # Evaluate each model
    for _, row in models_data.iterrows():
        model_name = row['model_name']
        threshold = row['threshold']
        window = row['window']
        models_results[model_name] = evaluate_model(model_name, threshold, window)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(models_results).T
    results_df.reset_index(inplace=True, names='model_name')
    results_df['model_name'] = results_df['model_name'].str.extract(r'(\w+)_')
    results_df.drop(columns='model', inplace=True)

    # Round numerical values to 2 decimal places
    results_df = results_df.applymap(lambda x: np.round(x, 2) if isinstance(x, (int, float)) else x)

    # Save results as a Markdown table
    markdown_table = results_df.to_markdown(index=False)
    with open('results/tables/TableC.3.md', 'w') as file:
        file.write(markdown_table)


if __name__ == "__main__":
    main()
