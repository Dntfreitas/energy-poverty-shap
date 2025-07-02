import os
from concurrent.futures import ProcessPoolExecutor

import joblib
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import RobustScaler

from utils import load_datasets


def obtain_shap(row):
    """
    Generates SHAP values for a model to explain its predictions.

    Parameters:
        row (dict): A dictionary containing:
            - 'window': Time window parameter for dataset preparation.
            - 'threshold': Threshold parameter for dataset filtering.
            - 'model_name': Name of the pre-trained model file to load.
            - 'sensitivity_0': Expected sensitivity for class 0 (for validation).
            - 'sensitivity_1': Expected sensitivity for class 1 (for validation).

    Steps:
        1. Check if SHAP values already exist for the model to avoid redundant computation.
        2. Load the dataset based on the 'window' and 'threshold' parameters.
        3. Scale features using RobustScaler to handle outliers effectively.
        4. Load the pre-trained model from the specified path.
        5. Validate model performance using the confusion matrix to ensure consistency.
        6. Use SHAP to compute feature importance and explain the model's predictions.
        7. Save the computed SHAP values to a file for future analysis.
    """

    # Extract model and dataset parameters
    window = row['window']
    threshold = row['threshold']
    model_name = row['model_name']
    contemporaneous = row['contemporaneous']  # Whether to use contemporaneous data
    sensitivity_0 = row['sensitivity_0']  # Expected sensitivity for class 0
    sensitivity_1 = row['sensitivity_1']  # Expected sensitivity for class 1

    # Check if SHAP values for this model already exist to avoid duplicate work
    shap_file = f"shap/shap_{model_name}"
    if os.path.exists(shap_file + ".pkl"):
        print(f"SHAP values already exist for {model_name}. Skipping...")
        return

    print(f"Computing SHAP values for {model_name}...")

    # Construct the path to the pre-trained model
    model_path = f'models/{model_name}.joblib'

    # Load datasets based on the parameters (training and testing splits)
    X_train, _, X_test, y_test = load_datasets(window=window, threshold=threshold, contemporaneous=contemporaneous)
    features = X_train.columns  # Extract feature names for SHAP visualization

    # Scale features to handle outliers using RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit the scaler to training data
    X_test_scaled = scaler.transform(X_test)  # Apply scaling to test data
    y_test = y_test.values.ravel()  # Flatten target variable array if necessary

    # Load the pre-trained model
    model = joblib.load(model_path)

    # Ensure the sensitivities match the expected values for model validation
    confusion_matrix_test = confusion_matrix(y_test, model.predict(X_test_scaled), normalize='true') * 100
    assert round(confusion_matrix_test[0][0], 2) == sensitivity_0, f"Sensitivity 0 mismatch: expected {sensitivity_0}, got {confusion_matrix_test[0][0]}"
    assert round(confusion_matrix_test[1][1], 2) == sensitivity_1, f"Sensitivity 1 mismatch: expected {sensitivity_1}, got {confusion_matrix_test[1][1]}"

    # Create a SHAP explainer for the model predictions
    explainer = shap.Explainer(model.predict, X_train_scaled, feature_names=features)
    shap_values = explainer(X_test_scaled, max_evals=2000)  # Limit evaluations for efficiency
    # Save SHAP values to a file for future use
    with open(shap_file + ".pkl", "wb") as f:
        joblib.dump(shap_values, f, protocol=5)  # Use protocol 5 for efficient serialization

    # Create a SHAP explainer for the model predictions
    explainer = shap.Explainer(model.predict_proba, X_train_scaled, feature_names=features)
    shap_values = explainer(X_test_scaled, max_evals=2000)  # Limit evaluations for efficiency
    # Save SHAP values to a file for future use
    with open(shap_file + "_proba.pkl", "wb") as f:
        joblib.dump(shap_values, f, protocol=5)  # Use protocol 5 for efficient serialization


# Load the DataFrame containing best model information
results = pd.read_csv('results/best_models.csv')

# Ensure the directory for saving SHAP values exists
shap_dir = 'shap'
os.makedirs(shap_dir, exist_ok=True)

# Parallelize SHAP value generation across available CPU cores
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Process each row in the results DataFrame in parallel
        executor.map(obtain_shap, [row for _, row in results.iterrows()])
