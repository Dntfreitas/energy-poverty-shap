import os
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier, RUSBoostClassifier
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from utils import load_datasets


# Function to train the model and compute sensitivity metrics
def train_test_model(model, X_train, y_train, X_test, y_test, train_model=True):
    """
    Train the model and evaluate its performance on both training and test data.
    Calculates sensitivity for each class based on confusion matrices.

    Parameters:
        model: The machine learning model to train and evaluate.
        X_train: Training dataset features.
        y_train: Training dataset labels.
        X_test: Test dataset features.
        y_test: Test dataset labels.
        train_model: Boolean, whether to train the model.

    Returns:
        Trained model, sensitivity metrics for training data, and test data.
    """
    if train_model:
        model.fit(X_train, y_train)

    # Create the directories to store the models and results
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Compute confusion matrices for training and test sets
    confusion_matrix_train = pd.crosstab(y_train, model.predict(X_train), rownames=['Actual'], colnames=['Predicted'])
    confusion_matrix_test = pd.crosstab(y_test, model.predict(X_test), rownames=['Actual'], colnames=['Predicted'])

    # Calculate sensitivity (row-wise percentage) for each class
    sensitivity_train = (confusion_matrix_train.apply(lambda x: x / x.sum(), axis=1) * 100).round(2)
    sensitivity_test = (confusion_matrix_test.apply(lambda x: x / x.sum(), axis=1) * 100).round(2)

    return model, sensitivity_train, sensitivity_test


# Function to evaluate the model over a parameter grid
def evaluate_model(model, param_grid, scoring):
    """
    Evaluate the model for a range of hyperparameters using cross-validation and sensitivity metrics.

    Parameters:
        model: The base machine learning model to evaluate.
        param_grid: Dictionary defining the parameter grid for evaluation.
        scoring: List of metrics to evaluate during cross-validation.

    Saves:
        - Trained models as joblib files.
        - Results as a CSV file containing metrics and parameters.
    """
    parameters = ParameterGrid(param_grid)  # Generate all combinations of parameters in the grid
    results = []

    print(f'Start evaluating model: {model.estimator.__class__.__name__}')

    for param in tqdm(parameters):
        # Load the dataset with the specified window and threshold from the parameters
        X_train, y_train, X_test, y_test = load_datasets(window=param['window'], threshold=param['threshold'], contemporaneous=param['contemporaneous'])

        # Scale the features using RobustScaler to handle outliers
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        # Update model with current parameters
        parameters_estimator = {key: value for key, value in param.items() if 'estimator__' in key}
        model.set_params(**parameters_estimator)

        # Perform cross-validation to evaluate performance across folds
        scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)

        # Train the model and calculate sensitivity metrics on the test set
        _, _, sensitivity_test = train_test_model(model, X_train_scaled, y_train, X_test_scaled, y_test, train_model=True)

        # Extract sensitivity metrics for class 0 and 1, handle missing metrics
        sensitivity_test_0 = sensitivity_test.get(0, {}).get(0, np.nan)
        sensitivity_test_1 = sensitivity_test.get(1, {}).get(1, np.nan)

        # Save the trained model to a joblib file with a unique identifier
        uuid = str(uuid4())
        model_name = f'{model.estimator.__class__.__name__}_{uuid}'
        joblib.dump(model, f'models/{model_name}.joblib')

        # Save partial results (parameters and metrics)
        partial_results = {}

        for key, value in scores.items():
            for i, v in enumerate(value):
                partial_results[f'{key}_{i}'] = v
            partial_results[f'mean_{key}'] = np.mean(value)
            partial_results[f'std_{key}'] = np.std(value)

        # Add sensitivity metrics to results
        partial_results['sensitivity_0'] = sensitivity_test_0
        partial_results['sensitivity_1'] = sensitivity_test_1

        # Save parameter combination and model name
        partial_results['parameters'] = param
        partial_results['model_name'] = model_name

        results.append(partial_results)

    # Save the final results as a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results/{model.estimator.__class__.__name__}.csv', index=False)


# List of scoring metrics for cross-validation
scoring = [
    'recall',
    'precision',
    'accuracy',
    'balanced_accuracy',
    'roc_auc',
    'f1',
    'f1_macro',
    'f1_micro',
    'f1_weighted',
    'precision_macro',
    'precision_micro',
    'precision_weighted',
    'recall_macro',
    'recall_micro',
    'recall_weighted'
]

# Parameter settings
windows = [2, 4, 8, 12, 14]
thresholds = [0.1, 0.2, 0.4]

# List of models and their corresponding parameter grids
models_and_params = [
    (EasyEnsembleClassifier(random_state=42), {
        'estimator__n_estimators': [10, 50, 100],
        'estimator__sampling_strategy': ['auto', 0.5, 1.0],
        'estimator__replacement': [True, False],
        'window': windows,
        'threshold': thresholds,
        'contemporaneous': [False, True]
    }),
    (BalancedBaggingClassifier(random_state=42), {
        'estimator__n_estimators': [10, 50, 100],
        'estimator__max_samples': [0.5, 1.0],
        'estimator__max_features': [0.5, 1.0],
        'estimator__bootstrap': [True, False],
        'estimator__bootstrap_features': [True, False],
        'estimator__sampling_strategy': ['auto', 0.5, 1.0],
        'estimator__replacement': [True, False],
        'window': windows,
        'threshold': thresholds,
        'contemporaneous': [False, True]
    }),
    (RUSBoostClassifier(random_state=42), {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__learning_rate': [0.01, 0.1, 1.0],
        'estimator__algorithm': ['SAMME', 'SAMME.R'],
        'estimator__sampling_strategy': ['auto', 0.5, 1.0],
        'estimator__replacement': [True, False],
        'window': windows,
        'threshold': thresholds,
        'contemporaneous': [False, True]
    })
]


# Wrapper to initialize the model and evaluate it
def parallel_evaluate(model_cls, param_grid):
    model = OneVsRestClassifier(model_cls)
    evaluate_model(model, param_grid, scoring)


# Run the evaluations in parallel
if __name__ == "__main__":
    Parallel(n_jobs=len(models_and_params))(delayed(parallel_evaluate)(model, params) for model, params in models_and_params)
