import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib import pyplot as plt

from utils import rename_variables, load_datasets, darken_color, get_feature_group

# Set the plotting style
sns.set(style='whitegrid')

# Set the font to plots to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Load the list of best models
models_data = pd.read_csv('results/best_models.csv')

# Extract model names, window sizes, and thresholds
shap_files = models_data['model_name'].tolist()
windows = models_data['window'].tolist()
thresholds = models_data['threshold'].tolist()
model = models_data['model'].tolist()

# Iterate over each model
for i in range(len(shap_files)):
    # Load the parameters
    shap_file = shap_files[i]
    window = windows[i]
    threshold = thresholds[i]
    model_name = model[i]

    # Load the SHAP values for the model
    explainer = joblib.load(f'shap/shap_{shap_file}_proba.pkl')[:, :, 1]

    # Update the variable names
    variable_names = explainer.feature_names

    # Compute the mean absolute SHAP value for each variable
    dict_variables = {}
    for i in range(len(variable_names)):
        shap_values = explainer.values
        dict_variables[variable_names[i]] = np.mean(np.abs(shap_values[:, i]))

    # Create a DataFrame for variable importance
    variable_importance = pd.DataFrame(dict_variables.items(), columns=['Variable', 'Importance'])

    # Assign groups to variables
    variable_importance['Group'] = variable_importance['Variable'].apply(lambda x: get_feature_group(x))
    # Remove the prefix _tX from the variable names
    variable_importance['VariableName'] = variable_importance['Variable'].apply(lambda x: x.split('_t')[0])

    # Normalize the importance values between 0 and 1
    scaler = sklearn.preprocessing.MinMaxScaler()
    variable_importance['Importance'] = scaler.fit_transform(variable_importance['Importance'].values.reshape(-1, 1))

    # Compute normalized importance percentages
    variable_importance['Normalized_Importance'] = variable_importance['Importance'] / variable_importance['Importance'].sum()

    # Compute group weights for normalization
    weight_A = variable_importance[variable_importance['Group'] == 'A']['Normalized_Importance'].sum()
    weight_B = variable_importance[variable_importance['Group'] == 'B']['Normalized_Importance'].sum()

    # Adjust normalized importance based on group weights
    variable_importance['Group_Weight'] = variable_importance['Group'].apply(lambda x: weight_A if x == 'A' else weight_B)
    variable_importance['Final_Importance'] = (variable_importance['Normalized_Importance'] / variable_importance['Group_Weight']) * 100

    # Rename variables based on the window
    variable_importance['Variable'] = variable_importance['Variable'].apply(lambda x: rename_variables([x], window)[0])

    # Ensure variable importance sums to 100% within groups
    variable_importance.groupby('Group')['Final_Importance'].sum()

    variable_importance['T'] = variable_importance.Variable.apply(lambda x: x.split('at T')[1].split('_')[0])

    # Drop feature from the group B
    variable_importance = variable_importance[variable_importance['Group'] == 'A'].copy()

    # Extrac the features to plot
    feature_importance_time_change = variable_importance.groupby('VariableName').sum().reset_index()[['VariableName', 'Final_Importance']]
    feature_importance_time_change = feature_importance_time_change.sort_values('Final_Importance', ascending=False)
    feature_importance_time_change = feature_importance_time_change[feature_importance_time_change['Final_Importance'] >= 1]
    features_to_plot = feature_importance_time_change.VariableName.tolist()[:5]

    ### PLOT 4
    # Load again the explainer, but we will change the feature space
    explainer_same_space = joblib.load(f'/Volumes/SSD—DNF/energy-poverty-after-conference-noextra/shap/shap_{shap_file}_proba.pkl')[:, :, 1]
    _, _, X_test, _ = load_datasets(window=window, threshold=threshold)

    for t in range(1, window):
        X_test[f'energyprice_t{t}'] = X_test[f'energyprice_t{t}'] * 0.001899
    explainer_same_space.data = X_test[explainer_same_space.feature_names]

    # Plot the new aggregated variable importance
    plt.figure()

    # Extract the feature names
    features_names = explainer_same_space.feature_names

    # Set the colors
    colors = sns.color_palette(n_colors=len(features_to_plot))
    i_color = 0

    for feature in features_to_plot:
        # Find all the features that contain the feature name
        feature_names = [feature + '_t' + str(i) for i in range(1, window)]
        # Let's create a scatter plot
        plt.figure()
        # Add a line to understand the trend
        X_data_all = explainer_same_space[:, feature_names].data.values.reshape(-1, 1).ravel()
        Y_values_all = explainer_same_space[:, feature_names].values.reshape(-1, 1).ravel()
        trend = np.polyfit(X_data_all, Y_values_all, deg=4)  # Linear fit
        trend_line = np.poly1d(trend)
        X_values = np.linspace(min(X_data_all), max(X_data_all), 1000)
        plt.plot(X_values, trend_line(X_values), color=darken_color(colors[i_color % len(colors)]), linestyle='--', linewidth=4)
        for feature_name in feature_names:
            # Extract data and values
            x_data = explainer_same_space[:, feature_name].data
            y_values = explainer_same_space[:, feature_name].values,
            plt.scatter(
                x_data,
                y_values,
                color=colors[i_color],
                alpha=0.2,
            )

        i_color += 1
        plt.xlabel(rename_variables([f'{feature}_t{window - 1}'], window)[0].split('at T')[0], )
        plt.ylabel('SHAP Value (higher indicates higher likelihood of energy poverty)', )
        plt.tight_layout()
        plt.savefig(f'results/figures/{model_name}/window_{window}_{threshold}_{feature}.png')
        plt.close()
