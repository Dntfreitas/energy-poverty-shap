import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler

from utils import rename_variables, get_feature_group

# Set the plotting style
sns.set(style='whitegrid')

# Set the font to plots to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Load the list of best models
models_data = pd.read_csv('results/best_models.csv')

# Extract model names, window sizes, and thresholds
shap_files = models_data['model_name'].tolist()
windows = models_data['window'].tolist()
thresholds = models_data['threshold'].tolist()
model = models_data['model'].tolist()
contemporaneouss = models_data['contemporaneous'].tolist()

# Iterate over each model
for i in range(len(shap_files)):

    # Load the parameters
    shap_file = shap_files[i]
    window = windows[i]
    threshold = thresholds[i]
    model_name = model[i]
    countemporaneous = contemporaneouss[i]

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

    ##### PLOT 1
    # Plot the top variables by importance
    plt.figure()

    # Select variables with at least 2% final importance
    variable_importance_plot = variable_importance[variable_importance['Final_Importance'] >= 1]
    variable_importance_plot = variable_importance_plot.sort_values('Final_Importance', ascending=False)

    # Use a reversed 'coolwarm' color palette
    palette = sns.color_palette("coolwarm", n_colors=len(variable_importance_plot))[::-1]

    # Create the bar plot
    barplot = sns.barplot(
        x='Final_Importance',
        y='Variable',
        data=variable_importance_plot,
        palette=palette
    )

    # Add title and axis labels
    plt.xlabel('Normalized Importance (%)', )
    plt.ylabel('Variable', )

    # Annotate the bars with the exact values
    for i, (value, variable) in enumerate(zip(variable_importance_plot['Final_Importance'], variable_importance_plot['Variable'])):
        barplot.text(value + 0.5, i, f'{value:.2f}%', va='center', fontsize=12)

    # Finalize layout and save the figure
    plt.tight_layout()
    os.makedirs(f'results/figures/{model_name}', exist_ok=True)
    plt.savefig(f'results/figures/{model_name}/window_{window}_{threshold}_contemporaneous_{countemporaneous}.png')
    plt.savefig(f'results/figures/{model_name}/window_{window}_{threshold}_contemporaneous_{countemporaneous}.pgf')
    plt.close()

    ##### PLOT 2
    feature_importance_notime = variable_importance.groupby('VariableName').sum().reset_index()[['VariableName', 'Final_Importance']]

    # Sort and select the top variables
    feature_importance_notime = feature_importance_notime.sort_values('Final_Importance', ascending=False)
    feature_importance_notime = feature_importance_notime[feature_importance_notime['Final_Importance'] >= 1]

    # Update the feature name
    feature_importance_notime['VariableName'] = feature_importance_notime['VariableName'] + f'_t{window}'
    feature_importance_notime['VariableName'] = feature_importance_notime['VariableName'].apply(lambda x: rename_variables([x], window)[0])

    # Remove the "at T" from the variable names
    feature_importance_notime['VariableName'] = feature_importance_notime['VariableName'].apply(lambda x: x.split('at T')[0])

    # Plot the new aggregated variable importance
    plt.figure()

    # Use the same reversed 'coolwarm' color palette
    palette = sns.color_palette("coolwarm", n_colors=len(feature_importance_notime))[::-1]

    # Create the bar plot
    barplot = sns.barplot(
        x='Final_Importance',
        y='VariableName',
        data=feature_importance_notime,
        palette=palette
    )

    # Add title and axis labels
    plt.xlabel('Summed Normalized Importance (%)', )
    plt.ylabel('Variable', )

    # Annotate the bars with the exact values
    for i, (value, variable) in enumerate(zip(feature_importance_notime['Final_Importance'], feature_importance_notime['VariableName'])):
        barplot.text(value + 0.5, i, f'{value:.2f}%', va='center', fontsize=12)

    # Finalize layout and save the figure
    plt.tight_layout()
    plt.savefig(f'results/figures/{model_name}/window_{window}_{threshold}_contemporaneous_{countemporaneous}_notime.png')
    plt.savefig(f'results/figures/{model_name}/window_{window}_{threshold}_contemporaneous_{countemporaneous}_notime.pgf')
    plt.close()

    ### PLOT 3

    # Extrac the features to plot
    feature_importance_time_change = variable_importance.groupby('VariableName').sum().reset_index()[['VariableName', 'Final_Importance']]
    feature_importance_time_change = feature_importance_time_change.sort_values('Final_Importance', ascending=False)
    feature_importance_time_change = feature_importance_time_change[feature_importance_time_change['Final_Importance'] >= 1]
    features_to_plot = feature_importance_time_change.VariableName.tolist()[:5]

    # Filter the data for features to plot
    feature_time_importance = variable_importance[variable_importance['VariableName'].isin(features_to_plot)].copy()

    # Smooth the importance values using a rolling average
    feature_time_importance['Final_Importance'] = feature_time_importance.groupby('VariableName')['Final_Importance'].transform(
        lambda x: x.rolling(5, min_periods=1).mean())

    fig, ax1 = plt.subplots()

    # Lists to store legend handles and labels
    lines = []
    labels = []
    colors = sns.color_palette(n_colors=len(features_to_plot))
    i_color = 0

    for feature in features_to_plot:
        feature_data = feature_time_importance[feature_time_importance['VariableName'] == feature]
        feature_name = rename_variables([f'{feature}_t{window - 1}'], window)[0].split('at T')[0]
        if countemporaneous:
            mask = feature_data['T'] == ''
            feature_data.loc[mask, 'T'] = 0
        if feature == 'lnndincome_heu':
            ax2 = ax1.twinx()
            # Plot on secondary axis and capture line object
            line, = ax2.plot(feature_data['T'].astype(int), feature_data['Final_Importance'], label=feature_name, linewidth=2, color=colors[i_color])
            ax2.set_ylabel(f'Scale for {feature_name}', )
        else:
            # Plot on primary axis and capture line object
            line, = ax1.plot(feature_data['T'].astype(int), feature_data['Final_Importance'], label=feature_name, linewidth=2, color=colors[i_color])

        # Append line object and label to lists
        lines.append(line)
        labels.append(feature_name)
        # Update color index
        i_color += 1

    # Setting common labels
    ax1.set_ylabel('Smoothed Importance (%)', )
    # Extrac the features to plot
    feature_importance_time_change = variable_importance.groupby('VariableName').sum().reset_index()[['VariableName', 'Final_Importance']]
    feature_importance_time_change = feature_importance_time_change.sort_values('Final_Importance', ascending=False)
    feature_importance_time_change = feature_importance_time_change[feature_importance_time_change['Final_Importance'] >= 1]
    features_to_plot = feature_importance_time_change.VariableName.tolist()[:5]

    # Filter the data for features to plot
    feature_time_importance = variable_importance[variable_importance['VariableName'].isin(features_to_plot)].copy()

    # Smooth the importance values using a rolling average
    feature_time_importance['Final_Importance'] = feature_time_importance.groupby('VariableName')['Final_Importance'].transform(
        lambda x: x.rolling(5, min_periods=1).mean())

    fig, ax1 = plt.subplots()

    # Lists to store legend handles and labels
    lines = []
    labels = []
    colors = sns.color_palette(n_colors=len(features_to_plot))
    i_color = 0

    for feature in features_to_plot:
        feature_data = feature_time_importance[feature_time_importance['VariableName'] == feature]
        feature_name = rename_variables([f'{feature}_t{window - 1}'], window)[0].split('at T')[0]
        if countemporaneous:
            mask = feature_data['T'] == ''
            feature_data.loc[mask, 'T'] = 0
        if feature == 'lnndincome_heu':
            ax2 = ax1.twinx()
            # Plot on secondary axis and capture line object
            line, = ax2.plot(feature_data['T'].astype(int), feature_data['Final_Importance'], label=feature_name, linewidth=2, color=colors[i_color])
            ax2.set_ylabel(f'Scale for {feature_name}', )
        else:
            # Plot on primary axis and capture line object
            line, = ax1.plot(feature_data['T'].astype(int), feature_data['Final_Importance'], label=feature_name, linewidth=2, color=colors[i_color])

        # Append line object and label to lists
        lines.append(line)
        labels.append(feature_name)
        # Update color index
        i_color += 1

    # Setting common labels
    ax1.set_ylabel('Smoothed Importance (%)', )

    # Adjust x-axis ticks
    ax1.set_xticks(feature_data['T'].astype(int).unique())  # Ensure ticks are at integer positions of 'T'
    ax1.set_xticklabels([f'T{int(t)}' for t in feature_data['T'].astype(int).unique()])  # Label as T-n

    # Create unified legend
    ax1.legend(lines, labels, fontsize=12)

    plt.tight_layout()

    plt.savefig(f'results/figures/{model_name}/window_{window}_{threshold}_contemporaneous_{countemporaneous}_temporal.png')
    plt.savefig(f'results/figures/{model_name}/window_{window}_{threshold}_contemporaneous_{countemporaneous}_temporal.pgf')
    plt.close()
