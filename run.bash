# This file contains the commands to run the program. I.e., a single command to run the program, and generate the results.
echo "Preparing the data..."
python 1_DataPreparation.py
echo "Data preparation complete."
echo "Training the models..."
python 2_ModelTraining.py
echo "Model training complete."
echo "Evaluating the models..."
python 3_ModelsResults.py
echo "Model evaluation complete."
echo "Obtain SHAP values..."
python 4_ObtainSHAPsExplain.py
echo "SHAP values obtained."
echo "Generating the results tables..."
python 4_ResultsTables.py
echo "Results tables generated."
echo "Generating the figures..."
python 5_ResultsFigures.py
python 5_ResultsFiguresScatters.py
echo "Figures generated."
