# MLOps Assignment 2 - MLflow Experiment Tracking

## Aim

To develop machine learning models and use MLflow to track and compare different experiments.

## Dataset

Breast Cancer Dataset from Scikit-learn.

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest

## Preprocessing

- Checked for missing values
- Split the dataset into 80% training and 20% testing data
- Applied StandardScaler to the features

## MLflow Tracking

The following information was logged for each experiment:

- Model parameters
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- Trained Model

## Best Model

Logistic Regression was selected as the best model based on the overall evaluation metrics.

The model was registered in MLflow as:

`BreastCancer_LogisticRegression`

## Project Structure

MLFlow_2/
- data/
- models/
- notebooks/
- screenshots/
- src/
- mlruns/
- README.md
- requirements.txt
- .gitignore

## Running the Project

Activate the virtual environment and run:

```bash
python src/train.py