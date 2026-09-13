import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
mlflow.end_run()
dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target

print(X.head())
print("\nTarget Classes:", dataset.target_names)
print("Dataset Shape:", X.shape)

print("\nMissing Values:")
print(X.isnull().sum().sum())

X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.2,random_state=42)
print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlflow.set_experiment("Breast Cancer Classification")
with mlflow.start_run(run_name="Logistic Regression"):
    model = LogisticRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    confusion = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)

with open("confusion_matrix.txt", "w") as file:
    file.write(str(confusion))

with open("classification_report.txt", "w") as file:
    file.write(report)

    mlflow.log_artifact("confusion_matrix.txt")
    mlflow.log_artifact("classification_report.txt")

    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, name="model")

    print("\nLogistic Regression")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nConfusion Matrix:")
    print(confusion)
    print("\nClassification Report:")
    print(report)

mlflow.end_run()

with mlflow.start_run(run_name="Decision Tree"):
    tree_model = DecisionTreeClassifier(random_state=42)

    tree_model.fit(X_train, y_train)

    tree_predictions = tree_model.predict(X_test)

    tree_accuracy = accuracy_score(y_test, tree_predictions)
    tree_precision = precision_score(y_test, tree_predictions)
    tree_recall = recall_score(y_test, tree_predictions)
    tree_f1 = f1_score(y_test, tree_predictions)

    tree_confusion = confusion_matrix(y_test, tree_predictions)
    tree_report = classification_report(y_test, tree_predictions)

    with open("tree_confusion_matrix.txt", "w") as file:
        file.write(str(tree_confusion))

    with open("tree_classification_report.txt", "w") as file:
        file.write(tree_report)

    mlflow.log_param("model", "Decision Tree")
    mlflow.log_metric("accuracy", tree_accuracy)
    mlflow.log_metric("precision", tree_precision)
    mlflow.log_metric("recall", tree_recall)
    mlflow.log_metric("f1_score", tree_f1)

    mlflow.log_artifact("tree_confusion_matrix.txt")
    mlflow.log_artifact("tree_classification_report.txt")

    mlflow.sklearn.log_model(tree_model, name="model")

    print("\nDecision Tree")
    print("Accuracy:", tree_accuracy)
    print("Precision:", tree_precision)
    print("Recall:", tree_recall)
    print("F1 Score:", tree_f1)
    print("\nConfusion Matrix:")
    print(tree_confusion)
    print("\nClassification Report:")
    print(tree_report)

with mlflow.start_run(run_name="Random Forest"):
    forest_model = RandomForestClassifier(random_state=42)

    forest_model.fit(X_train, y_train)

    forest_predictions = forest_model.predict(X_test)

    forest_accuracy = accuracy_score(y_test, forest_predictions)
    forest_precision = precision_score(y_test, forest_predictions)
    forest_recall = recall_score(y_test, forest_predictions)
    forest_f1 = f1_score(y_test, forest_predictions)

    forest_confusion = confusion_matrix(y_test, forest_predictions)
    forest_report = classification_report(y_test, forest_predictions)

    with open("forest_confusion_matrix.txt", "w") as file:
        file.write(str(forest_confusion))

    with open("forest_classification_report.txt", "w") as file:
        file.write(forest_report)

    mlflow.log_param("model", "Random Forest")
    mlflow.log_metric("accuracy", forest_accuracy)
    mlflow.log_metric("precision", forest_precision)
    mlflow.log_metric("recall", forest_recall)
    mlflow.log_metric("f1_score", forest_f1)

    mlflow.log_artifact("forest_confusion_matrix.txt")
    mlflow.log_artifact("forest_classification_report.txt")

    mlflow.sklearn.log_model(forest_model, name="model")

    print("\nRandom Forest")
    print("Accuracy:", forest_accuracy)
    print("Precision:", forest_precision)
    print("Recall:", forest_recall)
    print("F1 Score:", forest_f1)
    print("\nConfusion Matrix:")
    print(forest_confusion)
    print("\nClassification Report:")
    print(forest_report)