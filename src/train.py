import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

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

model = LogisticRegression()
model.fit(X_train, y_train)
mypredictions = model.predict(X_test)

accuracy = accuracy_score(y_test, mypredictions)
precision = precision_score(y_test, mypredictions)
recall = recall_score(y_test, mypredictions)
f1 = f1_score(y_test, mypredictions)

confusion = confusion_matrix(y_test, mypredictions)

print("\nLogistic Regression")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nConfusion Matrix:")
print(confusion)
print("\nClassification Report:")
print(classification_report(y_test, mypredictions))

tree_model = DecisionTreeClassifier(random_state=42)

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_predictions)
tree_precision = precision_score(y_test, tree_predictions)
tree_recall = recall_score(y_test, tree_predictions)
tree_f1 = f1_score(y_test, tree_predictions)

tree_confusion = confusion_matrix(y_test, tree_predictions)

print("\nDecision Tree")
print("Accuracy:", tree_accuracy)
print("Precision:", tree_precision)
print("Recall:", tree_recall)
print("F1 Score:", tree_f1)
print("\nConfusion Matrix:")
print(tree_confusion)
print("\nClassification Report:")
print(classification_report(y_test, tree_predictions))