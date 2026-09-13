import pandas as pd
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target

print(X.head())
print("\nTarget Classes:", dataset.target_names)
print("Dataset Shape:", X.shape)

print("\nMissing Values:")
print(X.isnull().sum().sum())