import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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