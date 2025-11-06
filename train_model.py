import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib 

np.random.seed(42)
X = np.random.rand(100, 2) * 10
Y = 5 * X[:, 0] + 2 * X[:, 1] + 15 + np.random.randn(100)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

print(f"Model R-squared: {model.score(X_test, Y_test):.4f}")

filename = 'model.pkl'
joblib.dump(model, filename)

print(f"\n✅ Model successfully trained and saved as {filename}")

loaded_model = joblib.load(filename)
print(f"Test prediction using loaded model: {loaded_model.predict([[5.0, 3.0]])}")
