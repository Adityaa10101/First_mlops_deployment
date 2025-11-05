import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib # The library for saving/loading the model artifact

# --- 1. Create Dummy Data ---
# We are creating a dataset with 100 samples and 2 features.
# The target variable (Y) is based on a simple linear equation + some noise.
np.random.seed(42)
X = np.random.rand(100, 2) * 10 # 100 rows, 2 features (e.g., Area, Rooms)
Y = 5 * X[:, 0] + 2 * X[:, 1] + 15 + np.random.randn(100) # Target (e.g., Price)

# --- 2. Train the Model ---
# In a real project, you would spend weeks on this step. For MLOps, we just need a working model.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

# Quick sanity check: Check R-squared value
print(f"Model R-squared: {model.score(X_test, Y_test):.4f}")

# --- 3. Save the Model Artifact (Persistence) ---
# This is the crucial MLOps step! We save the model object to disk.
filename = 'model.pkl'
joblib.dump(model, filename)

print(f"\n✅ Model successfully trained and saved as {filename}")

# --- (Optional) Test Loading the Model ---
loaded_model = joblib.load(filename)
print(f"Test prediction using loaded model: {loaded_model.predict([[5.0, 3.0]])}")