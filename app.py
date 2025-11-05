from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os # We'll use this later to find the model file

# --- 1. Load the Model ONCE (Crucial MLOps step for efficiency) ---
MODEL_PATH = 'model.pkl'

# Check if the model file exists before trying to load it
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Did you run train_model.py?")

model = joblib.load(MODEL_PATH)
print("✅ ML Model loaded successfully.")

# --- 2. Define the Pydantic Schema for Input Validation ---
# This ensures that all incoming data is structured and typed correctly.
class FeatureRequest(BaseModel):
    # These names (feature_1, feature_2) must match what your model expects.
    feature_1: float
    feature_2: float

# --- 3. Initialize the FastAPI Application ---
app = FastAPI(title="E-commerce Price Predictor")

# --- 4. Define the API Prediction Endpoint ---
@app.post("/predict")
def predict_price(request: FeatureRequest):
    """
    Takes two numerical features and returns a predicted price.
    """
    try:
        # FastAPI/Pydantic ensures the input data is safe and validated.
        # Convert the Pydantic model data into the NumPy array the model expects.
        data_point = [[request.feature_1, request.feature_2]]

        # Get the prediction from the loaded scikit-learn model.
        prediction = model.predict(data_point)[0]

        # Return the result as a simple JSON object.
        return {
            "prediction": round(prediction, 2),
            "status": "success"
        }

    except Exception as e:
        # Error handling is key in production
        return {"error": str(e), "status": "failure"}

# --- 5. Optional Health Check (Good practice) ---
@app.get("/")
def health_check():
    return {"status": "ok", "message": "API is running and model is loaded."}