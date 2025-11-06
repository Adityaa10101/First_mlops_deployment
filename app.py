from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os 

# Loading the Model In
MODEL_PATH = 'model.pkl'

# Checking if the model file exists before trying to load it
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Did you run train_model.py?")

model = joblib.load(MODEL_PATH)
print("ML Model loaded successfully.")

class FeatureRequest(BaseModel):
    feature_1: float
    feature_2: float

app = FastAPI(title="E-commerce Price Predictor")

@app.post("/predict")
def predict_price(request: FeatureRequest):
    """
    Takes two numerical features and returns a predicted price.
    """
    try:
        data_point = [[request.feature_1, request.feature_2]]

        prediction = model.predict(data_point)[0]

        return {
            "prediction": round(prediction, 2),
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e), "status": "failure"}
#health check
@app.get("/")
def health_check():
    return {"status": "ok", "message": "API is running and model is loaded."}
