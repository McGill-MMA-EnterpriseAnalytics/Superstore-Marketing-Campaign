from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# Load the trained model
model_path = "models/best_model_tuned.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = joblib.load(model_path)

# Create FastAPI app
app = FastAPI(
    title="Superstore Marketing Campaign Prediction API",
    description="API for predicting marketing campaign response",
    version="1.0.0"
)

# Define input schema
class PredictionInput(BaseModel):
    features: Dict[str, Any]

# Define output schema
class PredictionOutput(BaseModel):
    prediction: int
    probability: float

@app.get("/")
def read_root():
    return {"message": "Marketing Campaign Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Convert input dictionary to DataFrame
        df = pd.DataFrame([input_data.features])
        
        # Handle categorical features if model requires it
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            for col in object_cols:
                df[col] = df[col].astype('category').cat.codes
                
        # Make prediction
        prediction = int(model.predict(df)[0])
        
        # Get probability
        probability = float(model.predict_proba(df)[0][1])
        
        return PredictionOutput(prediction=prediction, probability=probability)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch")
def predict_batch(input_data: List[Dict[str, Any]]):
    try:
        # Convert input list to DataFrame
        df = pd.DataFrame(input_data)
        
        # Handle categorical features if model requires it
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            for col in object_cols:
                df[col] = df[col].astype('category').cat.codes
                
        # Make predictions
        predictions = model.predict(df).tolist()
        probabilities = model.predict_proba(df)[:, 1].tolist()
        
        return [
            {"prediction": int(pred), "probability": float(prob)} 
            for pred, prob in zip(predictions, probabilities)
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}") 