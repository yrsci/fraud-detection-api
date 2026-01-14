"""
API Design Decisions:

1. Preprocessing: API scales Amount internally using saved scaler
2. Input validation: Pydantic model enforces 30 features
3. Response format: prediction + probability + threshold + version
4. Error handling: Return 422 for validation errors with details
5. Model loading: On startup, not per-request
6. Threshold: Default 0.5, configurable via environment variable
"""

from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import numpy as np
import pandas as pd
import pickle
from contextlib import asynccontextmanager
import os
import uvicorn
from pathlib import Path


# ----- Configuration -----

BASE_DIR = Path(__file__).parent.parent
FRAUD_MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "amount_scaler.pkl"
DEFAULT_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5")) # lets me adjust the threshold without changing a value in the code
FRAUD_MODEL_VERSION = "v1.0"


# ----- Startup & Shutdown -----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler on startup, cleanup on shutdown"""
    global model, scaler
    with open(FRAUD_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Model loaded: {FRAUD_MODEL_VERSION}")
    print(f"Threshold: {DEFAULT_THRESHOLD}")
    yield  # API runs here
    print("Shutting down...")


# ----- Input & Output validation -----

class TransactionInput(BaseModel):
    """Input schema: 29 features (V1-V28 + Amount)"""
    
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    
    @field_validator("Amount", mode="before")
    def amount_must_be_positive(cls, v):
        """Validate Amount is non-negative"""
        if v < 0:
            raise ValueError("Amount must be non-negative")
            # TODO: Consider logging warning instead of rejecting negative input
        return v
    
class PredictionResponse(BaseModel):
    """Output schema: prediction + metadata"""
    
    prediction: int  # 0 or 1
    probability: float  # 0.0 to 1.0
    threshold: float
    fraud_model_version: str
        

# ----- Initialise the API -----

app = FastAPI(
    title="Fraud Detection API",
    description="Credit card fraud detection using Logistic Regression",
    version=FRAUD_MODEL_VERSION,
    lifespan=lifespan
)


# ----- Endpoints -----

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "fraud_model_version": FRAUD_MODEL_VERSION,
        "threshold": DEFAULT_THRESHOLD
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionInput):
    """
    Predict fraud probability for a transaction
    
    Returns:
        prediction: 0 (legitimate) or 1 (fraud)
        probability: fraud probability (0.0 to 1.0)
        threshold: classification threshold used
        fraud_model_version: model version identifier
    """

    # Rescale Amount
    amount_df = pd.DataFrame([[transaction.Amount]], columns=["Amount"])
    scaled_amount = scaler.transform(amount_df)[0][0]

    # Prepare featureset
    features = [getattr(transaction, f"V{i}") for i in range(1, 29)]
    features.append(scaled_amount)

    # Convert to numpy array with feature names
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled"]
    features_df = pd.DataFrame([features], columns=feature_names)

    # Run model to compute probability of fraud
    fraud_probability = model.predict_proba(features_df)[0][1]
    prediction = 1 if fraud_probability >= DEFAULT_THRESHOLD else 0

    return {
        "prediction": prediction,
        "probability": fraud_probability,
        "threshold": DEFAULT_THRESHOLD,
        "fraud_model_version": FRAUD_MODEL_VERSION,
    }


# ----- Start the server -----

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)