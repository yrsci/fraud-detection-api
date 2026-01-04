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
from pydantic import BaseModel, validator
import pickle
from contextlib import asynccontextmanager
import os


# ----- Configuration -----

MODEL_PATH = "../models/fraud_model.pkl"
SCALER_PATH = "../models/amount_scaler.pkl"
DEFAULT_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5")) # lets me adjust the threshold without changing a value in the code
MODEL_VERSION = "v1.0"


# ----- Global state (load on startup) -----

model = None
scaler = None


# ----- Startup & Shutdown -----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler on startup, cleanup on shutdown"""
    global model, scaler
    
    model = pickle.load(open(MODEL_PATH, 'rb'))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    
    print(f"Model loaded: {MODEL_VERSION}")
    print(f"Threshold: {DEFAULT_THRESHOLD}")
    
    yield  # API runs here
    
    # Cleanup (if needed)
    print("Shutting down...")



# ----- Input & Output validation -----

class TransactionInput(BaseModel):
    """Input schema: 30 features (V1-V28 + Amount)"""
    
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
    
    @validator('Amount')
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
    model_version: str

    @validator('prediction')
    def binary_prediction(cls, v):
        if v not in [0, 1]:
            raise ValueError("Prediction must be binary (0 or 1)")
    
    @validator('probability')
    def valid_probability(cls, v):
        if v < 0:
            raise ValueError(f"Invalid probability value ({v}); probability must be between 0 and 1")
        if v > 1:
            raise ValueError(f"Invalid probability value ({v}); probability must be between 0 and 1")
        
