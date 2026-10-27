import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import africastalking

# Initialize FastAPI application
app = FastAPI(
    title="Freetown Flood Early Warning System (EWS)",
    description="XGBoost geospatial flood risk prediction engine with automated SMS dispatch.",
    version="1.0.0"
)

# --- AFRICA'S TALKING CONFIGURATION ---
# Pulls from Render environment variables, defaulting to sandbox credentials for testing
AT_USERNAME = os.getenv("AT_USERNAME", "sandbox")
AT_API_KEY = os.getenv("AT_API_KEY", "L5bt87NPN")  # Replace or keep your generated key here

try:
    africastalking.initialize(AT_USERNAME, AT_API_KEY)
    sms_service = africastalking.SMS
except Exception as e:
    print(f"Warning: Failed to initialize Africa's Talking SDK: {e}")
    sms_service = None

# --- LOAD XGBOOST MODEL ---
# Ensure your trained model file is in the root directory
MODEL_PATH = "freetown_xgboost_flood_model.pkl" # Adjust filename if slightly different
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Successfully loaded Freetown XGBoost flood prediction model.")
    else:
        print(f"Warning: Model file {MODEL_PATH} not found on startup.")
except Exception as e:
    print(f"Error loading model: {e}")

# --- PYDANTIC SCHEMA ---
class SpatialTerrainPayload(BaseModel):
    elevation: float = Field(..., description="Terrain elevation in meters")
    slope: float = Field(..., description="Terrain slope angle")
    rainfall: float = Field(..., description="Accumulated rainfall in mm")
    flow_accumulation: float = Field(..., description="Upstream drainage area / flow accumulation")
    soil_moisture: float = Field(..., description="Volumetric soil moisture index")
    phone_number: str = Field(..., description="Recipient phone number in E.164 format (e.g., +23270000000)")

# --- ENDPOINTS ---
@app.get("/")
def home():
    return {
        "status": "online",
        "system": "Freetown Flood EWS",
        "threshold": 0.4859,
        "docs_url": "/docs"
    }

@app.post("/predict/spatial-terrain")
def predict_spatial_terrain(payload: SpatialTerrainPayload):
    if model is None:
        raise HTTPException(status_code=500, detail="Prediction model is not loaded on the server.")
    
    # Construct feature array matching training features
    input_data = pd.DataFrame([{
        "elevation": payload.elevation,
        "slope": payload.slope,
        "rainfall": payload.rainfall,
        "flow_accumulation": payload.flow_accumulation,
        "soil_moisture": payload.soil_moisture
    }])
    
    try:
        # Predict flood probability
        probabilities = model.predict_proba(input_data)
        flood_probability = float(probabilities[0][1])  # Probability of class 1 (flood risk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
    # Optimized Youden's J decision threshold
    THRESHOLD = 0.4859
    is_high_risk = flood_probability >= THRESHOLD
    
    sms_status = "Not triggered (Risk below threshold)"
    
    # Trigger automated SMS if high risk
    if is_high_risk:
        alert_message = (
            f"ALERT: High flood risk detected in your catchment area "
            f"(Probability: {flood_probability:.1%}). Please take precautionary measures."
        )
        try:
            if sms_service:
                response = sms_service.send(alert_message, [payload.phone_number])
                sms_status = f"SMS Dispatch Success: {response}"
                print(sms_status)
            else:
                sms_status = "SMS service uninitialized."
        except Exception as e:
            sms_status = f"Failed to send SMS via Africa's Talking: {str(e)}"
            print(sms_status)

    return {
        "flood_probability": flood_probability,
        "decision_threshold": THRESHOLD,
        "risk_status": "HIGH_RISK" if is_high_risk else "SAFE",
        "alert_triggered": is_high_risk,
        "sms_dispatch_result": sms_status
    }