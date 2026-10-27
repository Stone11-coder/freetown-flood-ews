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
AT_USERNAME = os.getenv("AT_USERNAME", "sandbox")
AT_API_KEY =os.getenv(atsk_aace751a0d73f567317475f21cbbfd8b2779ffabc18951df93a0b9d81dc3acb365d654f9)

try:
    africastalking.initialize(AT_USERNAME, AT_API_KEY)
    sms_service = africastalking.SMS
except Exception as e:
    print(f"Warning: Failed to initialize Africa's Talking SDK: {e}")
    sms_service = None

# --- LOAD XGBOOST MODEL ---
MODEL_PATH = "models/freetown_xgboost_flood_model_v1.pkl"
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Successfully loaded Freetown XGBoost flood prediction model.")
    else:
        print(f"Warning: Model file {MODEL_PATH} not found on startup.")
except Exception as e:
    print(f"Error loading model: {e}")

# --- PYDANTIC SCHEMA MATCHING THE FULL FEATURE SET ---
class SpatialTerrainPayload(BaseModel):
    daily_rainfall_mm: float = Field(..., description="Daily rainfall in mm")
    rainfall_3day_mm: float = Field(..., description="3-day cumulative rainfall in mm")
    rainfall_7day_mm: float = Field(..., description="7-day cumulative rainfall in mm")
    rainfall_14day_mm: float = Field(..., description="14-day cumulative rainfall in mm")
    month_sin: float = Field(..., description="Cyclical month sine feature")
    month_cos: float = Field(..., description="Cyclical month cosine feature")
    elevation_mean: float = Field(..., description="Mean catchment elevation in meters")
    slope_mean: float = Field(..., description="Mean catchment slope angle")
    flowacc_mean: float = Field(..., description="Mean flow accumulation")
    slope_pct_over_15deg: float = Field(..., description="Percentage of slope over 15 degrees")
    phone_number: str = Field(..., description="Recipient phone number in E.164 format (e.g., +23278616275)")

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
    
    # Construct feature DataFrame matching the exact training columns
    input_data = pd.DataFrame([{
        "daily_rainfall_mm": payload.daily_rainfall_mm,
        "rainfall_3day_mm": payload.rainfall_3day_mm,
        "rainfall_7day_mm": payload.rainfall_7day_mm,
        "rainfall_14day_mm": payload.rainfall_14day_mm,
        "month_sin": payload.month_sin,
        "month_cos": payload.month_cos,
        "elevation_mean": payload.elevation_mean,
        "slope_mean": payload.slope_mean,
        "flowacc_mean": payload.flowacc_mean,
        "slope_pct_over_15deg": payload.slope_pct_over_15deg
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