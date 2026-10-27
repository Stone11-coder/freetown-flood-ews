import os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import africastalking

app = FastAPI(
    title="Freetown Flood Early Warning System (EWS) - XGBoost Engine",
    description="Real-time flash flood risk scoring engine powered by XGBoost, spatial terrain integration, and automated SMS dispatch.",
    version="2.1.0"
)

# Model path
MODEL_PATH = Path("models/freetown_xgboost_flood_model_v1.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            print("Successfully loaded Freetown XGBoost flood prediction model.")
        except Exception as e:
            print(f"Error loading model artifact: {e}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Running in diagnostic mode.")

# Initialize Africa's Talking (pulls credentials securely from Render environment variables)
AT_USERNAME = os.getenv("AT_USERNAME", "sandbox")
AT_API_KEY = os.getenv("AT_API_KEY", "your_api_key")
africastalking.initialize(AT_USERNAME, AT_API_KEY)
sms = africastalking.SMS

# 1. UPDATED INPUT SCHEMA: Includes the phone number alongside your 10 terrain/weather features
class SpatialTerrainInput(BaseModel):
    daily_rainfall_mm: float = Field(..., description="Rainfall recorded in the last 24 hours (mm)")
    rainfall_3day_mm: float = Field(..., description="Cumulative rainfall over the past 3 days (mm)")
    rainfall_7day_mm: float = Field(..., description="Cumulative rainfall over the past 7 days (mm)")
    rainfall_14day_mm: float = Field(..., description="Cumulative rainfall over the past 14 days (mm)")
    month_sin: float = Field(..., description="Cyclical sine encoding for seasonality (month)")
    month_cos: float = Field(..., description="Cyclical cosine encoding for seasonality (month)")
    elevation_mean: float = Field(..., description="Mean elevation of the catchment (m)")
    slope_mean: float = Field(..., description="Mean terrain slope (degrees)")
    flowacc_mean: float = Field(..., description="Mean upslope contributing flow accumulation area")
    slope_pct_over_15deg: float = Field(..., description="Percentage of catchment slope over 15 degrees")
    phone_number: str = Field(..., description="Recipient phone number for emergency alerts (e.g., +23270000000)")

class FloodPredictionResponse(BaseModel):
    flood_probability: float
    alert_triggered: bool
    threshold_used: float
    risk_level: str

@app.get("/")
def health_check():
    return {
        "status": "active", 
        "system": "Freetown Flood EWS XGBoost Engine",
        "model_loaded": model is not None
    }

@app.post("/predict/spatial-terrain", response_model=FloodPredictionResponse)
def predict_spatial_terrain(data: SpatialTerrainInput):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Prediction model is not loaded. Ensure the XGBoost .pkl model exists in the models/ directory."
        )
    
    # Convert incoming request to DataFrame, excluding the phone_number 
    # (since XGBoost only needs the 10 numerical terrain/weather features)
    input_data = data.dict()
    phone = input_data.pop("phone_number")  # Remove phone number from model input
    
    input_df = pd.DataFrame([input_data])
    
    # Inference using XGBoost with your optimized Youden's J threshold
    probability = float(model.predict_proba(input_df)[:, 1][0])
    threshold = 0.4859  
    alert = probability >= threshold
    
    # Risk categorization
    if probability >= 0.98:
        risk = "SEVERE"
    elif probability >= threshold:
        risk = "WARNING"
    else:
        risk = "NORMAL"
        
    # --- AUTOMATED SMS DISPATCH WORKFLOW ---
    if alert and phone:
        alert_message = (
            f"🚨 FREETOWN FLOOD {risk} ALERT! "
            f"Risk Probability: {probability:.1%}. "
            f"Heavy rainfall detected in your catchment area. "
            f"Please move to higher ground immediately."
        )
        try:
            response = sms.send(alert_message, [phone])
            print(f"SMS Dispatch Success: {response}")
        except Exception as e:
            print(f"Failed to send SMS via Africa's Talking: {e}")

    return {
        "flood_probability": round(probability, 4),
        "alert_triggered": bool(alert),
        "threshold_used": threshold,
        "risk_level": risk
    }