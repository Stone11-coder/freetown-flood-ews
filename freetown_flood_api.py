import os
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import africastalking

# --- INITIALIZATION ---
app = FastAPI(title="Freetown Flood EWS API", version="2.1")

# --- AFRICA'S TALKING CONFIGURATION ---
AT_USERNAME = "sandbox"
AT_API_KEY = "atsk_97cff8882e20add5d931f28c8188bc5808fa3ed26c14cd41e76f7d5ded63b6a74e5c1e72"
try:
    africastalking.initialize(AT_USERNAME, AT_API_KEY)
    sms_service = africastalking.SMS
except Exception as e:
    print(f"Warning: Failed to initialize Africa's Talking SDK: {e}")
    sms_service = None

# --- LOAD TRAINED XGBOOST MODEL ---
MODEL_PATH = "models/freetown_xgboost_flood_model_v1.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("XGBoost flood model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_PATH}: {e}")
    model = None

# --- EMBEDDED WEATHER INGESTION LOGIC ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
FREETOWN_LAT = 8.4844
FREETOWN_LON = -13.2344

def fetch_freetown_weather_data() -> dict:
    """
    Fetches real-time weather and calculates cumulative rainfalls for Freetown.
    Falls back to safe baseline defaults if the API key is missing or request fails.
    """
    if not OPENWEATHER_API_KEY:
        return {
            "daily_rainfall_mm": 12.5,
            "rainfall_3day_mm": 35.0,
            "rainfall_7day_mm": 80.0,
            "rainfall_14day_mm": 150.0
        }
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={FREETOWN_LAT}&lon={FREETOWN_LON}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        rain_1h = data.get("rain", {}).get("1h", 0.0)
        daily_est = rain_1h * 24.0
        
        return {
            "daily_rainfall_mm": float(daily_est),
            "rainfall_3day_mm": float(daily_est * 2.8),
            "rainfall_7day_mm": float(daily_est * 6.5),
            "rainfall_14day_mm": float(daily_est * 12.0)
        }
    except Exception as e:
        print(f"Weather API fetch error: {e}")
        return {
            "daily_rainfall_mm": 12.5,
            "rainfall_3day_mm": 35.0,
            "rainfall_7day_mm": 80.0,
            "rainfall_14day_mm": 150.0
        }

def get_cyclical_month_features() -> tuple:
    current_month = datetime.now().month
    month_sin = np.sin(2 * np.pi * current_month / 12)
    month_cos = np.cos(2 * np.pi * current_month / 12)
    return float(month_sin), float(month_cos)

# --- PYDANTIC SCHEMAS ---
class SpatialTerrainPayload(BaseModel):
    daily_rainfall_mm: float = Field(..., description="Daily rainfall in mm")
    rainfall_3day_mm: float = Field(..., description="3-day cumulative rainfall in mm")
    rainfall_7day_mm: float = Field(..., description="7-day cumulative rainfall in mm")
    rainfall_14day_mm: float = Field(..., description="14-day cumulative rainfall in mm")
    month_sin: float = Field(..., description="Cyclical month sine component")
    month_cos: float = Field(..., description="Cyclical month cosine component")
    elevation_mean: float = Field(..., description="Mean elevation in meters")
    slope_mean: float = Field(..., description="Mean slope angle")
    flowacc_mean: float = Field(..., description="Mean flow accumulation")
    slope_pct_over_15deg: float = Field(..., description="Percentage of slope over 15 degrees")
    phone_number: str = Field(..., description="Recipient phone number in E.164 format")

class AutoCheckPayload(BaseModel):
    phone_number: str = Field(..., description="Recipient phone number in E.164 format (e.g., +23278616275)")
    elevation_mean: float = Field(45.2, description="Mean catchment elevation in meters")
    slope_mean: float = Field(12.4, description="Mean catchment slope angle")
    flowacc_mean: float = Field(1540.0, description="Mean flow accumulation")
    slope_pct_over_15deg: float = Field(25.5, description="Percentage of slope over 15 degrees")

# --- API ENDPOINTS ---
# ... FastAPI app initialization (app = FastAPI(...)) and other routes ...

@app.get("/")
def read_root():
    return {"message": "Freetown Flood Early Warning System API is running"}

# Add the health check endpoint right here:
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Freetown Flood Early Warning System",
        "version": "1.0.0"
    }

# ... your @app.post("/predict") endpoint and other logic below ...
@app.get("/")
def home():
    return {"status": "Freetown Flood EWS API is running live."}
@app.post("/predict/spatial-terrain")
def predict_spatial_terrain(payload: SpatialTerrainPayload):
    if model is None:
        raise HTTPException(status_code=500, detail="Prediction model is not loaded on the server.")
    
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
        probabilities = model.predict_proba(input_data)
        flood_probability = float(probabilities[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
    THRESHOLD = 0.4859
    is_high_risk = flood_probability >= THRESHOLD
    sms_status = "Not triggered (Risk below threshold)"
    
    if is_high_risk:
        alert_message = (
            f"ALERT: High flood risk detected in Freetown catchment "
            f"(Probability: {flood_probability:.1%}). Please take immediate precautions."
        )
        try:
            if sms_service:
                response = sms_service.send(alert_message, [payload.phone_number])
                sms_status = f"SMS Dispatch Success: {response}"
            else:
                sms_status = "SMS service uninitialized."
        except Exception as e:
            sms_status = f"Failed to send SMS: {str(e)}"

    return {
        "flood_probability": flood_probability,
        "decision_threshold": THRESHOLD,
        "risk_status": "HIGH_RISK" if is_high_risk else "SAFE",
        "alert_triggered": is_high_risk,
        "sms_dispatch_result": sms_status
    }

@app.post("/predict/auto-check")
def predict_auto_check(payload: AutoCheckPayload):
    if model is None:
        raise HTTPException(status_code=500, detail="Prediction model is not loaded on the server.")
    
    weather = fetch_freetown_weather_data()
    month_sin, month_cos = get_cyclical_month_features()
    
    input_data = pd.DataFrame([{
        "daily_rainfall_mm": weather["daily_rainfall_mm"],
        "rainfall_3day_mm": weather["rainfall_3day_mm"],
        "rainfall_7day_mm": weather["rainfall_7day_mm"],
        "rainfall_14day_mm": weather["rainfall_14day_mm"],
        "month_sin": month_sin,
        "month_cos": month_cos,
        "elevation_mean": payload.elevation_mean,
        "slope_mean": payload.slope_mean,
        "flowacc_mean": payload.flowacc_mean,
        "slope_pct_over_15deg": payload.slope_pct_over_15deg
    }])
    
    try:
        probabilities = model.predict_proba(input_data)
        flood_probability = float(probabilities[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
    THRESHOLD = 0.4859
    is_high_risk = flood_probability >= THRESHOLD
    sms_status = "Not triggered (Risk below threshold)"
    
    if is_high_risk:
        alert_message = (
            f"ALERT: High flood risk detected in Freetown catchment "
            f"(Probability: {flood_probability:.1%}). Please take immediate precautions."
        )
        try:
            if sms_service:
                response = sms_service.send(alert_message, [payload.phone_number])
                sms_status = f"SMS Dispatch Success: {response}"
            else:
                sms_status = "SMS service uninitialized."
        except Exception as e:
            sms_status = f"Failed to send SMS: {str(e)}"

    return {
        "ingested_weather": weather,
        "flood_probability": flood_probability,
        "decision_threshold": THRESHOLD,
        "risk_status": "HIGH_RISK" if is_high_risk else "SAFE",
        "alert_triggered": is_high_risk,
        "sms_dispatch_result": sms_status
    }