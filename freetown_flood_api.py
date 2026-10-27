from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Freetown Flood Early Warning System (EWS) - XGBoost Engine",
    description="Real-time flash flood risk scoring engine powered by XGBoost and spatial terrain integration.",
    version="2.0.0"
)

# Updated model path pointing to the XGBoost artifact
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

# Pydantic schema for Spatial & Terrain-Integrated Features (10 features)
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
    
    # Convert incoming request to DataFrame matching training schema
    input_df = pd.DataFrame([data.dict()])
    
    # Inference using XGBoost
    probability = float(model.predict_proba(input_df)[:, 1][0])
    threshold = 0.4859  # Optimized Youden's J threshold for XGBoost  
    alert = probability >= threshold
    
    # Risk categorization
    if probability >= 0.98:
        risk = "SEVERE"
    elif probability >= threshold:
        risk = "WARNING"
    else:
        risk = "NORMAL"
        
    return {
        "flood_probability": round(probability, 4),
        "alert_triggered": bool(alert),
        "threshold_used": threshold,
        "risk_level": risk
    }