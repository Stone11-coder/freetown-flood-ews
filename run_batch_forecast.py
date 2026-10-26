import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# Paths
DATA_PATH = Path("data/processed/freetown_catchment_daily_modeling_dataset_v3.csv")
MODEL_PATH = Path("models/freetown_xgboost_flood_model_v1.pkl")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / f"freetown_batch_forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

print("============================================================")
print("FREETOWN FLOOD EWS: AUTOMATED BATCH FORECASTING ENGINE")
print("============================================================\n")

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    raise FileNotFoundError("Required dataset or model artifact not found.")

# 1. Load data and model
print("Loading dataset and XGBoost model artifact...")
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# 2. Feature engineering (cyclical seasonals)
if "month" in df.columns:
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# 3. Define schema features
baseline_features = [
    "daily_rainfall_mm", 
    "rainfall_3day_mm", 
    "rainfall_7day_mm", 
    "rainfall_14day_mm", 
    "month_sin", 
    "month_cos"
]
terrain_features = [
    "elevation_mean", 
    "slope_mean", 
    "flowacc_mean", 
    "slope_pct_over_15deg"
]
spatial_features = baseline_features + [col for col in terrain_features if col in df.columns]

# Prepare prediction subset
predict_df = df[spatial_features].copy()
X = predict_df

# 4. Run Batch Inference
print(f"Running inference across {len(X):,} records...")
probabilities = model.predict_proba(X)[:, 1]

# Assign metrics and thresholds
threshold = 0.9537
df["flood_probability"] = np.round(probabilities, 4)
df["alert_triggered"] = probabilities >= threshold

# Categorize Risk Levels
def assign_risk(prob):
    if prob >= 0.98:
        return "SEVERE"
    elif prob >= threshold:
        return "WARNING"
    else:
        return "NORMAL"

df["risk_level"] = df["flood_probability"].apply(assign_risk)

# 5. Summary Statistics
severe_count = (df["risk_level"] == "SEVERE").sum()
warning_count = (df["risk_level"] == "WARNING").sum()
normal_count = (df["risk_level"] == "NORMAL").sum()

print("\n------------------------------------------------------------")
print("BATCH FORECAST SUMMARY RESULTS:")
print(f"Total Records Processed: {len(df):,}")
print(f" - SEVERE Risk Alerts:  {severe_count:,}")
print(f" - WARNING Risk Alerts: {warning_count:,}")
print(f" - NORMAL Status:       {normal_count:,}")
print("------------------------------------------------------------")

# 6. Save Report
df.to_csv(OUTPUT_PATH, index=False)
print(f"Batch forecast report successfully saved to: {OUTPUT_PATH}")