import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, classification_report, confusion_matrix

DATA_PATH = Path("data/processed/freetown_catchment_daily_modeling_dataset_v3.csv")
MODEL_PATH = Path("models/freetown_spatial_flood_model_v1.pkl")

print("============================================================")
print("FREETOWN FLOOD EWS: THRESHOLD OPTIMIZATION (YOUDEN'S J)")
print("============================================================")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

if "month" in df.columns:
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

target_col = "flood_onset_next_3d" if "flood_onset_next_3d" in df.columns else "flood_event_today"
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

eval_df = df[spatial_features + [target_col]].dropna()
X = eval_df[spatial_features]
y = eval_df[target_col]

# Get predicted probabilities
probabilities = model.predict_proba(X)[:, 1]

# Calculate Youden's J statistic to find the optimal threshold
fpr, tpr, thresholds = roc_curve(y, probabilities)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nPrevious Threshold: 0.3266")
print(f"Optimal Tuned Threshold: {optimal_threshold:.4f}")

# Evaluate with new threshold
predictions = (probabilities >= optimal_threshold).astype(int)

print("-" * 60)
print("Confusion Matrix (Optimized):")
print(confusion_matrix(y, predictions))
print("-" * 60)
print("\nClassification Report (Optimized):")
print(classification_report(y, predictions, target_names=["Normal", "Flood Alert"]))