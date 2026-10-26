import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Paths
DATA_PATH = Path("data/processed/freetown_catchment_daily_modeling_dataset_v3.csv")
MODEL_PATH = Path("models/freetown_spatial_flood_model_v1.pkl")

print("============================================================")
print("FREETOWN FLOOD EWS: FULL HISTORICAL DATASET EVALUATION")
print("============================================================")

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    raise FileNotFoundError("Dataset or model artifact not found. Ensure both paths exist.")

# Load data and model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# Derive seasonal features
if "month" in df.columns:
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Select target and features matching training schema
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

# Prepare evaluation subset
eval_df = df[spatial_features + [target_col]].dropna()
X = eval_df[spatial_features]
y = eval_df[target_col]

# Generate predictions across all rows
probabilities = model.predict_proba(X)[:, 1]
threshold = 0.3266
predictions = (probabilities >= threshold).astype(int)

# Calculate full-dataset metrics
full_auc = roc_auc_score(y, probabilities)
conf_matrix = confusion_matrix(y, predictions)

print(f"\nTotal Records Evaluated: {len(y):,}")
print(f"Full Dataset ROC-AUC Score: {full_auc:.4f}")
print("-" * 60)
print("Confusion Matrix:")
print(" [[True Negatives (Correct Normal), False Positives (False Alarms)]]")
print(" [[False Negatives (Missed Floods), True Positives (Caught Floods)]]")
print(conf_matrix)
print("-" * 60)
print("\nClassification Report:")
print(classification_report(y, predictions, target_names=["Normal", "Flood Alert"]))