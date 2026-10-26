import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Setup paths
DATA_PATH = Path("data/processed/freetown_catchment_daily_modeling_dataset_v3.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

print("============================================================")
print("FREETOWN FLOOD EWS: CORRECTED MODEL TRAINING PIPELINE")
print("============================================================")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}.")

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Derive cyclical month features if 'month' exists
if "month" in df.columns:
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Select target column
target_col = "flood_onset_next_3d" if "flood_onset_next_3d" in df.columns else "flood_event_today"
print(f"Selected target column: {target_col}")

# Feature sets
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

# Clean data
model_df = df[spatial_features + [target_col]].dropna()
print(f"Rows after handling missing values: {model_df.shape[0]}")

X = model_df[spatial_features]
y = model_df[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Terrain-Integrated Spatial Model on historical data...")
rf_spatial = RandomForestClassifier(
    n_estimators=100, 
    max_depth=12, 
    random_state=42, 
    class_weight="balanced"
)
rf_spatial.fit(X_train, y_train)

# Evaluate
proba = rf_spatial.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"Spatial Model ROC-AUC Score: {auc:.4f}")

# Save model artifact
model_path = MODELS_DIR / "freetown_spatial_flood_model_v1.pkl"
joblib.dump(rf_spatial, model_path)
print(f"Successfully trained and saved model artifact to: {model_path}")