import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix

DATA_PATH = Path("data/processed/freetown_catchment_daily_modeling_dataset_v3.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

print("============================================================")
print("FREETOWN FLOOD EWS: XGBOOST TRAINING & EVALUATION")
print("============================================================")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}.")

df = pd.read_csv(DATA_PATH)

# Derive cyclical seasonal features
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

model_df = df[spatial_features + [target_col]].dropna()
X = model_df[spatial_features]
y = model_df[target_col]

# Handle severe class imbalance using scale_pos_weight for XGBoost
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_pos_weight = neg_count / pos_count

print(f"Dataset rows after cleaning: {model_df.shape[0]}")
print(f"Class distribution -> Normal: {neg_count:,}, Flood Events: {pos_count}")
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

print("\nTraining XGBoost Classifier on spatial-terrain data...")
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)
xgb_model.fit(X, y)

# Evaluate predictions
probabilities = xgb_model.predict_proba(X)[:, 1]
auc = roc_auc_score(y, probabilities)
print(f"\n>>> XGBoost ROC-AUC Score: {auc:.4f}")

# Find optimal threshold using Youden's J statistic
fpr, tpr, thresholds = roc_curve(y, probabilities)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

predictions = (probabilities >= optimal_threshold).astype(int)

print(f"Optimal Tuned Threshold: {optimal_threshold:.4f}")
print("-" * 60)
print("Confusion Matrix (XGBoost):")
print(confusion_matrix(y, predictions))
print("-" * 60)
print("\nClassification Report (XGBoost):")
print(classification_report(y, predictions, target_names=["Normal", "Flood Alert"]))

# Save XGBoost artifact
model_path = MODELS_DIR / "freetown_xgboost_flood_model_v1.pkl"
joblib.dump(xgb_model, model_path)
print(f"Successfully saved XGBoost model artifact to: {model_path}")