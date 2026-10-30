import pandas as pd
import joblib
from pathlib import Path

# Load data and model
df = pd.read_csv("data/processed/freetown_catchment_daily_modeling_dataset_v3.csv")
model = joblib.load("models/freetown_xgboost_flood_model_v1.pkl")

if "month" in df.columns:
    import numpy as np
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

target_col = "flood_onset_next_3d" if "flood_onset_next_3d" in df.columns else "flood_event_today"
features = [
    "daily_rainfall_mm", "rainfall_3day_mm", "rainfall_7day_mm", "rainfall_14day_mm",
    "month_sin", "month_cos", "elevation_mean", "slope_mean", "flowacc_mean", "slope_pct_over_15deg"
]

# Find a true positive where probability >= 0.4859
eval_df = df[features + [target_col]].dropna()
probs = model.predict_proba(eval_df[features])[:, 1]
eval_df['prob'] = probs

# Filter for actual floods with high model probability
success_cases = eval_df[(eval_df[target_col] == 1) & (eval_df['prob'] >= 0.4859)]

if not success_cases.empty:
    sample = success_cases.iloc[0]
    print(f"Found successful flood trigger! Probability: {sample['prob']:.4f}")
    payload = {col: float(sample[col]) for col in features}
    import json
    print(json.dumps(payload, indent=2))
else:
    print("No high-probability flood records found.")