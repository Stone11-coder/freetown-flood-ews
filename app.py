import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Freetown Flood Early Warning System",
    page_icon="🌊",
    layout="wide"
)

# --- CONFIGURABLE BACKEND URL ---
# Automatically uses your live Render deployment URL
API_BASE_URL = "https://freetown-flood-ews.onrender.com"

# --- CACHE DATASET PREVIEW ---
@st.cache_data
def load_data():
    data_path = Path("data/processed/freetown_catchment_daily_modeling_dataset_v3.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

df = load_data()

# --- HEADER ---
st.title("🌊 Freetown Flood Early Warning System (EWS)")
st.markdown("**Operational Decision Support Dashboard | Powered by FastAPI, XGBoost & Africa's Talking SMS**")
st.markdown("---")

# --- SIDEBAR CONTROLS FOR LIVE SIMULATION ---
st.sidebar.header("Live Catchment Simulator")
st.sidebar.markdown("Adjust meteorological and terrain features to query the live cloud model.")

daily_rain = st.sidebar.slider("Daily Rainfall (mm)", 0.0, 150.0, 48.645)
rain_3d = st.sidebar.slider("3-Day Cumulative Rainfall (mm)", 0.0, 300.0, 49.475)
rain_7d = st.sidebar.slider("7-Day Cumulative Rainfall (mm)", 0.0, 500.0, 155.08)
rain_14d = st.sidebar.slider("14-Day Cumulative Rainfall (mm)", 0.0, 800.0, 259.52)
month_val = st.sidebar.slider("Month (1 = Jan, 8 = Aug Peak Monsoon)", 1, 12, 8)

st.sidebar.markdown("### Topography & Catchment Metrics")
elevation = st.sidebar.number_input("Elevation Mean (m)", value=142.35)
slope = st.sidebar.number_input("Slope Mean (degrees)", value=12.58)
flowacc = st.sidebar.number_input("Flow Accumulation Mean", value=144.53)
slope_15 = st.sidebar.number_input("Slope % Over 15°", value=0.328)

st.sidebar.markdown("### 📱 SMS Alert Recipient")
phone_number = st.sidebar.text_input("Phone Number (E.164 Format)", value="+23278616275")

# Cyclical Seasonality Encoding
month_sin = np.sin(2 * np.pi * month_val / 12)
month_cos = np.cos(2 * np.pi * month_val / 12)

# Run Inference Request to Live Backend
if st.sidebar.button("Run Live Assessment & Alert Check", type="primary"):
    payload = {
        "daily_rainfall_mm": daily_rain,
        "rainfall_3day_mm": rain_3d,
        "rainfall_7day_mm": rain_7d,
        "rainfall_14day_mm": rain_14d,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "elevation_mean": elevation,
        "slope_mean": slope,
        "flowacc_mean": flowacc,
        "slope_pct_over_15deg": slope_15,
        "phone_number": phone_number
    }
    
    try:
        with st.spinner("Contacting Render FastAPI Backend..."):
            response = requests.post(f"{API_BASE_URL}/predict/spatial-terrain", json=payload, timeout=15)
            
        if response.status_code == 200:
            result = response.json()
            prob = result["flood_probability"]
            threshold = result["decision_threshold"]
            risk_status = result["risk_status"]
            sms_result = result["sms_dispatch_result"]
            
            if risk_status == "HIGH_RISK":
                badge_color = "red"
                risk_label = "WARNING (HIGH RISK)"
            else:
                badge_color = "green"
                risk_label = "SAFE (NORMAL)"
                
            st.sidebar.markdown("---")
            st.sidebar.subheader("Live Model Prediction Result")
            st.sidebar.metric(label="Calculated Flood Probability", value=f"{prob:.4f}")
            st.sidebar.markdown(f"**Risk Level:** :{badge_color}[**{risk_label}**]")
            st.sidebar.markdown(f"**Decision Threshold:** `{threshold}`")
            st.sidebar.info(f"**SMS Status:** {sms_result}")
        else:
            st.sidebar.error(f"API Error ({response.status_code}): {response.text}")
    except Exception as e:
        st.sidebar.error(f"Failed to connect to backend: {e}")

# --- MAIN DASHBOARD VIEW ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Historical Records", f"{len(df):,}" if df is not None else "0")
col2.metric("Active Model Engine", "XGBoost Classifier v2.1")
col3.metric("Backend Connection", "LIVE (Render)")
col4.metric("System Status", "ONLINE")

st.markdown("---")

if df is not None:
    st.subheader("📊 Freetown Catchment Modeling Dataset Preview")
    st.markdown("Showing first 10 rows of processed topographical and meteorological data across Freetown catchments.")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("### Quick Dataset Summary Statistics")
    st.dataframe(df[['daily_rainfall_mm', 'rainfall_7day_mm', 'elevation_mean', 'slope_mean', 'flowacc_mean']].describe(), use_container_width=True)
else:
    st.warning("Processed dataset not found at `data/processed/freetown_catchment_daily_modeling_dataset_v3.csv`.")