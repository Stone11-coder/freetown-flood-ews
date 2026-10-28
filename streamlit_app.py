import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import datetime

# Page Configuration
st.set_page_config(
    page_title="Freetown Flood EWS Dashboard",
    page_icon="🌊",
    layout="wide"
)

# FastAPI Backend URL (Update this with your live Render URL once ready)
API_URL = "http://127.0.0.1:8000/predict/spatial-terrain"

st.title("🌊 Freetown Flood Early Warning System (EWS)")
st.markdown("Geospatial machine learning platform for automated catchment risk assessment and SMS alert dispatch.")

# Layout: Two columns (Left: Inputs & Map, Right: Results & Simulation)
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📍 Catchment & Meteorological Parameters")
    
    # Rainfall Inputs
    st.markdown("#### Rainfall Metrics (mm)")
    daily_rainfall = st.slider("Daily Rainfall (mm)", 0.0, 150.0, 45.5)
    rainfall_3day = st.slider("3-Day Cumulative Rainfall (mm)", 0.0, 300.0, 120.0)
    rainfall_7day = st.slider("7-Day Cumulative Rainfall (mm)", 0.0, 500.0, 210.0)
    rainfall_14day = st.slider("14-Day Cumulative Rainfall (mm)", 0.0, 800.0, 350.0)
    
    # Cyclical Month Calculation based on current date
    current_month = datetime.datetime.now().month
    month_sin = np_sin = np.sin(2 * np.pi * current_month / 12) if 'np' in globals() else 0.5
    month_cos = np.cos(2 * np.pi * current_month / 12) if 'np' in globals() else 0.5
    
    with st.expander("Advanced Temporal & Topographic Metrics"):
        elevation_mean = st.number_input("Mean Elevation (m)", value=45.2)
        slope_mean = st.number_input("Mean Slope Angle (deg)", value=12.4)
        flowacc_mean = st.number_input("Mean Flow Accumulation", value=1540.0)
        slope_pct_over_15deg = st.slider("Slope % > 15°", 0.0, 100.0, 25.5)
        phone_number = st.text_input("Recipient Phone Number (E.164)", value="+23278616275")

    predict_btn = st.button("Run Flood Risk Assessment & Dispatch Alert", type="primary", use_container_width=True)

with col2:
    st.subheader("🗺️ Freetown Catchment Focus")
    
    # Interactive Leaflet Map centered around Freetown
    freetown_map = folium.Map(location=[8.4844, -13.2344], zoom_start=12)
    folium.Marker(
        [8.4844, -13.2344],
        popup="Freetown Catchment Monitoring Point",
        tooltip="Freetown EWS Zone",
        icon=folium.Icon(color="blue", icon="tint")
    ).add_to(freetown_map)
    
    st_folium(freetown_map, height=300, use_container_width=True)
    
    st.subheader("📊 Real-Time Inference Results")
    
    if predict_btn:
        payload = {
            "daily_rainfall_mm": daily_rainfall,
            "rainfall_3day_mm": rainfall_3day,
            "rainfall_7day_mm": rainfall_7day,
            "rainfall_14day_mm": rainfall_14day,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "elevation_mean": elevation_mean,
            "slope_mean": slope_mean,
            "flowacc_mean": flowacc_mean,
            "slope_pct_over_15deg": slope_pct_over_15deg,
            "phone_number": phone_number
        }
        
        try:
            with st.spinner("Analyzing geospatial features and checking SMS gateway..."):
                response = requests.post(API_URL, json=payload, timeout=15)
                
            if response.status_code == 200:
                result = response.json()
                prob = result["flood_probability"]
                status = result["risk_status"]
                sms_res = result["sms_dispatch_result"]
                
                if status == "HIGH_RISK":
                    st.error(f"⚠️ **HIGH RISK DETECTED** (Probability: {prob:.1%})")
                else:
                    st.success(f"✅ **SAFE** (Probability: {prob:.1%})")
                    
                st.metric(label="Calculated Flood Probability", value=f"{prob:.2%}", delta=f"Threshold: 48.59%")
                st.info(f"**SMS Status:** {sms_res}")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to FastAPI backend: {e}")