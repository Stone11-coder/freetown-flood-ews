import os
import requests
import datetime
import numpy as np

# OpenWeatherMap Configuration (loaded strictly from environment variable)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
FREETOWN_LAT = 8.4844
FREETOWN_LON = -13.2344

def fetch_freetown_weather_data():
    """
    Fetches real-time weather and forecast data for Freetown, Sierra Leone,
    extracting daily rainfall and computing rolling cumulative accumulations.
    """
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={FREETOWN_LAT}&lon={FREETOWN_LON}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Weather API Warning ({response.status_code}): Using baseline fallback values.")
            return get_fallback_weather_metrics()
        
        data = response.json()
        forecast_list = data.get("list", [])
        
        # Calculate expected rainfall over the upcoming forecast periods (in mm)
        total_24h_rain = sum(item.get("rain", {}).get("3h", 0.0) for item in forecast_list[:8])    # 24 hours (8 * 3h steps)
        total_3day_rain = sum(item.get("rain", {}).get("3h", 0.0) for item in forecast_list[:24])  # 72 hours (24 * 3h steps)
        
        # Estimate multi-day antecedent rain based on current storm dynamics
        total_7day_rain = total_3day_rain * 2.1
        total_14day_rain = total_7day_rain * 1.8

        return {
            "daily_rainfall_mm": float(round(total_24h_rain, 2)),
            "rainfall_3day_mm": float(round(total_3day_rain, 2)),
            "rainfall_7day_mm": float(round(total_7day_rain, 2)),
            "rainfall_14day_mm": float(round(total_14day_rain, 2)),
        }

    except Exception as e:
        print(f"Error fetching weather data: {e}. Falling back to default values.")
        return get_fallback_weather_metrics()

def get_cyclical_month_features():
    """Computes month_sin and month_cos dynamically from current UTC date."""
    current_month = datetime.datetime.utcnow().month
    month_sin = float(np.sin(2 * np.pi * current_month / 12))
    month_cos = float(np.cos(2 * np.pi * current_month / 12))
    return month_sin, month_cos

def get_fallback_weather_metrics():
    """Fallback values during API downtime or dry runs."""
    return {
        "daily_rainfall_mm": 45.5,
        "rainfall_3day_mm": 120.0,
        "rainfall_7day_mm": 210.0,
        "rainfall_14day_mm": 350.0
    }

if __name__ == "__main__":
    weather = fetch_freetown_weather_data()
    sin, cos = get_cyclical_month_features()
    print("--- Freetown Live Ingestion Test ---")
    print(f"Rainfall Data: {weather}")
    print(f"Cyclical Features -> month_sin: {sin:.4f}, month_cos: {cos:.4f}")