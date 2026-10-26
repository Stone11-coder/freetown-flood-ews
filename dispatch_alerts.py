import os
import requests
import pandas as pd

# Africa's Talking API Configuration (Loads securely from environment variables)
AT_USERNAME = os.getenv("AT_USERNAME", "sandbox")  # Use "sandbox" for testing, or your live username
AT_API_KEY = os.getenv("AT_API_KEY", "your_api_key_here")
SENDER_ID = os.getenv("AT_SENDER_ID", "FloodEWS")   # Optional custom sender ID if approved

def send_sms_alert(phone_numbers: list[str], message: str):
    """
    Dispatches SMS alerts via Africa's Talking API to a list of phone numbers.
    """
    url = "https://api.sandbox.africastalking.com/version1/messaging" # Change to api.africastalking.com for live production
    
    headers = {
        "apiKey": AT_API_KEY,
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    payload = {
        "username": AT_USERNAME,
        "to": ",".join(phone_numbers),
        "message": message,
        "from": SENDER_ID
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        response_data = response.json()
        print("SMS Dispatch Response:", response_data)
        return response_data
    except Exception as e:
        print(f"Failed to dispatch SMS: {e}")
        return None

def process_and_dispatch_forecasts(forecast_csv_path: str, stakeholder_contacts: list[str]):
    """
    Reads batch forecast outputs, filters by Youden's J threshold (0.9537),
    and triggers emergency SMS warnings for high-risk catchments.
    """
    THRESHOLD = 0.9537
    df = pd.read_csv(forecast_csv_path)
    
    # Filter for catchments exceeding the optimized threshold
    high_risk_zones = df[df['flood_probability'] >= THRESHOLD]
    
    if high_risk_zones.empty:
        print("No catchments exceeded the flood risk threshold. All clear.")
        return

    print(f"ALERT: {len(high_risk_zones)} catchment(s) at critical risk!")
    
    for _, row in high_risk_zones.iterrows():
        catchment_id = row['catchment_id']
        prob = row['flood_probability']
        
        # Craft the emergency bulletin
        bulletin = (
            f"URGENT FLOOD ALERT [Freetown EWS]\n"
            f"Catchment: {catchment_id}\n"
            f"Risk Probability: {prob:.4f}\n"
            f"Status: CRITICAL. Immediate precautionary evacuation/action advised."
        )
        
        print(f"\nDispatching bulletin for {catchment_id}...")
        send_sms_alert(stakeholder_contacts, bulletin)

if __name__ == "__main__":
    # Example test contacts (Must include country code, e.g., +232...)
    test_contacts = ["+23270000000"]
    # process_and_dispatch_forecasts("reports/batch_forecast_results.csv", test_contacts)