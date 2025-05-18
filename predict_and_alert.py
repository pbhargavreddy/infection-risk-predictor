import requests
import pandas as pd
import joblib
import time

# Load trained components
scaler = joblib.load("model files/scaler.pkl")
pca = joblib.load("model files/pca.pkl")
model = joblib.load("model files/model.pkl")

# ThingSpeak configuration
READ_API_KEY = "KL184FDN8MQGS4TD"
WRITE_API_KEY = "BUT1G7Z2C06PGVS9"
CHANNEL_ID = "2963447"

# Get last 5 feeds from ThingSpeak
url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?results=5&api_key={READ_API_KEY}"
response = requests.get(url)
data = response.json()

feeds = data.get('feeds', [])
if not feeds:
    print("No feed data available, skipping prediction.")
    exit(0)

# Helper function
def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0

# Prepare DataFrame
df = pd.DataFrame([{
    'Temp': safe_float(feed.get('field1')),
    'Humidity': safe_float(feed.get('field2')),
    'Pressure': safe_float(feed.get('field3')),
    'PM2.5': safe_float(feed.get('field4')),
    'CO2': safe_float(feed.get('field5')),
    'TVOC': safe_float(feed.get('field6')),
} for feed in feeds])

# Scale and predict
df_scaled = scaler.transform(df)
df_pca = pca.transform(df_scaled)
predicted_clusters = model.predict(df_pca)

# Map clusters to risks
cluster_to_risk = {0: 'Low Risk', 1: 'High Risk', 2: 'Medium Risk'}
predicted_risks = [cluster_to_risk[c] for c in predicted_clusters]

# Prepare message for field8 (risk labels)
risk_summary = ", ".join(predicted_risks)

# Send to ThingSpeak
update_url = "https://api.thingspeak.com/update.json"
payload = {
    'api_key': WRITE_API_KEY,
    'field7': predicted_clusters[-1],  # Most recent cluster ID
    'field8': risk_summary             # Risk labels from last 5
}

response = requests.post(update_url, data=payload)
print("Sent to ThingSpeak.")
print("Predicted Risks:", risk_summary)
print("ThingSpeak response:", response.text)
