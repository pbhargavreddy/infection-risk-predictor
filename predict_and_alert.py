import requests
import pandas as pd
import joblib

# Load your models
scaler = joblib.load("model files/scaler.pkl")
pca = joblib.load("model files/pca.pkl")
model = joblib.load("model files/model.pkl")

# Fetch latest data from ThingSpeak channel
THINGSPEAK_API_KEY = "KL184FDN8MQGS4TD"
THINGSPEAK_CHANNEL_ID = "2963447"

url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?results=1&api_key={THINGSPEAK_API_KEY}"

response = requests.get(url)
data = response.json()

# Check if feeds exist and is not empty
feeds_list = data.get('feeds')
if not feeds_list or feeds_list[0] is None:
    print("No feed data available, skipping prediction.")
    exit(0)  # Gracefully exit workflow without error

feeds = feeds_list[0]

def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0  # default fallback

new_data = pd.DataFrame([{
    'Temp': safe_float(feeds.get('field1')),
    'Humidity': safe_float(feeds.get('field2')),
    'Pressure': safe_float(feeds.get('field3')),
    'PM2.5': safe_float(feeds.get('field4')),
    'CO2': safe_float(feeds.get('field5')),
    'TVOC': safe_float(feeds.get('field6')),
}])

print("Data for prediction:", new_data.to_dict(orient='records')[0])

# Continue with scaling, PCA, prediction
new_data_scaled = scaler.transform(new_data)
new_data_pca = pca.transform(new_data_scaled)
predicted_cluster = model.predict(new_data_pca)[0]

cluster_to_risk = {0: 'Low Risk', 1: 'High Risk', 2: 'Medium Risk'}
predicted_risk = cluster_to_risk[predicted_cluster]

print("Predicted Risk Level:", predicted_risk)

if predicted_risk in ['Low Risk','Medium Risk', 'High Risk']:
    WRITE_API_KEY = "BUT1G7Z2C06PGVS9"
    update_url = "https://api.thingspeak.com/update.json"
    payload = {
        'api_key': WRITE_API_KEY,
        'field7': predicted_risk,
        'field8': str(new_data.to_dict(orient='records')[0])
    }
    r = requests.post(update_url, data=payload)
    print("Alert sent, ThingSpeak response:", r.text)
