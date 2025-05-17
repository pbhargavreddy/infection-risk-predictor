import requests
import pandas as pd
import joblib

scaler = joblib.load("model files/scaler.pkl")
pca = joblib.load("model files/pca.pkl")
model = joblib.load("model files/model.pkl")

THINGSPEAK_API_KEY = "KL184FDN8MQGS4TD"
THINGSPEAK_CHANNEL_ID = "2963447"

url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?results=1&api_key={THINGSPEAK_API_KEY}"

try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
except Exception as e:
    print(f"Error fetching ThingSpeak data: {e}")
    exit(1)

feeds = data['feeds'][0]

new_data = pd.DataFrame([{
    'Temp': float(feeds['field1']),
    'Humidity': float(feeds['field2']),
    'Pressure': float(feeds['field3']),
    'PM2.5': float(feeds['field4']),
    'CO2': float(feeds['field5']),
    'TVOC': float(feeds['field6']),
}])

new_data_scaled = scaler.transform(new_data)
new_data_pca = pca.transform(new_data_scaled)
predicted_cluster = model.predict(new_data_pca)[0]

cluster_to_risk = {0: 'Low Risk', 1: 'High Risk', 2: 'Medium Risk'}
predicted_risk = cluster_to_risk[predicted_cluster]

print("Predicted Risk Level:", predicted_risk)

WRITE_API_KEY = "BUT1G7Z2C06PGVS9"
update_url = "https://api.thingspeak.com/update.json"

risk_to_code = {'Low Risk': 0, 'Medium Risk': 2, 'High Risk': 1}

payload = {
    'api_key': WRITE_API_KEY,
    'field7': risk_to_code[predicted_risk],
    'field8': str(new_data.to_dict(orient='records')[0])
}

try:
    r = requests.post(update_url, data=payload, timeout=10)
    print("ThingSpeak update status code:", r.status_code)
    print("ThingSpeak response text:", r.text)
    r.raise_for_status()
except Exception as e:
    print(f"Error sending update to ThingSpeak: {e}")
