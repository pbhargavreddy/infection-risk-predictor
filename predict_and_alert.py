import requests
import pandas as pd
import joblib

# Load your models
scaler = joblib.load("model files/scaler.pkl")
pca = joblib.load("model files/pca.pkl")
model = joblib.load("model files/model.pkl")

# Fetch latest data from ThingSpeak channel
THINGSPEAK_API_KEY = "BUT1G7Z2C06PGVS9"
THINGSPEAK_CHANNEL_ID = "2963447"

url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?results=1&api_key={THINGSPEAK_API_KEY}"

response = requests.get(url)
data = response.json()

feeds = data['feeds'][0]

new_data = pd.DataFrame([{
    'Temp': float(feeds['field1']),
    'Humidity': float(feeds['field2']),
    'Pressure': float(feeds['field3']),
    'PM2.5': float(feeds['field4']),
    'CO2': float(feeds['field5']),
    'TVOC': float(feeds['field6']),
}])

# Scale, PCA, predict
new_data_scaled = scaler.transform(new_data)
new_data_pca = pca.transform(new_data_scaled)
predicted_cluster = model.predict(new_data_pca)[0]

cluster_to_risk = {0: 'Low Risk', 1: 'High Risk', 2: 'Medium Risk'}
predicted_risk = cluster_to_risk[predicted_cluster]

print("Predicted Risk Level:", predicted_risk)

# If Medium or High risk, send alert by updating ThingSpeak field or trigger ThingSpeak notification
if predicted_risk in ['Low Risk','Medium Risk', 'High Risk']:
    WRITE_API_KEY = "KL184FDN8MQGS4TD"
    update_url = f"https://api.thingspeak.com/update.json"
    payload = {
        'api_key': WRITE_API_KEY,
        'field7': predicted_risk,
        'field8': str(new_data.to_dict(orient='records')[0])
    }
    r = requests.post(update_url, data=payload)
    print("Alert sent, ThingSpeak response:", r.text)
