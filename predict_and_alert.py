import requests
import pickle
import numpy as np

# Load models
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

# ThingSpeak parameters
READ_API_URL = "https://api.thingspeak.com/channels/2963447/feeds.json?api_key=KL184FDN8MQGS4TD&results=1"
WRITE_API_KEY = "BUT1G7Z2C06PGVS9"

# Get latest sensor data
response = requests.get(READ_API_URL)
data = response.json()

try:
    feed = data['feeds'][0]
    features = [
        float(feed['field1']),  # temperature
        float(feed['field2']),  # humidity
        float(feed['field3']),  # pressure
        float(feed['field5']),  # CO2
        float(feed['field6']),  # TVOC
        float(feed['field4'])   # Dust
    ]

    # Preprocessing
    X_scaled = scaler.transform([features])
    X_pca = pca.transform(X_scaled)
    prediction = model.predict(X_pca)[0]

    # Write back to ThingSpeak
    RISK_LABEL = ['Low', 'Medium', 'High'][prediction]
    print(f"Predicted Risk: {RISK_LABEL}")

    response = requests.get(
        f"https://api.thingspeak.com/update?api_key={WRITE_API_KEY}&field7={prediction}&field8={RISK_LABEL}"
    )
    print("Update Response:", response.status_code)

except Exception as e:
    print("Error:", e)
