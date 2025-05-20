from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
# ThingSpeak
READ_API_KEY = os.getenv("READ_API_KEY")
READ_CHANNEL_ID = os.getenv("READ_CHANNEL_ID")
PREDICTION_WRITE_API_KEY = os.getenv("PREDICTION_WRITE_API_KEY")

# Email settings
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

import os
import requests
import pandas as pd
import joblib
from scipy.stats import mode
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart



# ML model paths
SCALER_PATH = "model files/scaler.pkl"
PCA_PATH = "model files/pca.pkl"
MODEL_PATH = "model files/model.pkl"

# ThingSpeak
READ_API_KEY = "KL184FDN8MQGS4TD"
READ_CHANNEL_ID = "2963447"
PREDICTION_WRITE_API_KEY = "JPJL9MPVSH2VNR1B"  


SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "infectionriskprediction@gmail.com"
EMAIL_PASSWORD = "tleh zksr pafx ucbg"  
EMAIL_RECEIVER = "pbhargavreddy3@gmail.com"

# File to store last sent risk
LAST_RISK_FILE = 'last_sent_risk.txt'

#  Load ML Components 
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)
model = joblib.load(MODEL_PATH)

#  Helper functions 

def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0

def get_last_sent_risk():
    if os.path.exists(LAST_RISK_FILE):
        with open(LAST_RISK_FILE, 'r') as f:
            return f.read().strip()
    return None

def set_last_sent_risk(risk):
    with open(LAST_RISK_FILE, 'w') as f:
        f.write(risk)

def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("Email alert sent successfully.")
    except Exception as e:
        print("Failed to send email:", e)

#  Fetch Sensor Data 
url = f"https://api.thingspeak.com/channels/{READ_CHANNEL_ID}/feeds.json?results=20&api_key={READ_API_KEY}"
response = requests.get(url)
feeds = response.json().get('feeds', [])

if not feeds:
    print("No feed data available. Exiting.")
    exit()

# Prepare DataFrame
df = pd.DataFrame([{
    'Temp': safe_float(feed.get('field1')),
    'Humidity': safe_float(feed.get('field2')),
    'Pressure': safe_float(feed.get('field3')),
    'PM2.5': safe_float(feed.get('field4')),
    'CO2': safe_float(feed.get('field5')),
    'TVOC': safe_float(feed.get('field6')),
} for feed in feeds])

#  Prediction 
df_scaled = scaler.transform(df)
df_pca = pca.transform(df_scaled)
predicted_clusters = model.predict(df_pca)

# Map cluster to risk
cluster_to_risk = {0: 'Low Risk', 1: 'High Risk', 2: 'Medium Risk'}
predicted_risks = [cluster_to_risk[c] for c in predicted_clusters]

# Get mode of prediction
mode_cluster = int(mode(predicted_clusters, keepdims=False).mode)
mode_risk = cluster_to_risk[mode_cluster]
latest_feed = df.iloc[-1]

# Update ThingSpeak 
update_url = "https://api.thingspeak.com/update.json"
payload = {
    'api_key': PREDICTION_WRITE_API_KEY,
    'field1': latest_feed['Temp'],
    'field2': latest_feed['Humidity'],
    'field3': latest_feed['Pressure'],
    'field4': latest_feed['PM2.5'],
    'field5': latest_feed['CO2'],
    'field6': latest_feed['TVOC'],
    'field7': mode_cluster,
    'field8': mode_risk
}
response = requests.post(update_url, data=payload)

print("Sent to ThingSpeak (Predictions Channel).")
print("Predicted Risks:", predicted_risks)
print("Mode Cluster:", mode_cluster)
print("ThingSpeak response:", response.text)

# Send Email If Needed 
if mode_risk in ["Medium Risk", "High Risk"]:
    last_risk_sent = get_last_sent_risk()
    if mode_risk != last_risk_sent:
        subject = f"[ALERT] Infection Risk: {mode_risk}"
        body = f"""
Predicted Infection Risk: {mode_risk}
Cluster ID: {mode_cluster}

Latest Sensor Readings:
- Temperature: {latest_feed['Temp']} °C
- Humidity: {latest_feed['Humidity']} %
- Pressure: {latest_feed['Pressure']} hPa
- PM2.5: {latest_feed['PM2.5']} µg/m³
- CO₂: {latest_feed['CO2']} ppm
- TVOC: {latest_feed['TVOC']} ppb

Data updated on ThingSpeak: https://thingspeak.com/channels/{READ_CHANNEL_ID}
"""
        send_email(subject, body)
        set_last_sent_risk(mode_risk)
    else:
        print(f"Email already sent for risk level: {mode_risk}. No new email sent.")
else:
    print(f"Risk level '{mode_risk}' does not require an email alert.")
    set_last_sent_risk('Low Risk')
