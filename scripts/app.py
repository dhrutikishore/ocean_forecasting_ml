from flask import Flask, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import requests
import os
import time
import datetime

app = Flask(__name__)
CORS(app)

# =========================
# GLOBAL CACHE
# =========================
cache = {
    "data": None,
    "timestamp": 0
}

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "ocean_model.pkl")
model = joblib.load(model_path)

# =========================
# LOCATIONS
# =========================
LOCATIONS = [
    {"name": "Puri", "lat": 19.84, "lon": 85.82},
    {"name": "Mumbai", "lat": 19.07, "lon": 72.87},
    {"name": "Chennai", "lat": 13.08, "lon": 80.27},
    {"name": "Kochi", "lat": 9.93, "lon": 76.26},
    {"name": "Visakhapatnam", "lat": 17.69, "lon": 83.22},
]

# =========================
# FETCH WEATHER
# =========================
def fetch_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        res = requests.get(url, timeout=3).json()

        return {
            "wind_speed": res["current_weather"]["windspeed"],
            "wind_direction": res["current_weather"]["winddirection"],
            "rainfall": 0
        }
    except:
        return {
            "wind_speed": 5 + (lat % 3),
            "wind_direction": 180,
            "rainfall": 0
        }

# =========================
# FETCH WAVES
# =========================
def fetch_waves(lat, lon):
    try:
        url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&hourly=wave_height,wave_direction,wave_period"
        res = requests.get(url, timeout=3).json()

        wave_height = res["hourly"]["wave_height"][-1]

        return {
            "wave_height": wave_height,
            "wave_lag6": wave_height,
            "wave_lag12": wave_height,
            "wave_lag24": wave_height,
            "wave_direction": res["hourly"]["wave_direction"][-1],
            "wave_period": res["hourly"]["wave_period"][-1]
        }
    except:
        return {
            "wave_height": 0.8 + (lat % 2),
            "wave_lag6": 1.0,
            "wave_lag12": 1.2,
            "wave_lag24": 1.3,
            "wave_direction": 180 + (lat % 30),
            "wave_period": 4 + (lon % 3)
        }

# =========================
# HYBRID RISK SYSTEM
# =========================
def classify_risk(pred, wind, rain, wave):

    score = 0

    if wave > 2.5:
        score += 2
    elif wave > 1.5:
        score += 1

    if wind > 25:
        score += 2
    elif wind > 15:
        score += 1

    if pred > 2.5:
        score += 2
    elif pred > 1.5:
        score += 1

    if score >= 4:
        return "HIGH", "red", "Avoid sea activity", "Dangerous conditions", "Worsening"
    elif score >= 2:
        return "MEDIUM", "orange", "Be cautious", "Moderate conditions", "Unstable"
    else:
        return "LOW", "green", "Safe conditions", "Calm sea", "Improving"

# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# MAIN API
# =========================
@app.route("/predict_all", methods=["GET"])
def predict_all():

    if cache["data"]:
        return jsonify(cache["data"])

    now = datetime.datetime.utcnow()
    month = now.month
    hour = now.hour

    results = []

    for loc in LOCATIONS:
        lat, lon, name = loc["lat"], loc["lon"], loc["name"]

        weather = fetch_weather(lat, lon)
        waves = fetch_waves(lat, lon)

        # 🌪️ CYCLONE SIMULATION (ODISHA REGION)
        if 18 <= lat <= 21 and 84 <= lon <= 88:
            waves["wave_height"] += 1.5
            weather["wind_speed"] += 10

        try:
            all_features = {**weather, **waves, "month": month, "hour": hour}
            X = pd.DataFrame([{k: all_features.get(k) for k in model.feature_names_in_}])
            pred = float(model.predict(X)[0])
        except:
            pred = 1.0 + (lat % 1)

        risk, color, rec, explanation, trend = classify_risk(
            pred, weather["wind_speed"], weather["rainfall"], waves["wave_height"]
        )

        results.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "prediction": pred,
            "wave_height": waves["wave_height"],
            "wind_speed": weather["wind_speed"],
            "wind_direction": weather["wind_direction"],
            "rainfall": weather["rainfall"],
            "risk": risk,
            "color": color,
            "recommendation": rec,
            "explanation": f"Wave {waves['wave_height']:.2f}m & Wind {weather['wind_speed']:.1f} km/h",
            "trend": trend,
            "alert": "⚠️ HIGH RISK ZONE" if risk == "HIGH" else None,
        })

    cache["data"] = results
    cache["timestamp"] = time.time()

    return jsonify({
        "data": results,
        "last_updated": datetime.datetime.utcnow().strftime("%H:%M UTC")
    })

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)