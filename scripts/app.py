from flask import Flask, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import requests
import os
import time
import datetime
cache = {
    "data": None,
    "timestamp": None
}
app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL (FIXED PATH)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "ocean_model.pkl")
model = joblib.load(model_path)

# =========================
# LOCATIONS
# =========================
LOCATIONS = [
    {"name": "Puri", "lat": 19.84, "lon": 85.82},
    {"name": "Gopalpur", "lat": 19.26, "lon": 84.91},
    {"name": "Chennai", "lat": 13.08, "lon": 80.27},
    {"name": "Mumbai", "lat": 19.07, "lon": 72.87},
]

# =========================
# FETCH WEATHER
# =========================
def fetch_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    res = requests.get(url).json()

    return {
        "wind_speed": res["current_weather"]["windspeed"],
        "wind_direction": res["current_weather"]["winddirection"],
        "rainfall": 0
    }

# =========================
# FETCH WAVES
# =========================
def fetch_waves(lat, lon):
    url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&hourly=wave_height,wave_direction,wave_period"
    res = requests.get(url).json()

    wave_height = res["hourly"]["wave_height"][-1]

    return {
        "wave_height": wave_height,
        "wave_lag6": wave_height,
        "wave_lag12": wave_height,
        "wave_lag24": wave_height,
        "wave_direction": res["hourly"]["wave_direction"][-1],
        "wave_period": res["hourly"]["wave_period"][-1]
    }

# =========================
# RISK CLASSIFICATION
# =========================
def classify_risk(pred, wind, rain, wave):
    if pred > 2.5:
        return "HIGH", "red", "Avoid sea activity", "High waves expected", "Likely to worsen"
    elif pred > 1.5:
        return "MEDIUM", "orange", "Be cautious", "Moderate waves", "Stable"
    else:
        return "LOW", "green", "Safe conditions", "Low waves", "Improving"

# =========================
# HOME ROUTE (SERVES UI)
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# MAIN API
# =========================
@app.route("/predict_all", methods=["GET"])
def predict_all():
    cache = {
    "data": None,
    "timestamp": None
}
    now = datetime.datetime.utcnow()
    month = now.month
    hour = now.hour

    results = []

    for loc in LOCATIONS:
        lat, lon, name = loc["lat"], loc["lon"], loc["name"]

        # ---- WEATHER ----
        try:
            weather = fetch_weather(lat, lon)
        except Exception as e:
            print(f"Weather API failed for {name}: {e}")
            weather = {"wind_speed": 5, "wind_direction": 180, "rainfall": 0}

        # ---- WAVES ----
        try:
            waves = fetch_waves(lat, lon)
        except Exception as e:
            print(f"Wave API failed for {name}: {e}")
            waves = {
                "wave_height": 1.0,
                "wave_lag6": 1.0,
                "wave_lag12": 1.0,
                "wave_lag24": 1.0,
                "wave_direction": 180,
                "wave_period": 5
            }

        # ---- MODEL INPUT ----
        try:
            all_features = {**weather, **waves, "month": month, "hour": hour}
            X = pd.DataFrame([{k: all_features.get(k) for k in model.feature_names_in_}])
            pred = float(model.predict(X)[0])
        except Exception as e:
            print(f"Prediction failed for {name}: {e}")
            pred = 1.0

        # ---- CLASSIFICATION ----
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
            "explanation": explanation,
            "trend": trend,
        })

        # ---- RATE LIMIT FIX ----
        time.sleep(1)
    cache["data"] = results
    cache["timestamp"] = time.time()
    return jsonify(results)

# =========================
# RUN (LOCAL ONLY)
# =========================
if __name__ == "__main__":
    app.run(debug=True)