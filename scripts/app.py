from flask import Flask, jsonify
import joblib
import requests
import pandas as pd
from flask_cors import CORS
import os 

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "ocean_model.pkl")

model = joblib.load(model_path)

LOCATIONS = [
    {"name": "Puri",     "lat": 19.84, "lon": 85.82},
    {"name": "Gopalpur", "lat": 19.26, "lon": 84.91},
    {"name": "Chennai",  "lat": 13.08, "lon": 80.27},
    {"name": "Mumbai",   "lat": 19.07, "lon": 72.87},
]

TIMEOUT = 8


def fetch_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=windspeed_10m,winddirection_10m,precipitation"
    )
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    return {
        "wind_speed":     h["windspeed_10m"][0],
        "wind_direction": h["winddirection_10m"][0],
        "rainfall":       h["precipitation"][0],
    }


def fetch_waves(lat, lon):
    url = (
        f"https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=wave_height,wave_direction,wave_period"
    )
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    h = resp.json()["hourly"]

    heights = h["wave_height"]
    current = next((v for v in heights[:6] if v is not None), None)
    if current is None:
        raise ValueError("No valid wave height in forecast")

    def get(i):
        v = heights[i] if i < len(heights) else None
        return v if v is not None else current

    return {
        "wave_height":    current,
        "wave_lag6":      get(6),
        "wave_lag12":     get(12),
        "wave_lag24":     get(24),
        "wave_direction": next((v for v in h["wave_direction"][:6] if v is not None), None),
        "wave_period":    next((v for v in h["wave_period"][:6]    if v is not None), None),
    }


def classify_risk(pred, wind_speed, rainfall, wave_height):
    if pred < 1:
        risk  = "Low";    color = "green"
        rec   = "Safe conditions. Fishing and coastal activities can continue."
    elif pred < 2:
        risk  = "Medium"; color = "orange"
        rec   = "Moderate risk. Small boats should operate with caution."
    else:
        risk  = "High";   color = "red"
        rec   = "High risk. Avoid going to sea and stay away from the shore."

    if wind_speed > 8:
        explanation = "High wind speed is increasing wave intensity."
    elif rainfall > 0:
        explanation = "Rainfall indicates unstable ocean conditions."
    else:
        explanation = "Moderate environmental conditions affecting waves."

    trend = (
        "Wave conditions are expected to worsen."
        if pred > wave_height
        else "Wave conditions are expected to improve."
    )
    return risk, color, rec, explanation, trend


@app.route("/predict_all", methods=["GET"])
def predict_all():
    import datetime
    now   = datetime.datetime.utcnow()
    month = now.month
    hour  = now.hour

    results = []

    for loc in LOCATIONS:
        lat, lon, name = loc["lat"], loc["lon"], loc["name"]

        try:
            weather = fetch_weather(lat, lon)
        except Exception as e:
            # results.append({"name": name, "lat": lat, "lon": lon,
            #                 "error": f"Weather fetch failed: {e}"})
            # continue
            print(f"Weather API failed for {name}: {e}")
            weather = {"wind_speed": 5, "wind_direction": 180, "rainfall": 0}

        try:
            waves = fetch_waves(lat, lon)
        except Exception as e:
            results.append({"name": name, "lat": lat, "lon": lon,
                            "error": f"Wave fetch failed: {e}"})
            continue

        # Build feature vector matching training features exactly
        all_features = {**weather, **waves, "month": month, "hour": hour}
        X = pd.DataFrame([{k: all_features.get(k) for k in model.feature_names_in_}])

        try:
            pred = float(model.predict(X)[0])
        except Exception as e:
            # results.append({"name": name, "lat": lat, "lon": lon,
            #                 "error": f"Prediction failed: {e}"})
            # continue
            print(f"Wave API failed for {name}: {e}")
            waves = {
                "wave_height": 1.0,
                "wave_lag6": 1.0,
                "wave_lag12": 1.0,
                "wave_lag24": 1.0,
                "wave_direction": 180,
                "wave_pok chneriod": 5
            }
        risk, color, rec, explanation, trend = classify_risk(
            pred, weather["wind_speed"], weather["rainfall"], waves["wave_height"]
        )

        results.append({
            "name":           name,
            "lat":            lat,
            "lon":            lon,
            "prediction":     pred,
            "wave_height":    waves["wave_height"],
            "wind_speed":     weather["wind_speed"],
            "wind_direction": weather["wind_direction"],
            "rainfall":       weather["rainfall"],
            "risk":           risk,
            "color":          color,
            "recommendation": rec,
            "explanation":    explanation,
            "trend":          trend,
        })

    return jsonify(results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)