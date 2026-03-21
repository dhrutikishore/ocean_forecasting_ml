import requests
import pandas as pd
import time

print("Fetching wave data...")

START_DATE = "2019-01-01"
END_DATE   = "2023-01-01"

LOCATIONS = [
    {"name": "Puri",     "lat": 19.84, "lon": 85.82},
    {"name": "Gopalpur", "lat": 19.26, "lon": 84.91},
    {"name": "Chennai",  "lat": 13.08, "lon": 80.27},
    {"name": "Mumbai",   "lat": 19.07, "lon": 72.87},
]

all_dfs = []

for loc in LOCATIONS:
    print(f"\nFetching {loc['name']}...")
    url = (
        f"https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={loc['lat']}&longitude={loc['lon']}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        f"&hourly=wave_height,wave_direction,wave_period"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "hourly" not in data:
            print(f"  ERROR: no hourly data — {data}")
            continue
        df = pd.DataFrame({
            "time":           data["hourly"]["time"],
            "wave_height":    data["hourly"]["wave_height"],
            "wave_direction": data["hourly"]["wave_direction"],
            "wave_period":    data["hourly"]["wave_period"],
        })
        df["location"] = loc["name"]
        df["time"] = pd.to_datetime(df["time"])
        df = df[df["wave_height"].notna()]
        df = df[(df["wave_height"] >= 0) & (df["wave_height"] < 20)]
        df = df.sort_values("time").reset_index(drop=True)
        print(f"  Rows: {len(df)}")
        all_dfs.append(df)
    except Exception as e:
        print(f"  FAILED: {e}")
    time.sleep(1)

if not all_dfs:
    print("ERROR: nothing fetched.")
    exit()

combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv("data/wave.csv", index=False)
print(f"\nSaved data/wave.csv — {len(combined)} rows")
print(combined.groupby("location").size())