import pandas as pd

print("Loading datasets...")

weather = pd.read_csv("data/weather.csv")
wave    = pd.read_csv("data/wave.csv")

weather["time"] = pd.to_datetime(weather["time"])
wave["time"]    = pd.to_datetime(wave["time"])

print(f"Weather rows: {len(weather)} | Locations: {weather['location'].unique().tolist()}")
print(f"Wave rows:    {len(wave)}    | Locations: {wave['location'].unique().tolist()}")

merged_parts = []

for loc_name in weather["location"].unique():
    w  = weather[weather["location"] == loc_name].sort_values("time")
    wv = wave[wave["location"] == loc_name].sort_values("time")

    if len(wv) == 0:
        print(f"WARNING: no wave data for {loc_name}, skipping.")
        continue

    merged = pd.merge_asof(
        w,
        wv[["time", "wave_height", "wave_direction", "wave_period"]],
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("2h"),
    )

    bad = merged["wave_height"].isna().sum()
    if bad > 0:
        print(f"  {loc_name}: dropping {bad} rows with no wave match within 2h")

    merged_parts.append(merged)
    print(f"  {loc_name}: {len(merged)} rows")

df = pd.concat(merged_parts, ignore_index=True)

before = len(df)
df = df.dropna()
print(f"\nDropped {before - len(df)} rows with NaN. Remaining: {len(df)}")

df = df.sort_values(["location", "time"]).reset_index(drop=True)
df.to_csv("data/final_data.csv", index=False)

print(f"\nSaved data/final_data.csv")
print(f"Columns: {df.columns.tolist()}")
print(df.head())