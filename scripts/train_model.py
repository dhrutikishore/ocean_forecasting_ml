import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("Loading final dataset...")

df = pd.read_csv("data/final_data.csv")
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values(["location", "time"]).reset_index(drop=True)

print(f"Total rows: {len(df)}")
print(f"Locations:  {df['location'].unique().tolist()}")

# Time features
df["month"] = df["time"].dt.month
df["hour"]  = df["time"].dt.hour

# Target: wave height 24 hours ahead
df["target"] = df.groupby("location")["wave_height"].shift(-24)

# Split BEFORE lag features — no leakage
train_parts, test_parts = [], []
for loc_name, group in df.groupby("location"):
    group = group.sort_values("time").reset_index(drop=True)
    split_idx = int(len(group) * 0.8)
    train_parts.append(group.iloc[:split_idx])
    test_parts.append(group.iloc[split_idx:])

train_df = pd.concat(train_parts, ignore_index=True)
test_df  = pd.concat(test_parts,  ignore_index=True)
print(f"After split — Train: {len(train_df)}  |  Test: {len(test_df)}")

# Lag features AFTER split, per location
def add_lags(df_split):
    df_split = df_split.copy()
    for loc_name, group in df_split.groupby("location"):
        idx = group.index
        df_split.loc[idx, "wave_lag6"]  = group["wave_height"].shift(6).values
        df_split.loc[idx, "wave_lag12"] = group["wave_height"].shift(12).values
        df_split.loc[idx, "wave_lag24"] = group["wave_height"].shift(24).values
    return df_split

train_df = add_lags(train_df)
test_df  = add_lags(test_df)
train_df = train_df.dropna()
test_df  = test_df.dropna()
print(f"After dropna — Train: {len(train_df)}  |  Test: {len(test_df)}")

FEATURES = [
    "wind_speed", "wind_direction", "rainfall",
    "wave_lag6", "wave_lag12", "wave_lag24",
    "wave_direction", "wave_period",
    "month", "hour",
]
FEATURES = [f for f in FEATURES if f in train_df.columns]
print(f"Features: {FEATURES}")

X_train, y_train = train_df[FEATURES], train_df["target"]
X_test,  y_test  = test_df[FEATURES],  test_df["target"]

print("\nTraining...")
model = RandomForestRegressor(
    n_estimators=200, max_depth=15,
    min_samples_split=10, random_state=42, n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nOverall — MAE: {mae:.4f} m  RMSE: {rmse:.4f} m  R2: {r2:.4f}")

print("\nPer-location:")
for loc_name in test_df["location"].unique():
    mask = test_df["location"] == loc_name
    lp   = model.predict(X_test[mask])
    la   = y_test[mask]
    print(f"  {loc_name:<12}  MAE: {mean_absolute_error(la,lp):.4f} m   R2: {r2_score(la,lp):.4f}")

print("\nFeature importances:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:<20} {imp:.3f}  {'█' * int(imp * 50)}")

joblib.dump(model, "models/ocean_model.pkl")
print("\nSaved models/ocean_model.pkl")