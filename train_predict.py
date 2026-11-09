from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

DATA = Path("data/albuquerque_daily_weather.csv")
TARGET_DATE = pd.Timestamp("2026-11-20")

def make_features(df):
    d = pd.to_datetime(df["date"])
    return pd.DataFrame({
        "year": d.dt.year,
        "month": d.dt.month,
        "day": d.dt.day,
        "day_of_year": d.dt.dayofyear,
        "day_of_week": d.dt.dayofweek,
        "day_sin": __import__("numpy").sin(2 * __import__("numpy").pi * d.dt.dayofyear / 365.25),
        "day_cos": __import__("numpy").cos(2 * __import__("numpy").pi * d.dt.dayofyear / 365.25),
    }, index=df.index)

def f_to_c(f):
    return (f - 32) * 5 / 9

def main():
    if not DATA.exists():
        raise FileNotFoundError("Run `python fetch_data.py` first.")
    df = pd.read_csv(DATA, parse_dates=["date"]).dropna(subset=["high_f", "low_f"])
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 365:
        raise ValueError("Not enough historical data to train the baseline model.")

    X = make_features(df)
    split = int(len(df) * 0.8)  # chronological split; avoid random shuffling
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    metrics = {}
    models = {}
    for target in ("high_f", "low_f"):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=42, n_jobs=-1)
        )
        model.fit(X_train, df[target].iloc[:split])
        pred = model.predict(X_test)
        metrics[target] = mean_absolute_error(df[target].iloc[split:], pred)
        models[target] = model

    print("Chronological holdout MAE:")
    print(f"  High temperature: {metrics['high_f']:.2f} °F")
    print(f"  Low temperature:  {metrics['low_f']:.2f} °F")

    target_frame = pd.DataFrame({"date": [TARGET_DATE]})
    target_X = make_features(target_frame)
    high_f = float(models["high_f"].predict(target_X)[0])
    low_f = float(models["low_f"].predict(target_X)[0])

    result = pd.DataFrame([{
        "location": "Albuquerque, New Mexico",
        "target_date": TARGET_DATE.date().isoformat(),
        "predicted_high_f": round(high_f, 1),
        "predicted_high_c": round(f_to_c(high_f), 1),
        "predicted_low_f": round(low_f, 1),
        "predicted_low_c": round(f_to_c(low_f), 1),
        "holdout_mae_high_f": round(metrics["high_f"], 2),
        "holdout_mae_low_f": round(metrics["low_f"], 2),
        "model": "RandomForestRegressor baseline",
        "note": "Educational estimate, not an official forecast"
    }])
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    path = out / "prediction_2026-11-20.csv"
    result.to_csv(path, index=False)
    print("\\nPrediction:")
    print(result.to_string(index=False))
    print(f"\\nSaved: {path}")

if __name__ == "__main__":
    main()
