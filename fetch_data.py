from pathlib import Path
from io import StringIO
import requests
import pandas as pd

# Albuquerque International Sunport (ABQ), New Mexico ASOS station
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"
params = {
    "network": "NM_ASOS",
    "stations": "ABQ",
    "year1": 2000, "month1": 1, "day1": 1,
    "year2": 2025, "month2": 12, "day2": 31,
    "vars": "max_temp_f,min_temp_f",
    "what": "download",
    "delim": ",",
    "tz": "America/Denver",
}

def main():
    out = Path("data")
    out.mkdir(exist_ok=True)
    print("Downloading historical daily temperatures from IEM...")

    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    print("Columns received:", list(df.columns))

    lower = {c.lower().strip(): c for c in df.columns}
    date_col = next((lower[k] for k in ("valid", "date", "day") if k in lower), None)
    high_col = next((lower[k] for k in ("max_temp_f", "max", "high") if k in lower), None)
    low_col = next((lower[k] for k in ("min_temp_f", "min", "low") if k in lower), None)

    if not all([date_col, high_col, low_col]):
        raise ValueError(
            "Could not identify date/high/low columns. The data source may have "
            "changed its response format. Check the printed column names."
        )

    clean = df[[date_col, high_col, low_col]].copy()
    clean.columns = ["date", "high_f", "low_f"]
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["high_f"] = pd.to_numeric(clean["high_f"], errors="coerce")
    clean["low_f"] = pd.to_numeric(clean["low_f"], errors="coerce")
    clean = clean.dropna().drop_duplicates(subset="date").sort_values("date")
    clean = clean[
        clean["high_f"].between(-30, 130) & clean["low_f"].between(-50, 100)
    ]
    clean["high_c"] = (clean["high_f"] - 32) * 5 / 9
    clean["low_c"] = (clean["low_f"] - 32) * 5 / 9

    output_file = out / "albuquerque_daily_weather.csv"
    clean.to_csv(output_file, index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(clean):,} rows to {output_file}")
    print(clean.tail())

if __name__ == "__main__":
    main()
