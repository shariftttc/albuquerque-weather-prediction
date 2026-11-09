# Albuquerque Weather Prediction

Predict daily high and low temperatures for Albuquerque, New Mexico, using historical daily observations.

## Important
This is an educational baseline model, not an official weather forecast. A prediction for November 20, 2026 made on November 8 has substantial uncertainty. For operational decisions, use the National Weather Service forecast.

## Data source
The training-data script downloads daily observations for Albuquerque International Sunport (ABQ) from the Iowa Environmental Mesonet (IEM) ASOS daily archive. Check the source and data coverage before relying on results.

## Setup (Windows)
```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python fetch_data.py
python train_predict.py
```

## What it does
- Downloads daily historical high/low temperature data.
- Creates calendar features and day-of-year seasonal features.
- Trains separate Random Forest regressors for daily high and low.
- Evaluates using a chronological holdout set.
- Predicts high/low temperature for 2026-11-20 in Fahrenheit and Celsius.
- Saves predictions to `outputs/prediction_2026-11-20.csv`.

## Dataset for Kaggle
After running `python fetch_data.py`, upload `data/albuquerque_daily_weather.csv` to Kaggle as a dataset. Do not upload API keys or private information. See Kaggle's dataset creation guide: https://www.kaggle.com/docs/datasets
