import pandas as pd

raw_mta_path = "data/01_raw/sample_raw_mta.csv"
raw_weather_path = "data/01_raw/sample_raw_weather.csv"

mta_data = pd.read_csv(raw_mta_path)
weather_data = pd.read_csv(raw_weather_path)

print(mta_data.head())
