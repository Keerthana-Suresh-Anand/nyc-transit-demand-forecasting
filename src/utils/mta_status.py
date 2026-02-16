import requests
import pandas as pd
from datetime import datetime


def get_mta_api_latest_date():
    DATASET_ID = "5wq4-mkjj"
    url = f"https://data.ny.gov/resource/{DATASET_ID}.json"

    # USE 'AS' TO FIX THE NAME: This is the industry standard way
    params = {"$select": "max(transit_timestamp) AS latest_timestamp"}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Now we know EXACTLY what the key is: 'latest_timestamp'
        if data and "latest_timestamp" in data[0]:
            raw_date = data[0]["latest_timestamp"]
            latest_date = pd.to_datetime(raw_date).strftime("%Y-%m-%d")
            print(f"Latest date available on MTA API: {latest_date}")
            return latest_date
        else:
            print(f"API returned unexpected format. Data received: {data}")
            return None

    except Exception as e:
        print(f"Error fetching from API: {e}")
        return None


if __name__ == "__main__":
    get_mta_api_latest_date()
