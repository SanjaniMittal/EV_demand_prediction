"""
Data Loader Module for Washington State Electric Vehicle Population Data
"""

import os
import pandas as pd
import requests

DATA_URL = "https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD"
LOCAL_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Electric_Vehicle_Population_Data.csv")


def ensure_dataset_exists(csv_path: str = LOCAL_CSV_PATH) -> str:
    """Ensure the dataset CSV is available locally; downloads if missing."""
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1024 * 1024:
        return csv_path
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    print(f"Downloading Washington State EV Population data to {csv_path}...")
    response = requests.get(DATA_URL, stream=True, timeout=60)
    response.raise_for_status()
    with open(csv_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print("Download completed successfully.")
    return csv_path


def load_raw_data(csv_path: str = LOCAL_CSV_PATH) -> pd.DataFrame:
    """Load raw CSV data into a pandas DataFrame."""
    path = ensure_dataset_exists(csv_path)
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df):,} records with columns: {list(df.columns)}")
