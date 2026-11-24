"""
Preprocessing and Cleaning Module for Washington State EV Data
"""

import re
import numpy as np
import pandas as pd


# Known nominal battery ranges for top BEV/PHEV models where WA DOL records range as 0
KNOWN_MODEL_RANGES = {
    ("TESLA", "MODEL Y"): 310,
    ("TESLA", "MODEL 3"): 272,
    ("TESLA", "MODEL S"): 402,
    ("TESLA", "MODEL X"): 335,
    ("FORD", "MUSTANG MACH-E"): 260,
    ("VOLKSWAGEN", "ID.4"): 263,
    ("HYUNDAI", "IONIQ 5"): 266,
    ("KIA", "EV6"): 274,
    ("RIVIAN", "R1T"): 314,
    ("RIVIAN", "R1S"): 316,
    ("CHEVROLET", "BOLT EV"): 259,
    ("CHEVROLET", "BOLT EUV"): 247,
    ("CHEVROLET", "BLAZER EV"): 279,
    ("CHEVROLET", "EQUINOX EV"): 319,
    ("BMW", "I4"): 295,
    ("BMW", "IX"): 307,
    ("NISSAN", "ARIYA"): 289,
    ("TOYOTA", "BZX4"): 242,
    ("TOYOTA", "RAV4 PRIME (PHEV)"): 42,
    ("TOYOTA", "PRIUS PRIME"): 38,
    ("JEEP", "WRANGLER (PHEV)"): 21,
    ("JEEP", "GRAND CHEROKEE (PHEV)"): 25,
    ("CHRYSLER", "PACIFICA (PHEV)"): 32,
    ("VOLVO", "XC90 (PHEV)"): 35,
    ("VOLVO", "XC60 (PHEV)"): 35,
}


def parse_vehicle_location(loc_str):
    """
    Extracts (latitude, longitude) from strings like 'POINT (-122.329075 47.601815)'.
    Returns (np.nan, np.nan) if missing or unparseable.
    """
    if not isinstance(loc_str, str):
        return np.nan, np.nan
    match = re.search(r"POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)", loc_str)
    if match:
        lon = float(match.group(1))
        lat = float(match.group(2))
        return lat, lon
    return np.nan, np.nan


def clean_ev_data(df: pd.DataFrame, filter_wa_only: bool = True) -> pd.DataFrame:
    """
    Performs full data cleaning, missing value handling, feature parsing, and type standardizations.
    """
    df = df.copy()

    # Standardize column names
    col_map = {
        "VIN (1-10)": "vin",
        "County": "county",
        "City": "city",
        "State": "state",
        "Postal Code": "postal_code",
        "Model Year": "model_year",
        "Make": "make",
        "Model": "model",
        "Electric Vehicle Type": "ev_type_raw",
        "Clean Alternative Fuel Vehicle (CAFV) Eligibility": "cafv_eligibility_raw",
        "Electric Range": "electric_range",
        "Legislative District": "legislative_district",
        "DOL Vehicle ID": "dol_vehicle_id",
        "Vehicle Location": "vehicle_location",
        "Electric Utility": "electric_utility",
        "2020 GEOID": "geoid_2020",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Standardize string fields
    str_cols = ["county", "city", "state", "make", "model", "electric_utility"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})

    # Filter to Washington State if requested (State == 'WA')
    if filter_wa_only and "state" in df.columns:
        df = df[df["state"] == "WA"].copy()

    # Simplify EV Type: 'BEV' vs 'PHEV'
    def simplify_ev_type(val):
        if not isinstance(val, str):
            return "UNKNOWN"
        if "Battery Electric Vehicle" in val:
            return "BEV"
        elif "Plug-in Hybrid" in val:
            return "PHEV"
        return "OTHER"

    df["ev_type"] = df["ev_type_raw"].apply(simplify_ev_type)
    df["is_bev"] = (df["ev_type"] == "BEV").astype(int)

    # Clean CAFV eligibility
    def simplify_cafv(val):
        if not isinstance(val, str):
            return "Unknown"
        if "Clean Alternative Fuel Vehicle Eligible" in val:
            return "Eligible"
        elif "Not eligible due to low battery range" in val:
            return "Not Eligible"
        elif "Eligibility unknown" in val or "not been researched" in val:
            return "Unresearched"
        return "Unknown"

    df["cafv_status"] = df["cafv_eligibility_raw"].apply(simplify_cafv)

    # Parse Latitude and Longitude from vehicle_location
    if "vehicle_location" in df.columns:
        lat_lon = df["vehicle_location"].apply(parse_vehicle_location)
        df["latitude"] = [x[0] for x in lat_lon]
        df["longitude"] = [x[1] for x in lat_lon]
    else:
        df["latitude"] = np.nan
        df["longitude"] = np.nan

    # Fill missing geographical and categorical fields
    df["county"] = df["county"].fillna("UNKNOWN")
    df["city"] = df["city"].fillna("UNKNOWN")
    df["electric_utility"] = df["electric_utility"].fillna("UNKNOWN")

    # Clean Model Year: discard extreme outliers (keep years 2010 to current + 2)
    # The modern EV era began with the Nissan Leaf and Chevy Volt around 2011
    df = df[(df["model_year"] >= 2010) & (df["model_year"] <= 2027)].copy()

    # Estimated / Imputed Electric Range
    # Many modern BEVs have range recorded as 0 in DOL because DOL didn't test/research them
    df["range_is_zero_or_missing"] = (df["electric_range"].fillna(0) == 0).astype(int)

    # Impute known ranges for popular models with 0 recorded range
    def impute_range(row):
        recorded = row["electric_range"]
        if pd.notnull(recorded) and recorded > 0:
            return recorded
        make = row.get("make", "")
        model = row.get("model", "")
        key = (make, model)
        if key in KNOWN_MODEL_RANGES:
            return KNOWN_MODEL_RANGES[key]
        # Fallback default estimates by EV type
        if row["ev_type"] == "BEV":
            return 250.0  # Modern BEV avg range
        elif row["ev_type"] == "PHEV":
            return 32.0   # Modern PHEV avg electric range
        return 0.0

    df["estimated_range"] = df.apply(impute_range, axis=1)

    # Primary Electric Utility Simplification
    # WA utilities often have long concatenated names; extract primary provider
    def simplify_utility(util_str):
        if not isinstance(util_str, str) or util_str == "UNKNOWN":
            return "Other / Unknown"
        if "PUGET SOUND ENERGY" in util_str:
            return "Puget Sound Energy"
        elif "SEATTLE CITY LIGHT" in util_str:
            return "Seattle City Light"
        elif "SNOHOMISH COUNTY" in util_str:
            return "Snohomish County PUD"
        elif "TACOMA" in util_str:
            return "Tacoma Public Utilities"
        elif "CLARK PUBLIC UTILITIES" in util_str:
            return "Clark Public Utilities"
        elif "AVISTA" in util_str:
            return "Avista Corp"
        elif "INLAND POWER" in util_str:
            return "Inland Power & Light"
        elif "PACIFIC POWER" in util_str:
            return "Pacific Power"
        return "Other PUD / Municipal"

    df["primary_utility"] = df["electric_utility"].apply(simplify_utility)

    # Battery Capacity Estimation (approx kWh for energy demand modeling)
    # Average ~0.28 kWh per mile of range for BEV, ~0.35 kWh per mile for PHEV
    def estimate_battery_kwh(row):
        rng = row["estimated_range"]
        if row["ev_type"] == "BEV":
            return min(max(rng * 0.28, 30.0), 130.0)  # BEVs typically 30 - 130 kWh
        else:
            return min(max(rng * 0.35, 8.0), 25.0)   # PHEVs typically 8 - 25 kWh

    df["battery_kwh_est"] = df.apply(estimate_battery_kwh, axis=1)

    return df


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    raw_df = load_raw_data()
    clean_df = clean_ev_data(raw_df)
    print(f"Cleaned dataset shape: {clean_df.shape}")
    print(f"EV Type Breakdown:\n{clean_df['ev_type'].value_counts(normalize=True)*100}")
    print(f"Primary Utilities:\n{clean_df['primary_utility'].value_counts()}")
