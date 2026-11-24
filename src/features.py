"""
Feature Engineering Module
Prepares aggregated time-series and tabular feature matrices for EV demand forecasting and geospatial modeling.
"""

import numpy as np
import pandas as pd


def create_statewide_timeseries(df: pd.DataFrame, start_year: int = 2011, end_year: int = 2026) -> pd.DataFrame:
    """
    Creates statewide annual time-series features for EV adoption forecasting.
    Includes linear/polynomial trends, lag features, moving averages, and growth rates.
    """
    # Filter years
    filtered = df[(df["model_year"] >= start_year) & (df["model_year"] <= end_year)].copy()

    # Aggregate counts by model year and EV type
    agg = filtered.groupby(["model_year", "ev_type"]).size().unstack(fill_value=0)
    
    if "BEV" not in agg.columns:
        agg["BEV"] = 0
    if "PHEV" not in agg.columns:
        agg["PHEV"] = 0

    ts = pd.DataFrame(index=agg.index)
    ts["year"] = ts.index
    ts["bev_count"] = agg["BEV"]
    ts["phev_count"] = agg["PHEV"]
    ts["annual_ev_demand"] = agg["BEV"] + agg["PHEV"]
    ts["bev_share"] = ts["bev_count"] / ts["annual_ev_demand"]

    # Cumulative fleet size (installed base of EVs demanding charging)
    ts["cumulative_ev_fleet"] = ts["annual_ev_demand"].cumsum()

    # Time trend variables (t = 0 at start_year)
    ts["t"] = ts["year"] - start_year
    ts["t_squared"] = ts["t"] ** 2
    ts["t_cubed"] = ts["t"] ** 3

    # Log-transformed demand for exponential growth modeling
    ts["log_annual_demand"] = np.log1p(ts["annual_ev_demand"])
    ts["log_cumulative_fleet"] = np.log1p(ts["cumulative_ev_fleet"])

    # Year-over-Year Growth Rate
    ts["yoy_growth_rate"] = ts["annual_ev_demand"].pct_change().fillna(0)

    # Lag features
    ts["lag_demand_1"] = ts["annual_ev_demand"].shift(1).bfill()
    ts["lag_demand_2"] = ts["annual_ev_demand"].shift(2).bfill()

    # Rolling window statistics
    ts["rolling_mean_2yr"] = ts["annual_ev_demand"].rolling(2, min_periods=1).mean()
    ts["rolling_mean_3yr"] = ts["annual_ev_demand"].rolling(3, min_periods=1).mean()

    # Estimated charging energy requirement (Assuming 10.5 kWh/vehicle/day on average)
    ts["est_daily_mwh_demand"] = (ts["cumulative_ev_fleet"] * 10.5) / 1000.0

    return ts.reset_index(drop=True)


def create_county_timeseries(df: pd.DataFrame, start_year: int = 2013, end_year: int = 2026, min_total_evs: int = 200) -> pd.DataFrame:
    """
    Creates panel/time-series dataset of EV demand by Washington County over time.
    """
    filtered = df[(df["model_year"] >= start_year) & (df["model_year"] <= end_year) & (df["county"] != "UNKNOWN")].copy()
    
    # Filter counties with at least min_total_evs
    county_totals = filtered["county"].value_counts()
    valid_counties = county_totals[county_totals >= min_total_evs].index.tolist()
    filtered = filtered[filtered["county"].isin(valid_counties)]

    # Group by county and year
    panel = filtered.groupby(["county", "model_year"]).agg(
        annual_demand=("vin", "count"),
        bev_count=("is_bev", "sum"),
        avg_range=("estimated_range", "mean"),
    ).reset_index()

    panel["phev_count"] = panel["annual_demand"] - panel["bev_count"]
    panel["bev_share"] = panel["bev_count"] / panel["annual_demand"]
    panel["t"] = panel["model_year"] - start_year

    # Sort and calculate cumulative fleet per county
    panel = panel.sort_values(["county", "model_year"]).reset_index(drop=True)
    panel["cumulative_fleet"] = panel.groupby("county")["annual_demand"].cumsum()
    panel["lag_demand_1"] = panel.groupby("county")["annual_demand"].shift(1).fillna(0)
    panel["lag_demand_2"] = panel.groupby("county")["annual_demand"].shift(2).fillna(0)
    panel["yoy_growth"] = panel.groupby("county")["annual_demand"].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    panel["rolling_mean_2yr"] = panel.groupby("county")["annual_demand"].transform(lambda x: x.rolling(2, min_periods=1).mean())

    # Daily MWh electricity demand per county
    panel["daily_mwh_load"] = (panel["cumulative_fleet"] * 10.5) / 1000.0

    return panel


def create_utility_timeseries(df: pd.DataFrame, start_year: int = 2013, end_year: int = 2026) -> pd.DataFrame:
    """
    Creates time-series dataset of EV demand by primary Electric Utility service area.
    """
    filtered = df[(df["model_year"] >= start_year) & (df["model_year"] <= end_year)].copy()
    
    panel = filtered.groupby(["primary_utility", "model_year"]).agg(
        annual_demand=("vin", "count"),
        bev_count=("is_bev", "sum"),
    ).reset_index()

    panel = panel.sort_values(["primary_utility", "model_year"]).reset_index(drop=True)
    panel["cumulative_fleet"] = panel.groupby("primary_utility")["annual_demand"].cumsum()
    panel["daily_mwh_load"] = (panel["cumulative_fleet"] * 10.5) / 1000.0
    
    return panel


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_ev_data

    raw = load_raw_data()
    clean = clean_ev_data(raw)
    
    ts_state = create_statewide_timeseries(clean)
    print("Statewide Time Series (Tail 5):")
    print(ts_state[["year", "annual_ev_demand", "cumulative_ev_fleet", "bev_share", "yoy_growth_rate", "est_daily_mwh_demand"]].tail(5))
    
    ts_county = create_county_timeseries(clean)
    print(f"\nCounty Panel shape: {ts_county.shape}, Counties covered: {ts_county['county'].nunique()}")
