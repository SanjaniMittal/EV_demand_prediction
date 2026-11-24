"""
Prediction and Forecast Query CLI Tool
Allows users and planners to query statewide and county EV demand projections.
"""

import argparse
import os
import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import clean_ev_data
from src.features import create_statewide_timeseries, create_county_timeseries
from src.models import train_production_forecaster, train_county_demand_forecaster


def get_forecast(target_year: int = 2030, county: str = None):
    """Query projected EV adoption metrics for a target year and optional county."""
    raw = load_raw_data()
    clean = clean_ev_data(raw)
    ts_state = create_statewide_timeseries(clean)
    future_df, _ = train_production_forecaster(ts_state)

    if county:
        county_upper = county.strip().upper()
        panel = create_county_timeseries(clean)
        county_data = panel[panel["county"] == county_upper]
        if county_data.empty:
            print(f"County '{county}' not found. Available counties include: {sorted(panel['county'].unique()[:10])}...")
            return
        
        # Estimate county share of state growth
        latest_county_fleet = county_data["cumulative_fleet"].iloc[-1]
        latest_state_fleet = ts_state["cumulative_ev_fleet"].iloc[-1]
        county_ratio = latest_county_fleet / latest_state_fleet

        row = future_df[future_df["year"] == target_year]
        if row.empty:
            print(f"Target year {target_year} out of forecast range (2027-2035).")
            return
        
        state_cum = row["cumulative_fleet_moderate"].values[0]
        state_ann = row["forecast_moderate_annual"].values[0]
        county_cum = int(round(state_cum * county_ratio))
        county_ann = int(round(state_ann * county_ratio))
        county_dcfc = int(round(county_cum / 80))
        county_l2 = int(round(county_cum / 28))
        county_mwh = round(county_cum * 10.5 / 1000.0, 1)

        print(f"\n==================================================")
        print(f"  EV DEMAND PROJECTION: {county_upper} COUNTY ({target_year})")
        print(f"==================================================")
        print(f"  Projected Annual New EVs:       {county_ann:,}")
        print(f"  Projected Cumulative Fleet:     {county_cum:,}")
        print(f"  Share of Washington Fleet:      {county_ratio*100:.2f}%")
        print(f"  Daily Charging Electricity:     {county_mwh:,.1f} MWh/day")
        print(f"  DC Fast Chargers Needed (Ports):{county_dcfc:,}")
        print(f"  Level 2 Public Ports Needed:    {county_l2:,}")
        print(f"==================================================\n")
    else:
        row = future_df[future_df["year"] == target_year]
        if row.empty:
            print(f"Target year {target_year} out of forecast range (2027-2035).")
            return
        
        data = row.iloc[0]
        print(f"\n==================================================")
        print(f"  WASHINGTON STATEWIDE EV PROJECTION ({target_year})")
        print(f"==================================================")
        print(f"  Annual New EV Demand (Moderate):   {data['forecast_moderate_annual']:,}")
        print(f"  Annual New EV Demand (Cons.):      {data['forecast_conservative_annual']:,}")
        print(f"  Annual New EV Demand (Aggressive): {data['forecast_aggressive_annual']:,}")
        print(f"  Cumulative EV Fleet:               {data['cumulative_fleet_moderate']:,}")
        print(f"  Annual Grid Electricity Demand:    {data['annual_electricity_gwh']:,} GWh/yr")
        print(f"  Peak Charging Grid Capacity:       {data['peak_charging_grid_mw']:,} MW")
        print(f"  DC Fast Chargers Needed:           {data['dcfc_chargers_needed']:,} ports")
        print(f"  Level 2 Public Ports Needed:       {data['level2_public_ports_needed']:,} ports")
        print(f"  Avoided CO2 Greenhouse Emissions:  {data['avoided_co2_metric_tons']:,} MT/yr")
        print(f"==================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query EV Demand and Grid Projections")
    parser.add_argument("--year", type=int, default=2030, help="Target projection year (2027 - 2035)")
    parser.add_argument("--county", type=str, default=None, help="Specific Washington county name (e.g., King, Snohomish)")
    args = parser.parse_args()
    get_forecast(args.year, args.county)
