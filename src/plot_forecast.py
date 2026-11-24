"""
Forecast and Sustainability Visualizations
Generates publication-quality charts for projected EV adoption and infrastructure planning through 2035.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.features import create_statewide_timeseries, create_county_timeseries
from src.models import train_production_forecaster, train_county_demand_forecaster

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")


def plot_forecast_scenarios(ts_df: pd.DataFrame, future_df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """Plot historical and projected EV adoption scenarios to 2035 with confidence bands."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Annual EV Demand Projections ---
    ax1.plot(ts_df["year"], ts_df["annual_ev_demand"], marker="o", color="#1f77b4", linewidth=2.5, label="Historical Registrations (2011-2026)")
    
    # Forecast Scenarios
    ax1.plot(future_df["year"], future_df["forecast_moderate_annual"], marker="s", color="#ff7f0e", linewidth=2.5, linestyle="--", label="Moderate Scenario (Consensus ML)")
    ax1.plot(future_df["year"], future_df["forecast_aggressive_annual"], marker="^", color="#2ca02c", linewidth=2, linestyle=":", label="Aggressive Scenario (Clean Cars 2030 Mandate)")
    ax1.plot(future_df["year"], future_df["forecast_conservative_annual"], marker="v", color="#d62728", linewidth=2, linestyle="-.", label="Conservative Scenario (Macroeconomic Dampening)")

    # Fill between confidence band
    ax1.fill_between(future_df["year"], future_df["forecast_conservative_annual"], future_df["forecast_aggressive_annual"],
                     color="#ff7f0e", alpha=0.15, label="Forecast Uncertainty Range")

    ax1.axvline(x=2026.5, color="gray", linestyle="--", alpha=0.7)
    ax1.text(2026.7, ax1.get_ylim()[1] * 0.88, "Forecast Horizon (2027-2035)", color="#444444", fontsize=10, fontstyle="italic")

    ax1.set_title("Washington State Annual EV Demand Forecast (2011 - 2035)", pad=15)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual EV Registrations")
    ax1.set_xticks(np.arange(2012, 2036, 3))
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    # --- Plot 2: Cumulative Fleet Growth & Mandate Targets ---
    hist_cum = ts_df[["year", "cumulative_ev_fleet"]]
    
    ax2.plot(hist_cum["year"], hist_cum["cumulative_ev_fleet"] / 1000.0, marker="o", color="#1f77b4", linewidth=2.5, label="Historical Cumulative Fleet")
    ax2.plot(future_df["year"], future_df["cumulative_fleet_moderate"] / 1000.0, marker="s", color="#ff7f0e", linewidth=2.5, linestyle="--", label="Projected Fleet (Moderate)")
    ax2.fill_between(future_df["year"], future_df["cumulative_fleet_conservative"] / 1000.0, future_df["cumulative_fleet_aggressive"] / 1000.0,
                     color="#ff7f0e", alpha=0.15)

    # Policy targets annotations
    ax2.scatter([2030], [500], color="#d62728", s=100, zorder=5, label="WA 2030 Clean Cars Benchmark (500k EVs)")
    ax2.annotate("2030 Goal (~500k EVs)", (2030, 500), textcoords="offset points", xytext=(-70, 15),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5), fontweight="bold", color="#d62728")

    ax2.set_title("Cumulative Washington EV Fleet Projection (Thousands)", pad=15)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Cumulative Active EVs (Thousands)")
    ax2.set_xticks(np.arange(2012, 2036, 3))
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    plt.suptitle("Washington State Long-Range EV Market Projections & Sustainability Policy Alignment", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "07_ev_demand_forecast_2035.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_infrastructure_requirements(future_df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """Plot projected electrical grid load and public charging infrastructure requirements."""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Grid Load: Annual GWh and Peak MW
    years = future_df["year"]
    
    color1 = "#1f77b4"
    ax1.plot(years, future_df["annual_electricity_gwh"], marker="o", color=color1, linewidth=2.5, label="Annual Electricity Consumption (GWh)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual Electricity Demand (GWh/year)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax1_twin = ax1.twinx()
    color2 = "#e377c2"
    ax1_twin.plot(years, future_df["peak_charging_grid_mw"], marker="^", color=color2, linewidth=2.5, linestyle="--", label="Peak Charging Grid Capacity (MW)")
    ax1_twin.set_ylabel("Coincident Peak Grid Demand (MW)", color=color2)
    ax1_twin.tick_params(axis="y", labelcolor=color2)
    ax1_twin.grid(False)

    ax1.set_title("Projected Electric Grid Load for Washington State EV Fleet", pad=15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    # Public Charging Ports Requirements
    ax2.bar(years - 0.2, future_df["dcfc_chargers_needed"], width=0.4, label="DC Fast Chargers (DCFC) Required", color="#d62728", alpha=0.85)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("DC Fast Chargers Required (Ports)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(years, future_df["level2_public_ports_needed"], marker="s", color="#2ca02c", linewidth=2.5, label="Level 2 Public Ports Required")
    ax2_twin.set_ylabel("Public Level 2 Ports Required", color="#2ca02c")
    ax2_twin.tick_params(axis="y", labelcolor="#2ca02c")
    ax2_twin.grid(False)

    ax2.set_title("Projected Public Charging Infrastructure Expansion Needs", pad=15)
    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc="upper left", frameon=True)

    plt.suptitle("Washington State EV Infrastructure & Grid Planning (2027 - 2035)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "08_sustainability_infrastructure_plan.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_ev_data

    raw = load_raw_data()
    clean = clean_ev_data(raw)
    ts_state = create_statewide_timeseries(clean)
    future_df, _ = train_production_forecaster(ts_state)
    
    plot_forecast_scenarios(ts_state, future_df)
    plot_infrastructure_requirements(future_df)
