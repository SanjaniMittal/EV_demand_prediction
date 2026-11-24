"""
Exploratory Data Analysis (EDA) Module
Generates high-resolution visualizations of EV adoption patterns, trends, and grid impacts in Washington State.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.labelweight"] = "semibold"
plt.rcParams["figure.dpi"] = 300

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_annual_growth_trend(df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """Plot annual registrations and cumulative EV fleet growth."""
    ensure_dir(output_dir)
    
    # Filter stable historic to recent model years (2011 to 2026)
    yr_df = df[(df["model_year"] >= 2011) & (df["model_year"] <= 2026)]
    
    pivot_yr = pd.crosstab(yr_df["model_year"], yr_df["ev_type"])
    pivot_yr["Total"] = pivot_yr.sum(axis=1)
    pivot_yr["Cumulative"] = pivot_yr["Total"].cumsum()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = pivot_yr.index
    width = 0.65
    
    # Stacked bars for BEV and PHEV
    bev_vals = pivot_yr["BEV"] if "BEV" in pivot_yr.columns else pd.Series(0, index=x)
    phev_vals = pivot_yr["PHEV"] if "PHEV" in pivot_yr.columns else pd.Series(0, index=x)
    
    b1 = ax1.bar(x, bev_vals, width, label="Battery Electric Vehicle (BEV)", color="#1f77b4", alpha=0.9)
    b2 = ax1.bar(x, phev_vals, width, bottom=bev_vals, label="Plug-in Hybrid (PHEV)", color="#2ca02c", alpha=0.9)
    
    ax1.set_xlabel("Model Year")
    ax1.set_ylabel("Annual Vehicle Registrations (New EV Fleet)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x, rotation=45)
    
    # Dual axis for cumulative adoption curve
    ax2 = ax1.twinx()
    ax2.plot(x, pivot_yr["Cumulative"], color="#d62728", marker="o", linewidth=2.5, label="Cumulative EV Fleet")
    ax2.set_ylabel("Cumulative EV Registrations", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False)
    
    # Annotate total on recent years
    for yr in [2021, 2022, 2023, 2024, 2025, 2026]:
        if yr in pivot_yr.index:
            tot = pivot_yr.loc[yr, "Total"]
            ax1.annotate(f"{tot:,}", (yr, tot), textcoords="offset points", xytext=(0, 6),
                         ha='center', fontsize=9, fontweight='bold', color="#111111")
    
    plt.title("Washington State EV Adoption Growth Trend (2011 - 2026)\nAnnual Registrations & Cumulative Fleet Size", pad=15)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "01_ev_growth_trend.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_top_makes_and_models(df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """Plot market share of top EV manufacturers and top vehicle models."""
    ensure_dir(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Top 10 Makes
    top_makes = df["make"].value_counts().head(10)
    colors_makes = sns.color_palette("Blues_r", len(top_makes))
    sns.barplot(x=top_makes.values, y=top_makes.index, palette=colors_makes, ax=axes[0])
    axes[0].set_title("Top 10 EV Manufacturers in Washington")
    axes[0].set_xlabel("Total Vehicles Registered")
    axes[0].set_ylabel("Make")
    for i, v in enumerate(top_makes.values):
        pct = (v / len(df)) * 100
        axes[0].text(v + 1000, i, f"{v:,} ({pct:.1f}%)", va="center", fontsize=9, fontweight="bold")
    axes[0].set_xlim(0, top_makes.max() * 1.2)
    
    # 2. Top 10 Models
    model_counts = df.groupby(["make", "model"]).size().sort_values(ascending=False).head(10)
    model_labels = [f"{m[0]} {m[1]}" for m in model_counts.index]
    colors_models = sns.color_palette("viridis", len(model_counts))
    sns.barplot(x=model_counts.values, y=model_labels, palette=colors_models, ax=axes[1])
    axes[1].set_title("Top 10 Electric Vehicle Models")
    axes[1].set_xlabel("Total Vehicles Registered")
    axes[1].set_ylabel("Model")
    for i, v in enumerate(model_counts.values):
        pct = (v / len(df)) * 100
        axes[1].text(v + 500, i, f"{v:,} ({pct:.1f}%)", va="center", fontsize=9, fontweight="bold")
    axes[1].set_xlim(0, model_counts.max() * 1.2)
    
    plt.suptitle("Washington State EV Market Dominance: Manufacturers & Models", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "02_market_share_makes_models.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_county_distribution(df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """Plot geographical concentration of EV adoption across top Washington counties."""
    ensure_dir(output_dir)
    
    top_counties = df[df["county"] != "UNKNOWN"]["county"].value_counts().head(12)
    county_pct = (top_counties / len(df)) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#003f5c" if i < 3 else "#2f4b7c" if i < 6 else "#665191" for i in range(len(top_counties))]
    
    bars = ax.barh(top_counties.index[::-1], top_counties.values[::-1], color=colors[::-1], height=0.65)
    ax.set_title("Top 12 Washington Counties by EV Adoption Count", pad=15)
    ax.set_xlabel("Number of Registered Electric Vehicles")
    ax.set_ylabel("County")
    
    for bar, pct, count in zip(bars, county_pct.values[::-1], top_counties.values[::-1]):
        width = bar.get_width()
        ax.text(width + 1500, bar.get_y() + bar.get_height()/2, f"{count:,} ({pct:.1f}%)",
                va="center", ha="left", fontsize=9, fontweight="bold")
        
    ax.set_xlim(0, top_counties.max() * 1.22)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "03_county_geographic_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_electric_range_evolution(df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """Plot evolution of vehicle electric range over model years."""
    ensure_dir(output_dir)
    
    # Filter to model years with significant volume and non-zero / estimated range
    range_df = df[(df["model_year"] >= 2012) & (df["model_year"] <= 2026)].copy()
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    # Median estimated range per year by EV type
    agg_range = range_df.groupby(["model_year", "ev_type"])["estimated_range"].median().unstack()
    
    ax.plot(agg_range.index, agg_range["BEV"], marker="o", color="#1f77b4", linewidth=3, label="BEV Median Range (miles)")
    if "PHEV" in agg_range.columns:
        ax.plot(agg_range.index, agg_range["PHEV"], marker="s", color="#2ca02c", linewidth=2.5, label="PHEV Median Electric Range (miles)")
        
    ax.axhline(y=300, color="gray", linestyle="--", alpha=0.7, label="300-Mile Long Range Benchmark")
    
    ax.set_title("Evolution of Electric Driving Range in Washington Fleet (2012 - 2026)", pad=15)
    ax.set_xlabel("Model Year")
    ax.set_ylabel("Median Electric Range (Miles)")
    ax.set_xticks(agg_range.index)
    ax.set_xticklabels(agg_range.index, rotation=45)
    ax.legend(loc="center right", frameon=True)
    
    # Annotate milestone
    bev_latest = agg_range["BEV"].iloc[-1]
    ax.annotate(f"Current BEV Median: ~{bev_latest:.0f} miles",
                xy=(agg_range.index[-1], bev_latest),
                xytext=(-100, 25), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
                fontweight="bold", color="#1f77b4")
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "04_electric_range_progression.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_utility_demand_impact(df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """Plot EV concentration and estimated charging energy burden by primary utility."""
    ensure_dir(output_dir)
    
    util_counts = df["primary_utility"].value_counts()
    
    # Calculate estimated daily MWh charging load assuming 10 kWh/day avg consumption per EV
    util_kwh = df.groupby("primary_utility")["battery_kwh_est"].count() * 10.0 / 1000.0  # MWh/day
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar 1: Vehicle Counts
    colors = sns.color_palette("mako", len(util_counts))
    sns.barplot(x=util_counts.values, y=util_counts.index, palette=colors, ax=ax1)
    ax1.set_title("EV Fleet by Electric Utility Service Area")
    ax1.set_xlabel("Number of Registered EVs")
    ax1.set_ylabel("Utility Provider")
    for i, v in enumerate(util_counts.values):
        ax1.text(v + 1500, i, f"{v:,}", va="center", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, util_counts.max() * 1.25)
    
    # Bar 2: Estimated Daily Grid Energy Burden
    util_kwh_sorted = util_kwh.loc[util_counts.index]
    sns.barplot(x=util_kwh_sorted.values, y=util_kwh_sorted.index, palette=colors, ax=ax2)
    ax2.set_title("Estimated Daily Charging Energy Burden (MWh/day)")
    ax2.set_xlabel("Estimated Daily Demand (MWh/day)")
    ax2.set_ylabel("")
    for i, v in enumerate(util_kwh_sorted.values):
        ax2.text(v + 15, i, f"{v:,.0f} MWh/d", va="center", fontsize=9, fontweight="bold")
    ax2.set_xlim(0, util_kwh_sorted.max() * 1.25)
    
    plt.suptitle("Grid Infrastructure Impact: Utility Territory Demand in Washington", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "05_electric_utility_demand.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_geospatial_adoption(df: pd.DataFrame, output_dir: str = FIGURES_DIR):
    """Plot geospatial scatter distribution of EVs across Washington coordinates."""
    ensure_dir(output_dir)
    
    geo_df = df.dropna(subset=["latitude", "longitude"]).copy()
    # Filter roughly to WA state bounding box
    geo_df = geo_df[(geo_df["latitude"].between(45.5, 49.0)) & (geo_df["longitude"].between(-125.0, -116.8))]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    scatter = ax.scatter(
        geo_df["longitude"],
        geo_df["latitude"],
        c=geo_df["is_bev"],
        cmap="coolwarm",
        alpha=0.15,
        s=4,
        rasterized=True
    )
    
    ax.set_title("Geospatial Distribution of Electric Vehicles in Washington State\n(Blue: PHEV, Red: BEV)", pad=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # Annotate key metropolitan hubs
    cities = {
        "Seattle / Bellevue": (-122.33, 47.60),
        "Tacoma": (-122.44, 47.25),
        "Olympia": (-122.90, 47.04),
        "Vancouver (Clark Co.)": (-122.66, 45.63),
        "Spokane": (-117.42, 47.66),
        "Tri-Cities (Benton/Franklin)": (-119.14, 46.22),
        "Bellingham": (-122.48, 48.75),
    }
    for city, (lon, lat) in cities.items():
        ax.plot(lon, lat, marker="*", color="black", markersize=9)
        ax.annotate(city, (lon, lat), textcoords="offset points", xytext=(8, -3),
                    fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.7))
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, "06_geospatial_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def run_all_eda(df: pd.DataFrame):
    """Run full suite of exploratory data visualizations."""
    print("Generating EDA visualizations...")
    plot_annual_growth_trend(df)
    plot_top_makes_and_models(df)
    plot_county_distribution(df)
    plot_electric_range_evolution(df)
    plot_utility_demand_impact(df)
    plot_geospatial_adoption(df)
    print("All EDA charts generated successfully.")


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_ev_data
    raw = load_raw_data()
    clean = clean_ev_data(raw)
    run_all_eda(clean)
