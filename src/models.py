"""
Machine Learning Forecasting and Sustainability Planning Models
Includes:
- Multiple regression & time-series models for statewide EV demand (Polynomial, Ridge, Random Forest, Gradient Boosting, S-Curve/Logistic).
- Model evaluation (R2, RMSE, MAE, MAPE).
- Future forecasting up to 2035.
- County-level demand prediction.
- Sustainability impact conversion (MWh electricity, peak MW, public chargers needed, avoided CO2).
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# S-Curve / Logistic Diffusion Function: L / (1 + exp(-k * (t - t0)))
def logistic_growth(t, L, k, t0):
    return L / (1.0 + np.exp(-np.clip(k * (t - t0), -50, 50)))


def evaluate_forecast_models(ts_df: pd.DataFrame, train_end_year: int = 2022):
    """
    Trains and evaluates multiple regression and machine learning models on historical split.
    Train: start_year to train_end_year
    Test: (train_end_year + 1) to max_year
    """
    train_mask = ts_df["year"] <= train_end_year
    test_mask = ts_df["year"] > train_end_year

    train_data = ts_df[train_mask]
    test_data = ts_df[test_mask]

    X_train_t = train_data[["t"]].values
    y_train = train_data["annual_ev_demand"].values

    X_test_t = test_data[["t"]].values
    y_test = test_data["annual_ev_demand"].values

    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=0.01))
        ]),
        "Polynomial Regression (Deg 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2)),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0))
        ]),
        "Polynomial Regression (Deg 3)": Pipeline([
            ("poly", PolynomialFeatures(degree=3)),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=10.0))
        ]),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_t, y_train)
        pred_test = model.predict(X_test_t)
        pred_train = model.predict(X_train_t)

        r2 = r2_score(y_test, pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        mae = mean_absolute_error(y_test, pred_test)
        mape = mean_absolute_percentage_error(y_test, pred_test) * 100

        results.append({
            "Model": name,
            "R2_Score": round(r2, 4),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "MAPE (%)": round(mape, 2)
        })
        trained_models[name] = model

    # Fit Logistic S-Curve on Cumulative Fleet (Diffusion Model)
    try:
        t_all = ts_df["t"].values
        cum_train = train_data["cumulative_ev_fleet"].values
        # Initial guess: L=2,000,000, k=0.35, t0=14
        popt, _ = curve_fit(logistic_growth, train_data["t"].values, cum_train,
                            p0=[1500000, 0.35, 14], maxfev=10000,
                            bounds=([100000, 0.05, 5], [5000000, 1.5, 30]))
        pred_cum_test = logistic_growth(test_data["t"].values, *popt)
        # Annual demand is delta of cumulative
        pred_cum_all = logistic_growth(t_all, *popt)
        pred_annual_logistic = np.diff(np.insert(pred_cum_all, 0, 0))[test_mask]
        
        r2_log = r2_score(y_test, pred_annual_logistic)
        rmse_log = np.sqrt(mean_squared_error(y_test, pred_annual_logistic))
        mae_log = mean_absolute_error(y_test, pred_annual_logistic)
        mape_log = mean_absolute_percentage_error(y_test, pred_annual_logistic) * 100

        results.append({
            "Model": "Logistic S-Curve Diffusion",
            "R2_Score": round(r2_log, 4),
            "RMSE": round(rmse_log, 2),
            "MAE": round(mae_log, 2),
            "MAPE (%)": round(mape_log, 2)
        })
        trained_models["Logistic S-Curve"] = popt
    except Exception as e:
        print(f"Logistic fitting notice: {e}")

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    return results_df, trained_models


def train_production_forecaster(ts_df: pd.DataFrame):
    """
    Fits production forecasting models on full historical data (2011-2026) to forecast 2027-2035.
    Returns future projections table with confidence bounds and sustainability impacts.
    """
    X_full_t = ts_df[["t"]].values
    y_full = ts_df["annual_ev_demand"].values
    start_year = int(ts_df["year"].min())

    # Fit ensemble model: weighted blend of Polynomial (Deg 2) and Gradient Boosting
    poly_model = Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=2.0))
    ])
    poly_model.fit(X_full_t, y_full)

    gbr_model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.07, max_depth=3, random_state=42)
    gbr_model.fit(X_full_t, y_full)

    rf_model = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42)
    rf_model.fit(X_full_t, y_full)

    # Future timeline: 2027 to 2035
    future_years = np.arange(2027, 2036)
    t_future = (future_years - start_year).reshape(-1, 1)

    poly_preds = poly_model.predict(t_future)
    gbr_preds = gbr_model.predict(t_future)
    rf_preds = rf_model.predict(t_future)

    # Moderate consensus scenario
    moderate_forecast = 0.50 * poly_preds + 0.30 * gbr_preds + 0.20 * rf_preds
    # Ensure non-decreasing reasonable bounds
    moderate_forecast = np.maximum(moderate_forecast, ts_df["annual_ev_demand"].iloc[-1] * 0.95)

    # Conservative scenario (slower macroeconomic adoption)
    conservative_forecast = moderate_forecast * 0.82

    # Aggressive scenario (Clean Cars 2030 WA Mandate accelerating ZEVs)
    aggressive_forecast = moderate_forecast * 1.25

    # Build Forecast DataFrame
    future_df = pd.DataFrame({
        "year": future_years,
        "forecast_moderate_annual": np.round(moderate_forecast).astype(int),
        "forecast_conservative_annual": np.round(conservative_forecast).astype(int),
        "forecast_aggressive_annual": np.round(aggressive_forecast).astype(int),
    })

    # Historical fleet base
    curr_cumulative = ts_df["cumulative_ev_fleet"].iloc[-1]

    future_df["cumulative_fleet_moderate"] = curr_cumulative + future_df["forecast_moderate_annual"].cumsum()
    future_df["cumulative_fleet_conservative"] = curr_cumulative + future_df["forecast_conservative_annual"].cumsum()
    future_df["cumulative_fleet_aggressive"] = curr_cumulative + future_df["forecast_aggressive_annual"].cumsum()

    # Sustainability & Infrastructure Planning Calculations
    # 1. Annual Electricity Demand (GWh/year):
    # Fleet * 11,500 miles/yr * 0.32 kWh/mile / 1,000,000
    future_df["annual_electricity_gwh"] = np.round((future_df["cumulative_fleet_moderate"] * 11500 * 0.32) / 1e6, 1)

    # 2. Peak Charging Grid Capacity (MW):
    # Coincident peak factor ~ 1.15 kW per active EV
    future_df["peak_charging_grid_mw"] = np.round((future_df["cumulative_fleet_moderate"] * 1.15) / 1000.0, 1)

    # 3. Public Charging Infrastructure Requirement:
    # 1 DC Fast Charger per 80 EVs, 1 Public Level 2 Port per 28 EVs (NREL standard)
    future_df["dcfc_chargers_needed"] = np.round(future_df["cumulative_fleet_moderate"] / 80).astype(int)
    future_df["level2_public_ports_needed"] = np.round(future_df["cumulative_fleet_moderate"] / 28).astype(int)

    # 4. Avoided Carbon Emissions (Metric Tons of CO2 per year):
    # Washington clean grid vs ~4.6 MT CO2/yr gasoline vehicle displacement: net savings ~3.8 MT CO2/EV/yr
    future_df["avoided_co2_metric_tons"] = np.round(future_df["cumulative_fleet_moderate"] * 3.8).astype(int)

    # Save models
    joblib.dump(poly_model, os.path.join(MODELS_DIR, "poly_model.pkl"))
    joblib.dump(gbr_model, os.path.join(MODELS_DIR, "gbr_model.pkl"))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "rf_model.pkl"))

    return future_df, {"poly": poly_model, "gbr": gbr_model, "rf": rf_model}


def train_county_demand_forecaster(panel_df: pd.DataFrame):
    """
    Trains county-level demand forecasting model to project localized EV adoption across Washington.
    """
    feature_cols = ["t", "lag_demand_1", "lag_demand_2", "rolling_mean_2yr"]
    
    clean_panel = panel_df.dropna(subset=feature_cols + ["annual_demand"]).copy()
    
    X = clean_panel[feature_cols]
    y = clean_panel["annual_demand"]

    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)

    r2 = r2_score(y, model.predict(X))
    mae = mean_absolute_error(y, model.predict(X))
    print(f"County Forecaster Train R2: {r2:.4f}, MAE: {mae:.2f}")

    joblib.dump(model, os.path.join(MODELS_DIR, "county_forecaster.pkl"))
    return model


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.features import create_county_timeseries, create_statewide_timeseries
    from src.preprocessing import clean_ev_data

    raw = load_raw_data()
    clean = clean_ev_data(raw)
    
    ts_state = create_statewide_timeseries(clean)
    eval_df, _ = evaluate_forecast_models(ts_state)
    print("=== MODEL EVALUATION BENCHMARK (Train 2011-2022, Test 2023-2026) ===")
    print(eval_df.to_string(index=False))

    future_df, _ = train_production_forecaster(ts_state)
    print("\n=== WASHINGTON STATE EV DEMAND & SUSTAINABILITY FORECAST (2027 - 2035) ===")
    print(future_df[["year", "forecast_moderate_annual", "cumulative_fleet_moderate", "annual_electricity_gwh", "peak_charging_grid_mw", "dcfc_chargers_needed", "avoided_co2_metric_tons"]].to_string(index=False))

    panel_county = create_county_timeseries(clean)
    county_model = train_county_demand_forecaster(panel_county)
