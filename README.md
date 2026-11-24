# ⚡ Electric Vehicle (EV) Demand Prediction & Adoption Analysis
### Washington State Department of Licensing (WA DOL) Dataset
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B.svg)](https://ev-demandprediction.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook%20Executed-orange.svg)](notebooks/EV_Demand_Prediction_Analysis.ipynb)

> 🚀 **Live Interactive Web Application:** Explore the live dashboard, forecast scenarios, and charging infrastructure calculator at **[https://ev-demandprediction.streamlit.app/](https://ev-demandprediction.streamlit.app/)**

---

## 📌 Project Overview
This repository delivers an end-to-end data science, machine learning forecasting, and sustainability planning system analyzing **over 300,000 real-world electric vehicle registration records** from the Washington State Department of Licensing ([data.wa.gov](https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2)).

The project directly supports public policy, municipal utility operations, and grid infrastructure planning for the **Washington Clean Cars 2030** benchmark and the statutory mandate for **100% Zero-Emission Vehicles by 2035**.

### Key Deliverables:
- **Live Interactive Web Dashboard** ([ev-demandprediction.streamlit.app](https://ev-demandprediction.streamlit.app/)): Multi-tab interactive simulation platform for scenario exploration, county drill-downs, and electrical grid sizing.
- **Comprehensive Jupyter Notebook** ([notebooks/EV_Demand_Prediction_Analysis.ipynb](notebooks/EV_Demand_Prediction_Analysis.ipynb)): 100% pre-executed with markdown explanations, exploratory figures, and model benchmarks.
- **Machine Learning Forecasting Engine** (`src/models.py`): Non-linear curve fitting, Polynomial Regression, Ridge, Random Forest, Gradient Boosting, and Bass Logistic Diffusion models.
- **Sustainability & Infrastructure Sizer**: Quantifies annual electricity consumption (GWh), coincident peak grid power (MW), public DCFC / Level 2 charging ports needed, and avoided $CO_2$ emissions.
- **CLI Query Tool** (`src/predict.py`): Instant forecast lookups for any year (2027–2035) or Washington county.

---

## 🗂️ Project Architecture

```
EV demand prediction/
├── data/
│   └── Electric_Vehicle_Population_Data.csv   # Official Washington State EV records (~304k rows)
├── notebooks/
│   └── EV_Demand_Prediction_Analysis.ipynb    # Pre-executed Jupyter Notebook with full analysis
├── src/
│   ├── __init__.py
│   ├── data_loader.py                         # Data ingestion and local cache verification
│   ├── preprocessing.py                       # Cleaning, coordinate parsing, range imputation
│   ├── features.py                            # Time series, trend terms, lag, and panel engineering
│   ├── eda.py                                 # Publication-grade chart generation
│   ├── models.py                              # ML algorithms, training, evaluation, forecasting
│   ├── plot_forecast.py                       # 2035 forecast and infrastructure requirement plots
│   └── predict.py                             # Interactive CLI forecast query interface
├── figures/                                   # High-resolution (300 DPI) publication figures
│   ├── 01_ev_growth_trend.png
│   ├── 02_market_share_makes_models.png
│   ├── 03_county_geographic_distribution.png
│   ├── 04_electric_range_progression.png
│   ├── 05_electric_utility_demand.png
│   ├── 06_geospatial_heatmap.png
│   ├── 07_ev_demand_forecast_2035.png
│   └── 08_sustainability_infrastructure_plan.png
├── reports/
│   └── sustainability_planning_report.md      # Strategic policy and utility grid roadmap
├── app.py                                     # Interactive Streamlit Web Application
├── requirements.txt                           # Project dependencies
└── README.md                                  # Portfolio documentation
```

---

## 🔍 Key Findings & Adoption Patterns

1. **Explosive Adoption Growth:** Washington's electric vehicle fleet expanded from **12,419 vehicles in 2014** to **over 303,400 active vehicles today**, demonstrating clear exponential S-curve diffusion.
2. **BEV vs. PHEV Transition:** Pure Battery Electric Vehicles (BEVs) represent **80.6%** of registrations, while Plug-in Hybrids (PHEVs) account for **19.4%**.
3. **Automaker Leadership:**
   - **Tesla:** 40.8% market share (124,009 vehicles), led by the Model Y (68,034) and Model 3 (39,065).
   - **Top Competitors:** Chevrolet (20,328), Ford (16,088), Nissan (16,064), Kia (15,159), Toyota (14,801), Hyundai (12,830), BMW (12,537), Rivian (10,349).
4. **Geographic Concentration:**
   - **King County:** Hosts **145,585 EVs (47.9%)** of the entire state fleet.
   - **Top 3 Urban Counties:** King, Snohomish (12.6%), and Pierce (8.4%) contain nearly **70%** of all Washington EVs.
5. **Driving Range Expansion:** Median BEV range expanded from **73 miles in 2011** to **272–310 miles in modern platforms**, shifting infrastructure focus from highway range anxiety to urban curbside/overnight charging availability.

---

## 🤖 Machine Learning Model Benchmarks

Models were evaluated using a time-series split (Train: 2011–2022, Test: 2023–2026):

| Model Algorithm | RMSE (Annual Additions) | MAE | MAPE (%) | Key Characteristic |
| :--- | :---: | :---: | :---: | :--- |
| **Polynomial Regression (Deg 2)** | **19,144** | **13,092** | **23.2%** | **Best Out-of-Sample Performance** |
| **Gradient Boosting Regressor** | 19,685 | 17,254 | 34.1% | Captures non-linear adoption momentum |
| **Logistic S-Curve Diffusion** | 20,049 | 17,221 | 33.7% | Bass Diffusion saturation dynamics |
| **Polynomial Regression (Deg 3)** | 20,266 | 14,106 | 25.3% | Flexible higher-order acceleration |
| **Random Forest Regressor** | 22,494 | 20,402 | 41.0% | Tree ensemble with conservative plateau |
| **Linear Regression** | 23,469 | 20,363 | 40.1% | Underfits non-linear acceleration |

*County-Level Geospatial Forecaster:* A Random Forest Regressor trained on panel county features achieves **$R^2 = 0.9836$** with a Mean Absolute Error of only **66.8 vehicles/county/year**.

---

## 📈 Long-Range Projections & Sustainability Plan (2027–2035)

| Target Year | Projected Annual New EVs | Cumulative Fleet Size | Annual Energy Demand | Peak Charging Grid Power | Public DCFC Ports Needed | Avoided $CO_2$ Emissions |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2027** | 47,252 | 348,787 | 1,283.5 GWh/yr | 401.1 MW | 4,360 ports | 1.33M MT/yr |
| **2028** | 50,022 | 398,809 | 1,467.6 GWh/yr | 458.6 MW | 4,985 ports | 1.52M MT/yr |
| **2029** | 52,916 | 451,725 | 1,662.3 GWh/yr | 519.5 MW | 5,647 ports | 1.72M MT/yr |
| **2030 (Benchmark)**| **55,936** | **507,661** | **1,868.2 GWh/yr** | **583.8 MW** | **6,346 ports** | **1.93M MT/yr** |
| **2035 (ZEV Mandate)**| **72,901** | **836,989** | **3,080.1 GWh/yr** | **962.5 MW** | **10,462 ports** | **3.18M MT/yr** |

---

## 🚀 How to Run the Project

### 🌐 1. Access the Live Web Application
The fastest way to explore the project is via the live deployment hosted on Streamlit Community Cloud:  
👉 **[https://ev-demandprediction.streamlit.app/](https://ev-demandprediction.streamlit.app/)**

### 💻 2. Local Setup & Installation
Ensure Python 3.10+ is installed. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the Interactive Streamlit Dashboard Locally
Launch the web interface locally:

```bash
streamlit run app.py
```

### 4. Query the CLI Forecaster
Query projected EV adoption metrics for any year or Washington county:

```bash
# Statewide projection for 2030
python -m src.predict --year 2030

# County projection for King County in 2030
python -m src.predict --year 2030 --county King

# County projection for Spokane County in 2032
python -m src.predict --year 2032 --county Spokane
```

### 4. Re-generate Figures & Features
Run the full modular processing pipeline:

```bash
# Data inspection and cleaning
python -m src.preprocessing

# Exploratory data visualization (saves to figures/)
python -m src.eda

# Train and benchmark forecasting models
python -m src.models

# Render long-range 2035 forecast plots
python -m src.plot_forecast
```

### 5. Inspect the Executed Jupyter Notebook
Open `notebooks/EV_Demand_Prediction_Analysis.ipynb` in VS Code, JupyterLab, or GitHub to review all inline executed charts, statistical tables, and methodology write-ups.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
