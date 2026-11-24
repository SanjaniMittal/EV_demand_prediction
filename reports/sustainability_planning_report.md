# ⚡ Washington State EV Demand & Grid Infrastructure Planning Report
**A Comprehensive Machine Learning Forecast & Sustainability Strategy (2025–2035)**

---

## 1. Executive Summary
The State of Washington is currently experiencing one of the fastest electric vehicle (EV) transitions in North America. Analysis of over **300,000 vehicle registration records** from the Washington State Department of Licensing (WA DOL) reveals an exponential acceleration in adoption over the past decade. 

To support the State's **Clean Cars 2030** target and the statutory mandate for **100% zero-emission passenger vehicle sales by 2035**, electric utilities, municipal planners, and charging operators must coordinate massive capital deployment in transmission, distribution feeders, and fast-charging corridors.

### Core Projections:
- **2026 Fleet Baseline:** ~303,400 active electric vehicles registered in Washington (80.6% Battery Electric Vehicles [BEVs], 19.4% Plug-in Hybrids [PHEVs]).
- **2030 Fleet Milestone:** **507,661 cumulative EVs**, adding ~56,000 new EVs annually.
- **2035 Fleet Milestone:** **836,989 cumulative EVs**, adding ~73,000 new EVs annually.
- **Electrical Energy Requirement:** Washington's EV fleet will draw **1,868 GWh/year** by 2030 and **3,080 GWh/year** by 2035.
- **Peak Charging Capacity:** Unmitigated coincident peak load will reach **584 MW** by 2030 and **963 MW** by 2035.
- **Public Infrastructure Deficit:** By 2030, Washington will need at least **6,346 DC Fast Chargers (DCFC)** and **18,131 public Level 2 ports**.

---

## 2. Adoption Dynamics & Fleet Composition
Historical analysis of registration data across Model Years 2011 to 2026 highlights several structural trends:

```
Year      Annual EV Demand    Cumulative Fleet    BEV Share (%)    Daily Load (MWh)
2014                 3,099              12,419            70.5%                130.4
2017                 8,300              33,991            72.4%                356.9
2020                11,981              70,292            76.8%                738.1
2022                29,945             120,796            78.9%              1,268.4
2023                61,580             182,376            81.8%              1,914.9
2024                49,383             231,759            82.4%              2,433.5
2026                41,073             303,440            80.6%              3,186.1
```

### Key Technological Shifts:
1. **BEV Dominance over PHEV:** The share of pure battery electric vehicles has risen from 70% in 2014 to over 80% today, driven by rapid improvements in nominal battery range.
2. **Expansion of Median Driving Range:** Median BEV range expanded from ~73 miles in 2011 (early Nissan Leaf) to **272–310 miles** in 2024–2026 (Tesla Model 3/Y, Hyundai Ioniq 5, Ford Mach-E). This eliminates range anxiety for daily commuting but increases the peak power requirement during rapid DC charging.
3. **Automaker Market Concentration:**
   - **Tesla:** 40.8% market share (124,009 vehicles), led by Model Y (68,034) and Model 3 (39,065).
   - **Legacy Automakers Expanding:** Chevrolet (20,328), Ford (16,088), Nissan (16,064), Kia (15,159), Toyota (14,801), Hyundai (12,830), BMW (12,537), Rivian (10,349).

---

## 3. Geospatial Concentration & County Prioritization

EV adoption in Washington is heavily concentrated along the **Interstate-5 Western Washington Urban Corridor**:

| Priority Tier | County | Registered EVs | State Share (%) | Primary Utility | Infrastructure Priority |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Tier 1 (High Urban)** | King | 145,585 | 47.9% | Puget Sound Energy / Seattle City Light | Curbside Level 2, Multi-family housing, High-density DCFC hubs |
| **Tier 1 (High Urban)** | Snohomish | 38,260 | 12.6% | Snohomish County PUD / PSE | Suburban DCFC plazas, Commuter park-and-rides |
| **Tier 1 (High Urban)** | Pierce | 25,555 | 8.4% | Tacoma Public Utilities / PSE | Commercial retail hubs, I-5 corridor fast chargers |
| **Tier 2 (Regional Hub)**| Clark | 19,214 | 6.3% | Clark Public Utilities | Portland metro cross-border transit corridors |
| **Tier 2 (Regional Hub)**| Thurston | 11,202 | 3.7% | Puget Sound Energy | State capitol fleet electrification & regional hubs |
| **Tier 2 (Regional Hub)**| Kitsap | 10,547 | 3.5% | Puget Sound Energy | Ferry terminals & Peninsula transit links |
| **Tier 2 (Eastern WA)** | Spokane | 8,977 | 3.0% | Avista Corp / Inland Power | I-90 cross-state highway corridors, winter-range resilience |
| **Tier 3 (Secondary)**   | Whatcom | 7,652 | 2.5% | Puget Sound Energy | I-5 Canadian border cross-border corridor |
| **Tier 3 (Secondary)**   | Benton | 4,671 | 1.5% | Benton PUD / City of Richland | Tri-Cities regional charging hubs |

---

## 4. Machine Learning Forecast Horizon (2027–2035)

Using an ensemble blending Polynomial Trend Analysis, Gradient Boosting, and Logistic Diffusion (Bass S-Curve model), we project future annual adoption and cumulative fleet growth:

| Year | Annual New EVs (Moderate) | Annual New EVs (Aggressive) | Cumulative Fleet | Annual Grid GWh | Peak Grid Load (MW) | Public DCFC Ports Needed | Avoided $CO_2$ (MT/yr) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2027** | 47,252 | 59,065 | 348,787 | 1,283.5 | 401.1 | 4,360 | 1,325,391 |
| **2028** | 50,022 | 62,528 | 398,809 | 1,467.6 | 458.6 | 4,985 | 1,515,474 |
| **2029** | 52,916 | 66,145 | 451,725 | 1,662.3 | 519.5 | 5,647 | 1,716,555 |
| **2030** | **55,936** | **69,920** | **507,661** | **1,868.2** | **583.8** | **6,346** | **1,929,112** |
| **2031** | 59,079 | 73,849 | 566,740 | 2,085.6 | 651.8 | 7,084 | 2,153,612 |
| **2032** | 62,348 | 77,935 | 629,088 | 2,315.0 | 723.5 | 7,864 | 2,390,534 |
| **2033** | 65,741 | 82,176 | 694,829 | 2,557.0 | 799.1 | 8,685 | 2,640,350 |
| **2034** | 69,259 | 86,574 | 764,088 | 2,811.8 | 878.7 | 9,551 | 2,903,534 |
| **2035** | **72,901** | **91,126** | **836,989** | **3,080.1** | **962.5** | **10,462** | **3,180,558** |

---

## 5. Strategic Recommendations for Utilities and Planners

### 1. Mandatory Smart Charging & Time-of-Use (TOU) Rates
- If unmanaged, charging 500,000 EVs in 2030 could create **584 MW** of coincident peak power, severely straining evening sub-transmission substations.
- Puget Sound Energy, Seattle City Light, and Tacoma Power should expand automated, telemetry-linked off-peak charging incentives (11 PM - 6 AM) to shift charging to overnight base-load periods.

### 2. Multi-Family Housing (MUD) Charging Equity
- In King County, nearly 45% of residents live in multi-family apartment complexes without dedicated garages. 
- State grant programs must prioritize Level 2 curbside and communal parking infrastructure to prevent an adoption plateau among urban renters.

### 3. Highway Corridor Fast Charging (NEVI Alignment)
- Deploy National Electric Vehicle Infrastructure (NEVI) formula funds along I-5, I-90, US-2, and US-395 with minimum 150 kW - 350 kW DC fast chargers spaced every 50 miles, ensuring travel reliability between Western and Eastern Washington.

### 4. Fleet Electrification & Commercial Medium/Heavy-Duty Sizing
- While passenger cars require ~10-15 kWh/day, commercial delivery vans and transit buses will demand 80-250 kWh/day. Distribution feeders serving industrial corridors (Kent Valley, Spokane Industrial Park) should begin transformer upgrades now.
