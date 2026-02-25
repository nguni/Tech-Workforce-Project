# Tech Employment EDA & Forecasting

## Overview

This project performs exploratory data analysis (EDA) and workforce forecasting on tech industry employment data spanning 2000–2025. It covers data quality checks, trend visualizations, predictive modeling with linear regression, and time-series forecasting using Facebook Prophet.

## Project Structure

```
├── data/
│   └── tech_employment_2000_2025.csv
├── notebooks/
│   └── forecast.ipynb
├── models/
│   ├── prophet_basic.pkl
│   ├── prophet_covid.pkl
│   ├── prophet_full.pkl
│   ├── linear_regression_random_split.pkl
│   ├── linear_regression_time_split.pkl
│   ├── evaluation_metrics.json
│   └── forecasts/
│       ├── forecast_baseline.csv
│       ├── forecast_optimistic.csv
│       ├── forecast_pessimistic.csv
│       ├── forecast_10yr.csv
│       └── forecast_15yr.csv
└── README.md
```

## Dataset

**File:** `data/tech_employment_2000_2025.csv`

The dataset tracks annual employment figures for multiple tech companies with the following columns:

| Column | Description |
|---|---|
| `company` | Company name |
| `year` | Year of record |
| `employees_start` / `employees_end` | Headcount at start and end of year |
| `new_hires` / `layoffs` | Annual hiring and layoff counts |
| `net_change` | Net workforce change (hires − layoffs) |
| `hiring_rate_pct` / `attrition_rate_pct` | Percentage rates for hiring and attrition |
| `revenue_billions_usd` | Annual revenue in billions USD |
| `stock_price_change_pct` | Year-over-year stock price change |
| `gdp_growth_us_pct` | US GDP growth rate |
| `unemployment_rate_us_pct` | US unemployment rate |
| `is_estimated` | Whether values are estimated |
| `confidence_level` | Confidence label (e.g., Medium, High) |
| `data_quality_score` | Numeric data quality score |

## Notebook Structure

### 1. Setup & Data Loading
All imports are consolidated at the top. Data is loaded from `../data/tech_employment_2000_2025.csv` relative to the notebook.

### 2. Data Inspection
Structure, data types, missing values, and summary statistics are examined using `df.info()`, `df.isnull().sum()`, and `df.describe()`.

### 3. Exploratory Data Analysis
Aggregate industry trends visualised over time: total new hires, total layoffs, net workforce change, and the top 10 companies by both hires and layoffs.

### 4. Feature Engineering
Three derived features are created: `net_change` (hires − layoffs), `growth_rate` (net change relative to starting headcount), and `month` (extracted from year, used as a model feature).

### 5. Linear Regression
Two models are trained to predict `net_change` using company-level features. A random train/test split (80/20) provides a baseline R², and a time-based split (train ≤ 2018, test > 2018) gives a more realistic measure of out-of-sample performance.

### 6. Industry-Level Aggregation
Company-level data is aggregated by year to produce an industry-wide annual `net_change` series along with mean GDP growth and unemployment rate, which feed into the Prophet models.

### 7. Prophet Forecasting
Three progressively richer Prophet models are built — a basic model with no regressors, a COVID-adjusted model with a binary 2020–2022 flag, and a full model incorporating GDP growth, unemployment rate, and the COVID flag. The full model is evaluated on a held-out post-2018 test set using MAE.

### 8. Scenario Forecasting
Three 5-year forecast scenarios are generated from the full model: baseline (historical average economics), optimistic (GDP 4.5%, unemployment 3.5%), and pessimistic (GDP 0.5%, unemployment 7.0%).

### 9. Extended Forecasts
Forecasts are extended to 10- and 15-year horizons under baseline assumptions using a shared `plot_forecast()` helper function.

### 10. Save Models & Artifacts
All trained models, forecast CSVs, and evaluation metadata are saved to the `models/` directory.

---

## Findings

**Hiring trends followed macro cycles closely.** Tech hiring peaked in the early 2000s bull market, contracted sharply after the dot-com bust, recovered through the 2010s, and surged aggressively in 2020–2021 as companies bet on sustained remote-work demand. The 2022–2023 period saw a sharp reversal with layoffs spiking to some of the highest levels in the dataset.

**The COVID period (2020–2022) was a clear outlier.** Industry-wide net workforce change during this period deviated significantly from what macroeconomic indicators alone would predict, justifying the inclusion of the `covid_period` binary regressor. Without it, models overfit to this anomaly and produced distorted forecasts.

**Linear regression had limited predictive power on the time-based split.** The random split produced a misleadingly high R² because future data leaked into training. When evaluated properly on post-2018 data, performance dropped considerably, indicating that company-level features like revenue and stock price change are not sufficient on their own to predict net workforce change across different macro regimes.

**GDP growth and unemployment were meaningful regressors for Prophet.** Adding economic indicators to the Prophet model improved forecast coherence, particularly for the post-2018 test period. The MAE on the held-out set was noticeably lower with the full model than with the basic model.

**Scenario forecasts diverged significantly over 5 years.** The gap between the optimistic and pessimistic scenarios widened substantially even over a short horizon, highlighting how sensitive tech hiring is to macroeconomic conditions. The pessimistic scenario (GDP 0.5%, unemployment 7.0%) projected a sustained contraction in net headcount growth.

**Long-horizon uncertainty is large.** At 10 and 15 years out, the Prophet confidence intervals widened to the point where the forecast range encompassed both significant growth and contraction. These extended forecasts are best interpreted as directional indicators rather than precise predictions.

---

## Lessons Learned

**Always use time-based splits for time-series data.** Random splits leak future information into training and produce inflated evaluation metrics. The gap between the random-split and time-split R² scores in this project illustrates the issue clearly.

**Reuse variable names carefully in notebooks.** The original notebook reused `model` and `prophet_df` across multiple cells, which caused `NameError` failures in the save cell and made it impossible to compare models. Using descriptive names (`model_basic`, `model_covid`, `model_full`) throughout prevents this and makes the notebook reproducible end-to-end.

**Create directories before writing files.** Pandas and pickle will raise `OSError` if the target directory doesn't exist. Using `Path(...).mkdir(parents=True, exist_ok=True)` before any file write operations is a simple safeguard.

**Import everything at the top.** Scattering imports across cells makes notebooks fragile — cells fail if run out of order. A single setup cell with all imports makes the notebook more robust and easier to share.

**Prophet regressors must be provided at prediction time.** Any regressor added during training must also be present in the future dataframe. For the COVID flag this means explicitly setting `future['covid_period'] = 0` for post-training periods, otherwise Prophet will throw an error.

**Refactor repeated code into functions.** The original notebook had three nearly identical forecast plot blocks. Replacing them with a single `plot_forecast()` helper reduced duplication and made it easy to add new forecast horizons without copying and pasting.

---

---
---

## Industry Applications

**Workforce Planning & Budgeting**
HR and finance teams can use the scenario forecasts (optimistic, baseline, pessimistic) to set annual headcount budgets tied to macroeconomic outlooks. Rather than planning in a vacuum, decisions on hiring freezes or expansion targets can be anchored to GDP and unemployment projections that companies already track for other business purposes.

**Recession & Downturn Preparedness**
The pessimistic scenario gives companies an early-warning signal. If leading economic indicators trend toward low GDP growth and high unemployment, leadership has a quantified estimate of likely industry-wide contraction and can respond with measured slowdowns rather than reactive mass layoffs.

**Competitive Benchmarking**
Because the dataset covers multiple companies, a business can compare its own hiring and attrition rates against industry aggregates year by year. Growing headcount during a period of industry-wide contraction either signals competitive advantage or unsustainable risk — the model helps frame which.

**Investor & Analyst Signaling**
Equity analysts covering tech can use the scenario forecasts as a cross-check on company guidance. If a company projects aggressive hiring under pessimistic macro conditions, that is a flag worth investigating. The industry-level net change trend also serves as a useful leading indicator of sector-wide productivity and capex expectations.

**Policy & Labor Market Research**
Government labor agencies and researchers can use this model to quantify how sensitive tech employment is to macroeconomic levers. The relationship between GDP growth, unemployment rate, and tech sector headcount is difficult to see from aggregate employment figures alone, and having it modelled explicitly supports more informed policy decisions.

**Limitations**
This model is best used as a directional planning tool rather than a precise predictor. Wide confidence intervals at 10–15 year horizons, a relatively small dataset of annual observations, and the manually coded COVID regressor all mean forecasts should inform decisions rather than drive them. Predictive power would improve meaningfully with quarterly data, a broader company sample, and additional regressors such as interest rates or tech sector VC funding levels.

---

## Dependencies

```
pandas
scikit-learn
prophet
matplotlib
joblib
```

Install with:
```bash
pip install pandas scikit-learn prophet matplotlib joblib
```

## Usage

Open the notebook from the `notebooks/` directory:

```bash
jupyter notebook forecast.ipynb
```

Ensure the dataset is located at `../data/tech_employment_2000_2025.csv` relative to the notebook. Run all cells top to bottom — the final cell will save all models and artifacts to `../models/`.
