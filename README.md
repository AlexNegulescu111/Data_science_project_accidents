# Romanian Traffic Accidents – Data Science Project

This project is a complete end-to-end data science pipeline focused on analyzing and modeling traffic accident data collected from the official Romanian Police website (politiaromana.ro, "Info trafic" – about 3,900 reports, 2019–2025).

## Project Objectives

- Automatically collect and clean structured data on traffic accidents in Romania
- Extract semantic information from unstructured text (number of victims, deaths, vehicle types)
- Perform exploratory and risk analysis at national level
- Train classification and regression models to predict accident severity and traffic resumption time
- Deliver an interactive Streamlit dashboard for exploration

## Project Structure

```
├── accidents/            shared package – every script imports from here
│   ├── config.py         paths, constants, county list, dictionaries
│   ├── text.py           text extraction (county, road, resumption, vehicles, victims)
│   ├── data.py           loading + common filters
│   ├── features.py       accident filter + circumstance features for the models
│   ├── stats.py          confidence intervals and shrunk rates
│   ├── plotting.py       saves figures to visuals/ (optionally shows them)
│   └── scraping.py       HTML parsing for the police website
├── scripts/              pipeline steps: scraper, cleaning, processing, models
├── analysis/             EDA, risk scoring, vehicle analysis
├── dashboards/           Streamlit app
├── tests/                pytest suite (extraction rules, scraper parsing, data checks)
├── data/                 raw, clean and processed CSVs
├── visuals/              generated figures (one folder per analysis)
├── reports/              generated model metrics (JSON)
├── run_pipeline.py       runs every step in order
├── Makefile
├── requirements.txt      pinned versions (requirements-dev.txt adds pytest)
└── README.md
```

## Technologies Used

- Selenium, BeautifulSoup – Web scraping
- pandas, numpy, regex – Data cleaning and feature extraction
- matplotlib, seaborn – Visualizations
- scikit-learn, XGBoost, Optuna – ML models and tuning
- SciPy – confidence intervals
- SHAP – Model interpretability
- Streamlit – Interactive dashboard
- pytest – Tests

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Scrape (optional) | `scripts/scraper_data.py` | `data/accidente.csv` |
| 2. Clean | `scripts/cleaning_data.py` | `data/accidente_clean.csv` – county, road, resumption minutes, timestamp |
| 3. Extract | `scripts/process_data.py` | `data/accidente_processed.csv` – vehicles, injured, deaths |
| 4. EDA | `analysis/vizualization_eda_accidente.py` | `visuals/eda/` |
| 5. Risk (burden + severity) | `analysis/processed_data_analysis.py` | `visuals/risk/` |
| 6. Vehicles | `analysis/processed_data_analysis_vehicles.py` | `visuals/vehicles/` |
| 7. Classification | `scripts/classification_model.py` | `visuals/classification/`, `reports/` |
| 8. Regression | `scripts/regression_time_resumption.py` | `visuals/regression/`, `reports/` |
| Dashboard | `dashboards/dashboards.py` | Streamlit app |

## How to Run

Requires Python 3.13 (other recent versions will likely work).

```bash
pip install -r requirements-dev.txt
```

Run the whole pipeline (about 30 seconds); figures are saved, not shown:

```bash
python run_pipeline.py
```

Useful variants: `--skip-models`, `--show` (also open every figure), `--scrape` (re-scrape first). Each script can also be run on its own, e.g. `python scripts/classification_model.py`, from any working directory.

Tests:

```bash
python -m pytest
```

Dashboard:

```bash
streamlit run dashboards/dashboards.py
```

Scraper (needs Chrome; Selenium downloads the driver automatically). It overwrites `data/accidente.csv` unless `--append` is given:

```bash
python scripts/scraper_data.py --start-page 1 --end-page 5 --output data/sample.csv
```

With `make`: `make install-dev`, `make pipeline`, `make test`, `make dashboard`.

## Results

All metrics are computed on a held-out test set that the model never sees during training or tuning,
and are written to `reports/classification_metrics.json` and `reports/regression_metrics.json` on every run.

### Data used
About 15% of the reports are not accidents (roadworks, vehicle fires, oversize transports, railway incidents).
They are detected from the text and excluded from the severity analysis and the classifier:
**3,271 accidents**, of which **9.7% are fatal** in the test set.

### Classification – is the accident fatal?
The deaths are extracted from the same text the features come from, so the model only uses
**circumstances** of the accident, never outcome words (enforced by a test):
time (hour, day, month, weekend), county, road type, number of vehicles and flags extracted from the
description – pedestrian, cyclist, motorcycle, frontal collision, rollover, run-off-road, fixed obstacle,
animal, railway, heavy vehicle, bus, fire, bad weather, skid.

Four models were compared with 5-fold cross-validation on the training set:

| Model | CV ROC-AUC | CV PR-AUC |
|---|---|---|
| Logistic regression (one-hot counties) | 0.704 | **0.269** |
| Random forest (target-encoded county) | 0.717 | 0.240 |
| Gradient boosting | 0.660 | 0.227 |
| Phase 1 features (injured, resumption time, county, road) – reference | 0.672 | 0.185 |

Selected model (best PR-AUC): **logistic regression**. On the test set:

| Metric | Value |
|---|---|
| ROC-AUC | **0.73** |
| PR-AUC | **0.27** (random model: 0.097, i.e. 2.8× better) |
| Recall / precision for fatal accidents | 0.41 / 0.26 (threshold 0.65, chosen on out-of-fold predictions) |

The strongest risk factors (odds ratios, other features held constant, saved in the metrics file):
**frontal collision 5.4×, pedestrian 4.3×, cyclist 2.8×**, vehicle fire 2.2×, motorcycle 1.6×, weekend 1.6×,
heavy vehicle 1.5×. Rollovers (0.3×) and animal collisions (0.6×) are *less* often fatal than average.
County effects are also large (Dolj, Giurgiu, Buzau high; Ilfov, Bistrita-Nasaud low) but may partly reflect
differences in how reports are written, not only in danger.
The first version of this project reported ROC-AUC 0.985, which came from upsampling before the train/test split.

### Regression – traffic resumption time
XGBoost tuned with Optuna on a separate validation set; outliers (> 105 min) removed with
bounds computed on the training data. Both feature sets are tuned with the same procedure.

| Model | MAE (min) | RMSE (min) |
|---|---|---|
| Baseline – predict the mean | 14.23 | 17.87 |
| XGBoost – phase 2 features | 13.93 | 17.60 |
| XGBoost – circumstances | 13.87 | 17.60 |

The models barely beat a constant. The target explains why: it is the **police estimate** of the resumption
time, not a measured duration – 62% of the values are multiples of 15 minutes and 28% are exactly 30 minutes.
There is little signal left to learn.

### Risk by road, county and vehicle
Two complementary views on accidents only (no traffic-volume data is available, so no per-km or per-vehicle exposure):
- **Burden** – number of fatal accidents: DN 2, DN 1, DN 6 and DN 7 dominate, largely because they carry the most traffic.
- **Severity** – share of accidents that are fatal, with 95% Wilson confidence intervals, ranked by a rate shrunk
  toward the national average (so 1 fatal out of 2 does not top the list), for groups with at least 30 accidents.
  Most county/road differences have overlapping intervals; the clearest signal is by road user:
  **37% of accidents involving a pedestrian are fatal** (95% CI 29–45%), 30% for cyclists, vs. 9.7% overall.

### Data quality fixes
- Diacritics with cedilla (ş, ţ) are now removed too; previously county names such as `CONSTANȚA` were truncated (`CONSTAN`, `BRA`, `MURE`...) → now 41 counties + Bucharest, no garbage values.
- Two-word counties (`SATU MARE`) and source typos (`TELELORMAN`, `VACEA`...) are handled.
- Resumption time: hour-based estimates ("in 2 ore") are parsed, dates such as `30.01.2024` are no longer read as clock times, impossible ~24h values are dropped.
- Victim extraction only counts numbers next to a person noun (no more road numbers / ages / hours counted as victims); negation uses whole words. All 25 accidents whose text mentions a death but had `nr_morti = 0` are now detected.
- Vehicle regex: fixed missing `|` that broke the `ansambluri rutiere` / `autocisterna` patterns.

## Visuals

All figures are regenerated by `python run_pipeline.py`.

- **EDA** (`visuals/eda/`): [accidents per hour](visuals/eda/accidents_per_hour.png), [per day](visuals/eda/accidents_per_day.png), [per month](visuals/eda/accidents_per_month.png), [per year](visuals/eda/accidents_per_year.png), [day × hour heatmap](visuals/eda/heatmap_day_hour.png), [top counties](visuals/eda/top_counties.png), [top roads](visuals/eda/top_roads.png), [resumption time histogram](visuals/eda/resumption_histogram.png), [boxplot](visuals/eda/resumption_boxplot.png), [mean resumption by county](visuals/eda/mean_resumption_by_county.png)
- **Risk** (`visuals/risk/`): burden – [roads](visuals/risk/roads_burden.png), [counties](visuals/risk/counties_burden.png); severity – [roads](visuals/risk/roads_severity.png), [counties](visuals/risk/counties_severity.png); [victims by resumption interval](visuals/risk/victims_by_resumption.png)
- **Vehicles** (`visuals/vehicles/`): [fatality rate by vehicle / road user](visuals/vehicles/fatality_rate_by_vehicle.png), [vehicles in serious accidents](visuals/vehicles/vehicles_in_serious_accidents.png)
- **Classification** (`visuals/classification/`): [model comparison](visuals/classification/model_comparison.png), [odds ratios](visuals/classification/odds_ratios.png), [ROC & PR curves](visuals/classification/roc_pr_curves.png), [confusion matrix](visuals/classification/confusion_matrix.png) (a tree model, if selected, is explained with SHAP instead of odds ratios)
- **Regression** (`visuals/regression/`): [model comparison](visuals/regression/model_comparison.png), [predicted vs actual](visuals/regression/predicted_vs_actual.png), [feature importance](visuals/regression/feature_importance.png)
- **Dashboard** (`visuals/dashboard/`): [overview](visuals/dashboard/dashboard_overview.png), [roads & heatmap](visuals/dashboard/dashboard_roads_heatmap.png)

## TODOs (Optional Improvements)

- [ ] Improve dashboard with time filters
- [ ] Add map visualizations per region
- [ ] Export trained models to `.pkl`

## Author

Alex Negulescu – [GitHub](https://github.com/AlexNegulescu111)
