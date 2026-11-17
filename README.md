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
| 5. Risk | `analysis/processed_data_analysis.py` | `visuals/risk/` |
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

All metrics are computed on a held-out test set that the model never sees during training or tuning.
They are written to `reports/classification_metrics.json` and `reports/regression_metrics.json` on every run.

### Classification – is the accident fatal?
Fatal accidents are ~7.4% of the data. Features: number of injured, resumption time, county, road.

| Metric | Value |
|---|---|
| ROC-AUC (test) | 0.64 |
| PR-AUC (test) | 0.15 (random model: 0.074) |
| CV ROC-AUC (5-fold, train) | 0.63 ± 0.03 |
| Recall / precision for fatal class (threshold 0.5) | 0.33 / 0.13 |

The model is only slightly better than chance: these four features carry little signal about fatality.
An earlier version reported ROC-AUC ≈ 0.98, but that was caused by upsampling the minority class
*before* the train/test split (copies of the same accidents ended up in both sets).

### Regression – traffic resumption time
Tuned with Optuna on a separate validation set; outliers (> 105 min) removed using bounds computed on the training data.

| Model | MAE (min) | RMSE (min) |
|---|---|---|
| Baseline – predict the mean | 13.46 | 16.59 |
| XGBoost | 13.25 | 16.37 |

XGBoost barely beats the constant baseline, so the current features do not explain resumption time.

### Data quality fixes (v2)
- Diacritics with cedilla (ş, ţ) are now removed too; previously county names such as `CONSTANȚA` were truncated (`CONSTAN`, `BRA`, `MURE`...) → now 41 counties + Bucharest, no garbage values.
- Two-word counties (`SATU MARE`) and source typos (`TELELORMAN`, `VACEA`...) are handled.
- Resumption time: hour-based estimates ("in 2 ore") are parsed, dates such as `30.01.2024` are no longer read as clock times, impossible ~24h values are dropped.
- Victim extraction only counts numbers next to a person noun (no more road numbers / ages / hours counted as victims); negation uses whole words. All 25 accidents whose text mentions a death but had `nr_morti = 0` are now detected.
- Vehicle regex: fixed missing `|` that broke the `ansambluri rutiere` / `autocisterna` patterns.
- Risk score: roads/counties without fatal accidents are no longer dropped (NaN) from the ranking.

## Visuals

All figures are regenerated by `python run_pipeline.py`.

- **EDA** (`visuals/eda/`): [accidents per hour](visuals/eda/accidents_per_hour.png), [per day](visuals/eda/accidents_per_day.png), [per month](visuals/eda/accidents_per_month.png), [per year](visuals/eda/accidents_per_year.png), [day × hour heatmap](visuals/eda/heatmap_day_hour.png), [top counties](visuals/eda/top_counties.png), [top roads](visuals/eda/top_roads.png), [resumption time histogram](visuals/eda/resumption_histogram.png), [boxplot](visuals/eda/resumption_boxplot.png), [mean resumption by county](visuals/eda/mean_resumption_by_county.png)
- **Risk** (`visuals/risk/`): [roads](visuals/risk/top_roads_risk.png), [counties](visuals/risk/top_counties_risk.png), [victims by resumption interval](visuals/risk/victims_by_resumption.png)
- **Vehicles** (`visuals/vehicles/`): [vehicles in serious accidents](visuals/vehicles/vehicles_in_serious_accidents.png), [mean deaths by vehicle type](visuals/vehicles/mean_deaths_by_vehicle.png)
- **Classification** (`visuals/classification/`): [confusion matrix](visuals/classification/confusion_matrix.png), [ROC & PR curves](visuals/classification/roc_pr_curves.png), [SHAP importance](visuals/classification/shap_importance.png), SHAP explanations of fatal cases ([1](visuals/classification/shap_fatal_case_1.png), [2](visuals/classification/shap_fatal_case_2.png), [3](visuals/classification/shap_fatal_case_3.png))
- **Regression** (`visuals/regression/`): [predicted vs actual](visuals/regression/predicted_vs_actual.png), [feature importance](visuals/regression/feature_importance.png)
- **Dashboard** (`visuals/dashboard/`): [overview](visuals/dashboard/dashboard_overview.png), [roads & heatmap](visuals/dashboard/dashboard_roads_heatmap.png)

## TODOs (Optional Improvements)

- [ ] Improve dashboard with time filters
- [ ] Add map visualizations per region
- [ ] Export trained models to `.pkl`

## Author

Alex Negulescu – [GitHub](https://github.com/AlexNegulescu111)
