# Romanian Traffic Accidents – Data Science Project

This project is a complete end-to-end data science pipeline focused on analyzing and modeling traffic accident data collected from the official Romanian Police website.

## Project Objectives

- Automatically collect and clean structured data on traffic accidents in Romania
- Extract semantic information from unstructured text (number of victims, deaths, vehicle types)
- Perform exploratory and risk analysis at national level
- Train classification and regression models to predict accident severity and traffic resumption time
- Deliver an interactive Streamlit dashboard for exploration

## Project Structure

```
accidente_project/
├── data/
├── scripts/
├── analysis/
├── dashboards/
├── visuals/
├── requirements.txt
└── README.md
* all the scripts use Pathlib for establishing the path to the resources, so they are compatible with any working directory.
```

## Technologies Used

- Selenium, BeautifulSoup – Web scraping
- pandas, numpy, regex – Data cleaning and feature extraction
- matplotlib, seaborn – Visualizations
- scikit-learn, XGBoost, Optuna – ML models and tuning
- SHAP – Model interpretability
- Streamlit – Interactive dashboard

## Main Deliverables

### ETL + Feature Extraction
- scraper_data.py: Automatically extracts structured articles
- cleaning_data.py: Cleans raw accident info, constructs datetime and duration
- process_data.py: Extracts vehicle types and counts, victims and deaths

### EDA + Risk Analysis
- vizualization_eda_accidente.py: Distributions by time and location
- processed_data_analysis.py: Risk scoring for roads and counties
- processed_data_analysis_vehicles.py: Risk by vehicle type

### Machine Learning
- classification_model.py: Predicts whether an accident involves death
- regression_time_resumption.py: Predicts duration of blocked traffic using XGBoost and Optuna

### Dashboard
- dashboards.py: Explore accident metrics and patterns interactively

## How to Run

1. Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Run the scraper:

```bash
python scripts/scraper_data.py
```

3. Clean and process the data:

```bash
python scripts/cleaning_data.py
python scripts/process_data.py
```

4. Run analysis and models:

```bash
python analysis/vizualization_eda_accidente.py
python analysis/processed_data_analysis.py
python scripts/classification_model.py
python scripts/regression_time_resumption.py
```

5. Launch the dashboard:

```bash
streamlit run dashboards/dashboards.py
```

## Results

All metrics are computed on a held-out test set that the model never sees during training or tuning.

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

> Note: the images in `visuals/` were produced before these fixes and will be regenerated.

## Example Visuals

### EDA
- [eda_1.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/eda_1.png)
- [eda_2.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/eda_2.png)
- [eda_3.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/eda_3.png)
- [eda_4.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/eda_4.png)
- [eda_5.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/eda_5.png)
- [eda_6.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/eda_6.png)

### Classification Model
- [classification_model_1.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/classification_model_1.png)
- [classificatio_model_2.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/classificatio_model_2.png)
- [classification_model_3.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/classification_model_3.png)
- [classification_model_4.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/classification_model_4.png)

### Regression Model
- [regression_model_1.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/regression_model_1.png)

### Streamlit Dashboard
- [streamlit_1.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/streamlit_1.png)
- [streamlit_2.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/streamlit_2.png)
- [streamlit_3.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/streamlit_3.png)

### Risk and Victim Analysis
- [processed_da_2.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/processed_da_2.png)
- [processed_da_3.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/processed_da_3.png)
- [Processed_da_4.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/Processed_da_4.png)

### Vehicle Analysis
- [processed_veh_1.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/processed_veh_1.png)
- [processed_veh_2.png](https://github.com/AlexNegulescu111/Data_science_project_accidents/blob/main/visuals/processed_veh_2.png)

## TODOs (Optional Improvements)

- [ ] Improve dashboard with time filters
- [ ] Add map visualizations per region
- [ ] Export trained models to `.pkl`

## Author

Alex Negulescu – [GitHub](https://github.com/AlexNegulescu111)
