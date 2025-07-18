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

## Example Visuals

### EDA
- [eda_1.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/eda_1.png)
- [eda_2.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/eda_2.png)
- [eda_3.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/eda_3.png)
- [eda_4.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/eda_4.png)
- [eda_5.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/eda_5.png)
- [eda_6.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/eda_6.png)

### Classification Model
- [classification_model_1.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/classification_model_1.png)
- [classificatio_model_2.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/classificatio_model_2.png)
- [classification_model_3.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/classification_model_3.png)
- [classification_model_4.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/classification_model_4.png)

### Regression Model
- [regression_model_1.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/regression_model_1.png)

### Streamlit Dashboard
- [streamlit_1.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/streamlit_1.png)
- [streamlit_2.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/streamlit_2.png)
- [streamlit_3.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/streamlit_3.png)

### Risk and Victim Analysis
- [processed_da_2.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/processed_da_2.png)
- [processed_da_3.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/processed_da_3.png)
- [Processed_da_4.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/Processed_da_4.png)

### Vehicle Analysis
- [processed_veh_1.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/processed_veh_1.png)
- [processed_veh_2.png](https://github.com/AlexNegulescu111/accidente_project/blob/main/visuals/processed_veh_2.png)

## TODOs (Optional Improvements)

- [ ] Improve dashboard with time filters
- [ ] Add map visualizations per region
- [ ] Export trained models to `.pkl`

## Author

Alex Negulescu – [GitHub](https://github.com/AlexNegulescu111)
