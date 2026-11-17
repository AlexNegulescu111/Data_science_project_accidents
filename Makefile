PYTHON ?= python

.PHONY: install install-dev pipeline etl analysis models test dashboard scrape clean

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

pipeline:            ## clean -> process -> analysis -> models
	$(PYTHON) run_pipeline.py

etl:
	$(PYTHON) scripts/cleaning_data.py
	$(PYTHON) scripts/process_data.py

analysis: etl
	$(PYTHON) analysis/vizualization_eda_accidente.py
	$(PYTHON) analysis/processed_data_analysis.py
	$(PYTHON) analysis/processed_data_analysis_vehicles.py

models: etl
	$(PYTHON) scripts/classification_model.py
	$(PYTHON) scripts/regression_time_resumption.py

test:
	$(PYTHON) -m pytest

dashboard:
	streamlit run dashboards/dashboards.py

scrape:              ## overwrites data/accidente.csv
	$(PYTHON) scripts/scraper_data.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
