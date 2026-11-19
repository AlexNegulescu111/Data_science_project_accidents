PYTHON ?= python

.PHONY: install install-dev pipeline etl analysis models test dashboard scrape clean

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

pipeline:            ## clean -> process -> analysis -> models
	$(PYTHON) run_pipeline.py

etl:
	$(PYTHON) scripts/clean_data.py
	$(PYTHON) scripts/process_data.py

analysis: etl
	$(PYTHON) analysis/eda.py
	$(PYTHON) analysis/risk_analysis.py
	$(PYTHON) analysis/vehicle_analysis.py

models: etl
	$(PYTHON) scripts/train_fatality_classifier.py
	$(PYTHON) scripts/train_resumption_regressor.py

test:
	$(PYTHON) -m pytest

dashboard:
	streamlit run dashboard/app.py

scrape:              ## overwrites data/raw_reports.csv
	$(PYTHON) scripts/scrape_data.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
