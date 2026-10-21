.PHONY: help setup data train-cost train-member train-fraud train-all cards readme api app test lint format docker-up clean

PYTHON ?= python
PYTEST ?= pytest
RUFF ?= ruff
UVICORN ?= uvicorn
STREAMLIT ?= streamlit

help:
	@echo "Available targets:"
	@echo "  setup         Set up virtual environment and install dependencies"
	@echo "  data          Download and verify raw datasets"
	@echo "  train-cost    Train cost prediction model, conformal intervals, fairness audit"
	@echo "  train-member  Train member risk regression and classification models"
	@echo "  train-fraud   Train provider fraud anomaly and supervised models"
	@echo "  train-all     Train all three modules sequentially"
	@echo "  cards         Generate markdown model cards for all models"
	@echo "  readme        Update README.md with metrics from reports/metrics"
	@echo "  api           Run FastAPI backend service on port 8000"
	@echo "  app           Run Streamlit web app on port 8501"
	@echo "  test          Run pytest suite with coverage"
	@echo "  lint          Run ruff linter check"
	@echo "  format        Format code with ruff"
	@echo "  docker-up     Run api and app services via docker-compose"
	@echo "  clean         Clean cache and build artifacts"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

data:
	$(PYTHON) scripts/download_data.py

train-cost:
	$(PYTHON) -m claimsight.cli train-cost

train-member:
	$(PYTHON) -m claimsight.cli train-member

train-fraud:
	$(PYTHON) -m claimsight.cli train-fraud

train-all: train-cost train-member train-fraud

cards:
	$(PYTHON) -m claimsight.cli generate-cards

readme:
	$(PYTHON) scripts/update_readme.py

api:
	$(UVICORN) claimsight.api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	$(STREAMLIT) run src/claimsight/app/Home.py --server.port 8501 --server.address 0.0.0.0

test:
	$(PYTEST)

lint:
	$(RUFF) check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

docker-up:
	docker-compose up --build

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('.pytest_cache')]"
