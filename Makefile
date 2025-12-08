.PHONY: help install init-db quality-report clean-data feature-eng train-models run-api run-dashboard test clean docker-build docker-up

help:
	@echo "Netflix ML Pipeline - Available Commands"
	@echo "========================================"
	@echo "install          - Install Python dependencies"
	@echo "init-db          - Initialize PostgreSQL database"
	@echo "quality-report   - Generate data quality report"
	@echo "clean-data       - Clean and preprocess data"
	@echo "feature-eng      - Engineer features"
	@echo "train-models     - Train all ML models"
	@echo "run-api          - Start FastAPI server"
	@echo "run-dashboard    - Launch Dash dashboard"
	@echo "test             - Run all tests"
	@echo "clean            - Clean generated files"
	@echo "docker-build     - Build Docker image"
	@echo "docker-up        - Start Docker services"
	@echo "all              - Run complete pipeline"

install:
	pip install -r requirements.txt

init-db:
	python -m src.data.etl --init-schema

quality-report:
	python -m src.data.quality_check

clean-data:
	python -m src.data.cleaning

feature-eng:
	python -m src.data.feature_engineering

train-models:
	python -m src.models.classifier --train
	python -m src.models.recommender --train
	python -m src.models.forecaster --train

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	python src/dashboard/app.py

test:
	pytest tests/ -v --cov=src --cov-report=html

clean:
	rm -rf data/processed/* data/features/* models/* reports/* logs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t netflix-ml-pipeline .

docker-up:
	docker-compose up -d

all: quality-report clean-data feature-eng train-models
	@echo "Pipeline completed successfully!"
