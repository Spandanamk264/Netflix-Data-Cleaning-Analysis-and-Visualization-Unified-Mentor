
install:
	pip install -r requirements.txt

data_clean:
	python src/data/quality_audit.py
	python src/data/clean_etl.py

features:
	python src/features/build_features.py

train:
	python src/models/train_classification.py
	python src/models/train_forecasting.py
	python src/models/train_recommender.py

test:
	pytest tests/

api:
	uvicorn src.api.main:app --reload

dashboard:
	panel serve src/dashboard/app_pro_light.py --show

all: data_clean features train test
