
# Netflix Data Science Pipeline - "The Production Standard"

This project demonstrates a production-grade End-to-End Data Science lifecycle using the Netflix Dataset.

## 📁 Structure

- `src/data`: ETL pipelines (deterministic cleaning, quality reports, SQL loader).
- `src/features`: Embeddings (S-Transformer), One-Hot, Temporal engineering.
- `src/models`: Training scripts for Classification, Forecasting, and Recommendations.
- `src/api`: FastAPI serving the models.
- `notebooks`: Reproducible experiments.
- `reports`: Generated Data Quality and Model Performance reports.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline (ETL -> ML)**
   ```bash
   # 1. Clean Data
   python src/data/quality_audit.py
   python src/data/clean_etl.py
   
   # 2. Build Features (Embeddings + Metadata)
   python src/features/build_features.py
   
   # 3. Train Models
   python src/models/train_classification.py
   python src/models/train_forecasting.py
   python src/models/train_recommender.py
   ```

3. **Start API**
   ```bash
   uvicorn src.api.main:app --reload
   ```

4. **View Dashboard**
   ```bash
   panel serve src/dashboard/app_pro_light.py --show
   ```

## 📊 Models Implemented

| Task | Model | Status |
|------|-------|--------|
| **Classification** | XGBoost (Type & Rating) | ✅ |
| **Forecasting** | Prophet (Monthly Volume) | ✅ |
| **Recommendation** | Hybrid (SVD + Content Embedding) | ✅ |

## 🐳 Docker Deployment

To run in a container:
```bash
docker build -t netflix-ml-app .
docker run -p 8000:8000 -p 5006:5006 netflix-ml-app
```
