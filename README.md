# Netflix Content Analysis & ML Pipeline (2008-2021)

## 🎯 Project Overview

A production-quality, end-to-end data science project analyzing Netflix's content catalog (2008-2021) with advanced machine learning, time-series forecasting, recommendation systems, and interactive dashboards.

## 📊 Key Deliverables

### 1. Data Pipeline & Quality
- Automated data quality reports
- Deterministic cleaning & transformation pipeline
- Feature engineering (50+ features)
- PostgreSQL relational schema with ETL

### 2. Exploratory Analysis
- Interactive Plotly/Dash dashboards
- Temporal trend analysis
- Geographic & genre distributions
- Director/cast popularity metrics

### 3. Machine Learning Models

#### A. Recommendation System
- **Collaborative Filtering**: Matrix factorization (ALS)
- **Content-Based**: Title embeddings + metadata features
- **Hybrid Ranking**: Combined approach with explainability
- **Target**: NDCG@10 improvement ≥20% over popularity baseline

#### B. Multi-Task Classification
- **Type Prediction**: Movie vs TV Show
- **Rating Classification**: Family/Adult content buckets
- **Genre Classification**: Primary genre from title + metadata
- **Models**: XGBoost/LightGBM + DistilBERT fine-tuning
- **Explainability**: SHAP values for interpretability
- **Target**: F1 > 0.80 on major genre labels

#### C. Time-Series Forecasting
- **Objective**: Predict monthly new title volume by genre (24 months)
- **Models**: Prophet + LSTM ensemble
- **Uncertainty**: Prediction intervals & confidence bands
- **Target**: RMSE capturing seasonal spikes

### 4. Deployment & Monitoring
- FastAPI endpoints for recommendations & classification
- Interactive web dashboard
- Docker containerization
- Data drift monitoring
- Unit & integration tests

## 🏗️ Project Structure

```
netflix_project/
├── data/
│   ├── raw/                    # Original netflix1.csv (versioned)
│   ├── processed/              # Cleaned datasets
│   ├── features/               # Engineered features
│   └── exports/                # Final exports
├── database/
│   ├── schema/                 # PostgreSQL DDL scripts
│   ├── migrations/             # Database migrations
│   └── etl/                    # ETL scripts
├── notebooks/
│   ├── 01_data_quality.ipynb   # Quality assessment
│   ├── 02_cleaning.ipynb       # Data cleaning
│   ├── 03_eda.ipynb            # Exploratory analysis
│   ├── 04_feature_eng.ipynb    # Feature engineering
│   ├── 05_recommender.ipynb    # Recommendation system
│   ├── 06_classification.ipynb # Multi-task classification
│   └── 07_forecasting.ipynb    # Time-series forecasting
├── src/
│   ├── data/
│   │   ├── quality_check.py    # Data quality module
│   │   ├── cleaning.py         # Cleaning functions
│   │   ├── feature_engineering.py
│   │   └── etl.py              # Database ETL
│   ├── models/
│   │   ├── recommender.py      # Recommendation models
│   │   ├── classifier.py       # Classification models
│   │   ├── forecaster.py       # Time-series models
│   │   └── explainer.py        # SHAP & interpretability
│   ├── evaluation/
│   │   ├── metrics.py          # Custom metrics
│   │   └── error_analysis.py   # Error analysis tools
│   ├── api/
│   │   ├── main.py             # FastAPI application
│   │   ├── endpoints.py        # API endpoints
│   │   └── middleware.py       # Rate limiting, caching
│   └── dashboard/
│       ├── app.py              # Dash/Plotly dashboard
│       └── components.py       # Dashboard components
├── models/                     # Trained model artifacts
├── reports/
│   ├── data_quality/           # Quality reports
│   ├── eda/                    # EDA visualizations
│   ├── model_performance/      # Model evaluation
│   └── business_insights/      # Action items & insights
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── data_contracts/         # Data validation tests
├── deployment/
│   ├── Dockerfile              # Container definition
│   ├── docker-compose.yml      # Multi-service orchestration
│   └── monitoring/             # Monitoring configs
├── logs/
│   └── transformation_changelog.json
├── requirements.txt            # Python dependencies
├── Makefile                    # Pipeline automation
├── tox.ini                     # Testing automation
├── .env.example                # Environment variables
└── presentation.pdf            # Executive summary deck
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 13+
- Docker & Docker Compose
- 8GB+ RAM recommended

### Installation

```bash
# Clone and navigate
cd d:/Unified_internship/netflix_project

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your database credentials

# Initialize database
make init-db

# Run full pipeline
make all
```

### Running Individual Components

```bash
# Data quality report
make quality-report

# Clean data
make clean-data

# Train models
make train-models

# Start API server
make run-api

# Launch dashboard
make run-dashboard

# Run tests
make test
```

## 📈 Model Performance Targets

| Model | Metric | Target | Status |
|-------|--------|--------|--------|
| Genre Classification | F1 Score | > 0.80 | TBD |
| Type Classification | Accuracy | > 0.90 | TBD |
| Rating Classification | F1 Score | > 0.75 | TBD |
| Recommender System | NDCG@10 | +20% vs baseline | TBD |
| Time-Series Forecast | RMSE | Seasonal capture | TBD |

## 🔍 Key Features Engineered

### Text Features
- `title_tokens`: Tokenized title text
- `title_embeddings`: Sentence-BERT embeddings (384-dim)
- `title_length`: Character count

### Categorical Features
- `number_of_genres`: Genre count per title
- `genre_onehot`: One-hot encoded genres
- `director_popularity`: Director's total title count
- `country_encoded`: Label-encoded countries
- `rating_bucket`: Grouped rating categories

### Temporal Features
- `release_age_at_add`: Days between release and Netflix addition
- `year_added`, `month_added`, `day_added`: Parsed from date_added
- `days_since_first_record`: Days from earliest record
- `is_weekend_add`: Boolean for weekend additions

### Content Features
- `season_count`: Number of seasons (TV shows)
- `duration_minutes`: Normalized duration
- `has_director`: Missing director indicator
- `cast_count`: Number of cast members

## 🗄️ Database Schema

### Tables
1. **titles**: Core title information
2. **people**: Directors and cast members
3. **genres**: Genre taxonomy
4. **countries**: Country information
5. **title_people**: Many-to-many relationship
6. **title_genres**: Many-to-many relationship
7. **title_countries**: Many-to-many relationship
8. **time_series**: Daily/monthly aggregations

## 🎨 Dashboard Features

- **Overview**: Type distribution, total titles by year
- **Genre Analysis**: Top genres, trends over time
- **Geographic Insights**: Country heatmap, regional preferences
- **Temporal Patterns**: Monthly/annual release trends
- **Director Analytics**: Top directors, collaboration networks
- **Title Explorer**: Searchable table with filters
- **Recommendation Widget**: Get personalized recommendations
- **Model Predictions**: Live classification & forecasting

## 🔬 Experiments & Analysis

### Error Analysis
- Confusion matrices per genre/country
- Misclassification patterns
- Performance by content type
- Temporal performance drift

### Business Insights
- Under-served genres/regions
- Optimal release timing
- Content gap analysis
- Director/cast ROI analysis

## 🐳 Docker Deployment

```bash
# Build image
docker build -t netflix-ml-pipeline .

# Run with docker-compose
docker-compose up -d

# Access services
# API: http://localhost:8000
# Dashboard: http://localhost:8050
# PostgreSQL: localhost:5432
```

## 📊 API Endpoints

```
GET  /api/v1/recommend/{title_id}     # Get recommendations
POST /api/v1/classify                 # Classify new content
GET  /api/v1/forecast/{genre}         # Genre forecast
GET  /api/v1/health                   # Health check
GET  /api/v1/metrics                  # Model metrics
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Coverage report
pytest --cov=src tests/
```

## 📝 Computational Budget

| Task | Estimated Time | Resources |
|------|---------------|-----------|
| Data Quality & Cleaning | 5-10 min | CPU |
| Feature Engineering | 10-15 min | CPU |
| EDA & Visualization | 5 min | CPU |
| Model Training (XGBoost) | 15-30 min | CPU |
| Model Training (DistilBERT) | 30-60 min | GPU recommended |
| Time-Series Forecasting | 10-20 min | CPU |
| Recommender Training | 20-40 min | CPU |
| **Total Pipeline** | **2-3 hours** | 8GB RAM, GPU optional |

## ⚖️ Ethical Considerations

### Bias & Fairness
- **Geographic Bias**: Dataset heavily skewed toward US/Western content
- **Director Bias**: Popular directors over-represented in recommendations
- **Language Bias**: English-language content dominates
- **Mitigation**: Diversity-aware ranking, regional quotas in recommendations

### Privacy
- No user-level data collected
- Simulated user interactions for collaborative filtering
- Aggregated statistics only in monitoring

### Transparency
- SHAP explanations for all predictions
- Feature importance reporting
- Model card documentation

## 🔮 Next Steps & Future Work

1. **User-Level Personalization**
   - Integrate real user interaction data
   - Build user profile embeddings
   - A/B testing framework

2. **Model Improvements**
   - Multi-modal models (posters, trailers)
   - Graph neural networks for cast/director relationships
   - Transformer-based forecasting (Temporal Fusion Transformer)

3. **Infrastructure**
   - GPU inference for real-time embeddings
   - Model versioning with MLflow
   - Automated retraining pipeline
   - Kubernetes deployment

4. **Business Integration**
   - Content acquisition recommendations
   - Churn prediction integration
   - ROI modeling for content investments

## 📚 References & Resources

- [Netflix Prize Dataset](https://www.kaggle.com/netflix-inc/netflix-prize-data)
- [Sentence Transformers](https://www.sbert.net/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Prophet Forecasting](https://facebook.github.io/prophet/)
- [FastAPI](https://fastapi.tiangolo.com/)

## 👥 Contributors

This project is part of the Unified Internship Data Science portfolio.

## 📄 License

MIT License - See LICENSE file for details

---

**Last Updated**: December 2025  
**Status**: 🚧 In Development  
**Version**: 1.0.0
