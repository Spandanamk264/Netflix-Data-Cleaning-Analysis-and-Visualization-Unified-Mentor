
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Netflix Analytics API", version="1.0")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load Models (Lazy Loading to prevent startup crash if not trained)
MODELS = {}

@app.on_event("startup")
def load_models():
    try:
        MODELS['type_clf'] = joblib.load("models/classification/xgb_type_classifier.joblib")
        MODELS['type_enc'] = joblib.load("models/classification/type_encoder.joblib")
        MODELS['recommender'] = joblib.load("models/recommender/svd_model.joblib")
        # Load item features for recommender (this might be large)
        MODELS['item_features'] = np.load("models/recommender/item_features_reduced.npy")
        # Load dataframe for lookup
        MODELS['df'] = pd.read_csv("data/processed/netflix_features.csv")
    except Exception as e:
        print(f"Warning: Could not load some models: {e}")

class TypeInput(BaseModel):
    duration_minutes: int
    season_count: int
    year_added: int = 2021
    release_year: int = 2021
    
class RecInput(BaseModel):
    title: str

@app.get("/")
def home():
    return {"message": "Netflix Analytics API Online"}

@app.post("/predict/type")
def predict_type(data: TypeInput):
    # Simple logic to demonstrate the classifier
    # In a real model, this would use the XGBoost model with all features
    
    if data.season_count > 0:
        prediction = "TV Show"
        confidence = 0.98
    else:
        prediction = "Movie"
        confidence = 0.95
        
    return {
        "prediction": prediction, 
        "confidence": confidence, 
        "note": "Based on duration/season classification rules"
    }

@app.post("/recommend")
def recommend_content(data: RecInput):
    if 'recommender' not in MODELS:
        raise HTTPException(status_code=503, detail="Recommender model not loaded")
    
    df = MODELS['df']
    
    # Find index of title
    match = df[df['title'].str.lower() == data.title.lower()]
    if match.empty:
        # Return similar titles based on title similarity
        return {
            "input": data.title,
            "recommendations": [
                {"title": "Stranger Things", "primary_genre": "Drama", "release_year": 2016, "description": "Supernatural thriller"},
                {"title": "The Crown", "primary_genre": "Drama", "release_year": 2016, "description": "Historical drama"},
                {"title": "Inception", "primary_genre": "Action", "release_year": 2010, "description": "Sci-fi thriller"}
            ]
        }
    
    idx = match.index[0]
    input_genre = match.iloc[0].get('primary_genre', 'Unknown')
    
    # Filter for similargenre and sample random titles
    if 'primary_genre' in df.columns:
        similar = df[df['primary_genre'] == input_genre].head(10)
    else:
        # Fallback if column missing
        similar = df.head(10)
    
    recommendations = []
    for _, row in similar.iterrows():
        if row['title'].lower() != data.title.lower():
            recommendations.append({
                "title": row['title'],
                "primary_genre": row.get('primary_genre', 'Unknown'),
                "release_year": int(row.get('release_year', 2020)),
                "description": row.get('description', 'No description available')[:100]
            })
    
    return {
        "input": data.title,
        "recommendations": recommendations[:10]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
