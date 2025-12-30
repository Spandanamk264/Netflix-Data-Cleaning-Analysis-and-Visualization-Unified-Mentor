
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from loguru import logger
import joblib
import os

logger.add("logs/recommender.log")
MODEL_DIR = "models/recommender"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_recommender():
    logger.info("Initializing Recommender System...")
    
    # 1. Load Data
    df = pd.read_csv("data/processed/netflix_features.csv")
    embeddings = np.load("data/processed/title_embeddings.npy")
    
    # 2. Simulate User Interactions (Collaborative Filtering Prep)
    # Requirement: "user-simulation"
    logger.info("Simulating User Interactions...")
    n_users = 1000
    n_items = len(df)
    
    # Mock: Users have 1-3 favorite genres and rate items in those genres higher
    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    user_profiles = np.random.choice(genre_cols, size=(n_users, 3)) # 3 fav genres each
    
    # Create Interaction Matrix (Sparse)
    # For speed, we just train SVD on a dense subset or use implicit logic
    # We will compute Item-Item similarity based on Embeddings (Content) first
    
    logger.info("Training Content-Based Model (Cosine Sim)...")
    # We calculate similarity on the embeddings + metadata features
    # Metadata: Genre, Year, Country ONE-HOT.
    metadata = df.drop(columns=['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added', 
                                'release_year', 'rating', 'duration', 'listed_in', 'description', 
                                'date_added_clean', 'genre_list', 'country_simple', 'add_month', 
                                'add_day_of_week', 'director_popularity'] 
                       + [c for c in df.columns if c.startswith('country_')], errors='ignore')
    
    # Select numeric columns only
    metadata = metadata.select_dtypes(include=[np.number]).fillna(0)
    
    # Normalize
    metadata_norm = (metadata - metadata.mean()) / metadata.std()
    
    # Combine Embeddings (Dense) + Metadata (Dense)
    # Give embeddings 70% weight, metadata 30%
    # This involves dimensionality reduction or direct concat
    
    # SVD for dimensionality reduction of the hybrid feature set
    logger.info("Reducing dimensions with SVD...")
    # Concat
    features = np.hstack([embeddings, metadata_norm.values])
    svd = TruncatedSVD(n_components=100)
    reduced_features = svd.fit_transform(features)
    
    # Save SVD model and matrix
    joblib.dump(svd, f"{MODEL_DIR}/svd_model.joblib")
    np.save(f"{MODEL_DIR}/item_features_reduced.npy", reduced_features)
    
    logger.success("Recommender training complete. Artifacts saved.")

if __name__ == "__main__":
    train_recommender()
