
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from loguru import logger
import os

# Configure Logging
logger.add("logs/feature_engineering.log", rotation="20 MB")

INPUT_PATH = r"data/processed/netflix_cleaned_v2.csv"
OUTPUT_PATH = r"data/processed/netflix_features.csv"
EMBEDDINGS_PATH = r"data/processed/title_embeddings.npy"

def build_features():
    logger.info("Starting feature engineering pipeline...")
    
    # 1. Load Cleaned Data
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded data with shape {df.shape}")
    
    # 2. Text Features (Sentence Transformer Embeddings)
    logger.info("Generating Title Embeddings (DistilBERT)... This may take time.")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df['title'].astype(str).tolist(), show_progress_bar=True)
        np.save(EMBEDDINGS_PATH, embeddings)
        logger.success(f"Saved embeddings to {EMBEDDINGS_PATH}")
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
    
    # 3. Genre Engineering (One-Hot)
    # listed_in is comma separated e.g., "Documentaries, International Movies"
    logger.info("Processing Genres...")
    df['genre_list'] = df['listed_in'].fillna("Unknown").str.split(', ')
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genre_list'])
    genre_df = pd.DataFrame(genre_matrix, columns=[f"genre_{g}" for g in mlb.classes_])
    df = pd.concat([df, genre_df], axis=1)
    df['number_of_genres'] = df['genre_list'].apply(len)
    
    # 4. Director Popularity (Frequency Encoding)
    logger.info("Processing Director Popularity...")
    director_counts = df['director'].value_counts()
    df['director_popularity'] = df['director'].map(director_counts)
    
    # 5. Temporal Features
    if 'date_added_clean' in df.columns:
        df['date_added_clean'] = pd.to_datetime(df['date_added_clean'])
        df['add_month'] = df['date_added_clean'].dt.month
        df['add_day_of_week'] = df['date_added_clean'].dt.dayofweek
        
        # Days since first record
        min_date = df['date_added_clean'].min()
        df['days_since_start'] = (df['date_added_clean'] - min_date).dt.days
    
    # 6. Country Encoding (Simple Top-N One-Hot to avoid high dimensionality)
    # We take top 10 countries, others 'Other'
    top_countries = df['country'].value_counts().nlargest(10).index
    df['country_simple'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')
    df = pd.get_dummies(df, columns=['country_simple'], prefix='country')

    # 7. Save Final Feature Set
    df.to_csv(OUTPUT_PATH, index=False)
    logger.success(f"Feature Engineering complete. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_features()
