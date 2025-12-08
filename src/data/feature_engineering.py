"""
Feature Engineering Module
Advanced feature creation including embeddings, encodings, and derived features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# NLP and Embeddings
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class FeatureEngineer:
    """Advanced feature engineering for Netflix dataset"""
    
    def __init__(self, input_path: str, output_dir: str = "data/features"):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.encoders = {}
        self.scalers = {}
        self.feature_log = []
        
    def log_feature(self, feature_name: str, details: Dict):
        """Log feature creation"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'feature': feature_name,
            'details': details
        }
        self.feature_log.append(entry)
        print(f"  ✓ Created: {feature_name}")
    
    def load_data(self) -> pd.DataFrame:
        """Load cleaned data"""
        print("Loading cleaned data...")
        self.df = pd.read_csv(self.input_path)
        print(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
        
        # Ensure date columns are datetime objects
        if 'date_added_parsed' in self.df.columns:
            self.df['date_added_parsed'] = pd.to_datetime(self.df['date_added_parsed'], errors='coerce')
            
        return self.df
    
    def create_title_embeddings(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Create sentence embeddings for titles using Sentence-BERT"""
        print("\n=== Creating Title Embeddings ===")
        print(f"Loading model: {model_name}...")
        
        try:
            # Load pre-trained model (384-dimensional embeddings)
            model = SentenceTransformer(model_name)
            
            # Generate embeddings
            titles = self.df['title'].fillna('').tolist()
            print(f"Generating embeddings for {len(titles)} titles...")
            
            embeddings = model.encode(
                titles,
                show_progress_bar=True,
                batch_size=32
            )
            
            # Save embeddings
            embedding_path = self.output_dir / 'title_embeddings.npy'
            np.save(embedding_path, embeddings)
            
            # Add embedding columns to dataframe (for reference)
            for i in range(embeddings.shape[1]):
                self.df[f'title_emb_{i}'] = embeddings[:, i]
            
            self.log_feature(
                'title_embeddings',
                {
                    'model': model_name,
                    'dimensions': embeddings.shape[1],
                    'shape': embeddings.shape,
                    'saved_to': str(embedding_path)
                }
            )
            
            print(f"✓ Embeddings saved to {embedding_path}")
            
        except Exception as e:
            print(f"✗ Error creating embeddings: {e}")
            print("  Skipping embeddings (requires internet for first download)")
    
    def create_title_tokens(self):
        """Tokenize titles and create token-based features"""
        print("\n=== Creating Title Tokens ===")
        
        if 'title' not in self.df.columns:
            return
        
        # Tokenize (simple whitespace split)
        self.df['title_tokens'] = (
            self.df['title']
            .fillna('')
            .str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.split()
        )
        
        # Token count
        self.df['title_token_count'] = self.df['title_tokens'].apply(len)
        
        # Average token length
        self.df['title_avg_token_length'] = self.df['title_tokens'].apply(
            lambda tokens: np.mean([len(t) for t in tokens]) if tokens else 0
        )
        
        self.log_feature(
            'title_tokens',
            {
                'new_columns': ['title_tokens', 'title_token_count', 'title_avg_token_length'],
                'avg_tokens': self.df['title_token_count'].mean()
            }
        )
    
    def encode_genres(self):
        """Create genre-based features"""
        print("\n=== Encoding Genres ===")
        
        if 'listed_in' not in self.df.columns:
            return
        
        # Split genres
        genre_lists = (
            self.df['listed_in']
            .fillna('')
            .str.split(',')
            .apply(lambda x: [g.strip() for g in x if g.strip()])
        )
        
        # One-hot encode genres
        mlb = MultiLabelBinarizer()
        genre_encoded = mlb.fit_transform(genre_lists)
        
        # Create genre columns
        genre_columns = [f'genre_{g.replace(" ", "_").replace("&", "and").lower()}' 
                        for g in mlb.classes_]
        
        genre_df = pd.DataFrame(
            genre_encoded,
            columns=genre_columns,
            index=self.df.index
        )
        
        # Add to main dataframe
        self.df = pd.concat([self.df, genre_df], axis=1)
        
        # Save encoder
        self.encoders['genre_mlb'] = mlb
        
        # Primary genre (first listed)
        self.df['primary_genre'] = genre_lists.apply(
            lambda x: x[0] if x else 'Unknown'
        )
        
        self.log_feature(
            'genre_encoding',
            {
                'total_genres': len(mlb.classes_),
                'genre_columns_created': len(genre_columns),
                'sample_genres': list(mlb.classes_)[:10]
            }
        )
    
    def calculate_director_popularity(self):
        """Calculate director popularity (number of titles)"""
        print("\n=== Calculating Director Popularity ===")
        
        if 'director' not in self.df.columns:
            return
        
        # Count titles per director
        director_counts = (
            self.df['director']
            .value_counts()
            .to_dict()
        )
        
        # Map to dataframe
        self.df['director_popularity'] = (
            self.df['director']
            .map(director_counts)
            .fillna(0)
        )
        
        # Director rank (higher rank = more popular)
        self.df['director_rank'] = (
            self.df['director_popularity']
            .rank(method='dense', ascending=False)
        )
        
        # Has prolific director (>5 titles)
        self.df['has_prolific_director'] = (
            self.df['director_popularity'] > 5
        ).astype(int)
        
        self.log_feature(
            'director_popularity',
            {
                'unique_directors': len(director_counts),
                'max_titles': max(director_counts.values()),
                'avg_titles_per_director': np.mean(list(director_counts.values())),
                'new_columns': ['director_popularity', 'director_rank', 'has_prolific_director']
            }
        )
    
    def calculate_cast_features(self):
        """Calculate cast-based features"""
        print("\n=== Calculating Cast Features ===")
        
        if 'cast' not in self.df.columns:
            return
        
        # Cast member count (already have cast_count from cleaning)
        # Calculate cast popularity
        
        # Split cast
        cast_lists = (
            self.df['cast']
            .fillna('')
            .str.split(',')
            .apply(lambda x: [c.strip() for c in x if c.strip() and c.strip() != 'Unknown Cast'])
        )
        
        # Get all cast members
        all_cast = []
        for cast_list in cast_lists:
            all_cast.extend(cast_list)
        
        # Count appearances
        from collections import Counter
        cast_counts = Counter(all_cast)
        
        # Top cast member popularity
        self.df['top_cast_popularity'] = cast_lists.apply(
            lambda cast_list: max([cast_counts.get(c, 0) for c in cast_list]) if cast_list else 0
        )
        
        self.log_feature(
            'cast_features',
            {
                'unique_cast_members': len(cast_counts),
                'most_frequent_actor': cast_counts.most_common(1)[0] if cast_counts else None,
                'new_columns': ['top_cast_popularity']
            }
        )
    
    def encode_countries(self):
        """Encode country information"""
        print("\n=== Encoding Countries ===")
        
        if 'country' not in self.df.columns:
            return
        
        # Primary country (first listed)
        self.df['primary_country'] = (
            self.df['country']
            .fillna('Unknown Country')
            .str.split(',')
            .apply(lambda x: x[0].strip() if x else 'Unknown Country')
        )
        
        # Label encode primary country
        le = LabelEncoder()
        self.df['country_encoded'] = le.fit_transform(self.df['primary_country'])
        self.encoders['country_le'] = le
        
        # Country popularity
        country_counts = self.df['primary_country'].value_counts().to_dict()
        self.df['country_popularity'] = (
            self.df['primary_country']
            .map(country_counts)
        )
        
        # Is US content
        self.df['is_us_content'] = (
            self.df['primary_country'] == 'United States'
        ).astype(int)
        
        # Is international (non-US)
        self.df['is_international'] = (
            ~self.df['is_us_content'].astype(bool)
        ).astype(int)
        
        self.log_feature(
            'country_encoding',
            {
                'unique_countries': len(le.classes_),
                'top_country': self.df['primary_country'].mode()[0],
                'new_columns': ['primary_country', 'country_encoded', 'country_popularity', 
                               'is_us_content', 'is_international']
            }
        )
    
    def encode_rating(self):
        """Encode rating categories"""
        print("\n=== Encoding Ratings ===")
        
        if 'rating' not in self.df.columns:
            return
        
        # Rating buckets
        rating_buckets = {
            'Family': ['TV-Y', 'TV-Y7', 'TV-Y7-FV', 'TV-G', 'G', 'PG'],
            'Teen': ['TV-PG', 'PG-13', 'TV-14'],
            'Adult': ['TV-MA', 'R', 'NC-17'],
            'Unrated': ['NR', 'UR']
        }
        
        def get_rating_bucket(rating):
            for bucket, ratings in rating_buckets.items():
                if rating in ratings:
                    return bucket
            return 'Unrated'
        
        self.df['rating_bucket'] = self.df['rating'].apply(get_rating_bucket)
        
        # Label encode
        le_rating = LabelEncoder()
        self.df['rating_encoded'] = le_rating.fit_transform(self.df['rating'])
        self.encoders['rating_le'] = le_rating
        
        le_bucket = LabelEncoder()
        self.df['rating_bucket_encoded'] = le_bucket.fit_transform(self.df['rating_bucket'])
        self.encoders['rating_bucket_le'] = le_bucket
        
        self.log_feature(
            'rating_encoding',
            {
                'rating_buckets': list(rating_buckets.keys()),
                'unique_ratings': len(le_rating.classes_),
                'new_columns': ['rating_bucket', 'rating_encoded', 'rating_bucket_encoded']
            }
        )
    
    def create_temporal_features(self):
        """Create advanced temporal features"""
        print("\n=== Creating Temporal Features ===")
        
        if 'date_added_parsed' not in self.df.columns:
            return
        
        # Quarter
        self.df['quarter_added'] = self.df['date_added_parsed'].dt.quarter
        
        # Week of year
        self.df['week_of_year'] = self.df['date_added_parsed'].dt.isocalendar().week
        
        # Is holiday season (Nov-Dec)
        self.df['is_holiday_season'] = (
            self.df['month_added'].isin([11, 12])
        ).astype(int)
        
        # Is summer (Jun-Aug)
        self.df['is_summer'] = (
            self.df['month_added'].isin([6, 7, 8])
        ).astype(int)
        
        # Days since release
        if 'release_age_days' in self.df.columns:
            # Binned release age
            self.df['release_age_category'] = pd.cut(
                self.df['release_age_days'],
                bins=[-np.inf, 365, 365*3, 365*10, np.inf],
                labels=['Recent', 'Few_Years', 'Decade', 'Classic']
            )
        
        self.log_feature(
            'temporal_features',
            {
                'new_columns': ['quarter_added', 'week_of_year', 'is_holiday_season', 
                               'is_summer', 'release_age_category']
            }
        )
    
    def create_content_features(self):
        """Create content-specific features"""
        print("\n=== Creating Content Features ===")
        
        # Type encoding
        if 'type' in self.df.columns:
            le_type = LabelEncoder()
            self.df['type_encoded'] = le_type.fit_transform(self.df['type'])
            self.encoders['type_le'] = le_type
            
            self.df['is_movie'] = (self.df['type'] == 'Movie').astype(int)
            self.df['is_tv_show'] = (self.df['type'] == 'TV Show').astype(int)
        
        # Duration features
        if 'duration_minutes' in self.df.columns:
            # Binned duration for movies
            self.df['duration_category'] = pd.cut(
                self.df['duration_minutes'],
                bins=[0, 60, 90, 120, np.inf],
                labels=['Short', 'Medium', 'Long', 'Very_Long']
            )
        
        if 'season_count' in self.df.columns:
            # Season categories for TV shows
            self.df['season_category'] = pd.cut(
                self.df['season_count'],
                bins=[0, 1, 3, 5, np.inf],
                labels=['Limited', 'Short', 'Medium', 'Long']
            )
            
            # Is limited series
            self.df['is_limited_series'] = (
                self.df['season_count'] == 1
            ).astype(int)
        
        self.log_feature(
            'content_features',
            {
                'new_columns': ['type_encoded', 'is_movie', 'is_tv_show', 
                               'duration_category', 'season_category', 'is_limited_series']
            }
        )
    
    def create_interaction_features(self):
        """Create interaction features"""
        print("\n=== Creating Interaction Features ===")
        
        # Genre count × duration
        if 'number_of_genres' in self.df.columns and 'duration_minutes' in self.df.columns:
            self.df['genre_duration_interaction'] = (
                self.df['number_of_genres'] * self.df['duration_minutes'].fillna(0)
            )
        
        # Country count × cast count
        if 'number_of_countries' in self.df.columns and 'cast_count' in self.df.columns:
            self.df['country_cast_interaction'] = (
                self.df['number_of_countries'] * self.df['cast_count']
            )
        
        # Release age × director popularity
        if 'release_age_years' in self.df.columns and 'director_popularity' in self.df.columns:
            self.df['age_director_interaction'] = (
                self.df['release_age_years'].fillna(0) * self.df['director_popularity']
            )
        
        self.log_feature(
            'interaction_features',
            {
                'new_columns': ['genre_duration_interaction', 'country_cast_interaction', 
                               'age_director_interaction']
            }
        )
    
    def scale_numeric_features(self):
        """Scale numeric features"""
        print("\n=== Scaling Numeric Features ===")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude encoded columns and IDs
        exclude_cols = ['show_id', 'type_encoded', 'rating_encoded', 'country_encoded',
                       'rating_bucket_encoded', 'is_movie', 'is_tv_show', 'is_us_content',
                       'is_international', 'is_weekend_add', 'has_director', 'has_cast',
                       'has_country', 'has_prolific_director', 'is_limited_series',
                       'is_holiday_season', 'is_summer']
        
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols 
                        and not col.startswith('genre_') 
                        and not col.startswith('title_emb_')]
        
        if cols_to_scale:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.df[cols_to_scale].fillna(0))
            
            # Create scaled columns
            for i, col in enumerate(cols_to_scale):
                self.df[f'{col}_scaled'] = scaled_data[:, i]
            
            self.scalers['standard_scaler'] = scaler
            self.scalers['scaled_columns'] = cols_to_scale
            
            self.log_feature(
                'scaled_features',
                {
                    'columns_scaled': len(cols_to_scale),
                    'scaler_type': 'StandardScaler'
                }
            )
    
    def run_full_engineering(self) -> pd.DataFrame:
        """Execute complete feature engineering pipeline"""
        print("=" * 60)
        print("NETFLIX FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        
        self.load_data()
        self.create_title_tokens()
        self.create_title_embeddings()  # May skip if no internet
        self.encode_genres()
        self.calculate_director_popularity()
        self.calculate_cast_features()
        self.encode_countries()
        self.encode_rating()
        self.create_temporal_features()
        self.create_content_features()
        self.create_interaction_features()
        self.scale_numeric_features()
        
        return self.df
    
    def save_features(self, filename: str = "netflix_features.csv"):
        """Save engineered features"""
        output_path = self.output_dir / filename
        self.df.to_csv(output_path, index=False)
        print(f"\n✓ Features saved to {output_path}")
        
        # Save encoders and scalers
        artifacts_path = self.output_dir / "feature_artifacts.joblib"
        joblib.dump({
            'encoders': self.encoders,
            'scalers': self.scalers
        }, artifacts_path)
        print(f"✓ Encoders/scalers saved to {artifacts_path}")
        
        # Save feature log
        log_path = Path("logs") / "feature_engineering_log.json"
        with open(log_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'input_file': str(self.input_path),
                'output_file': str(output_path),
                'features_created': self.feature_log,
                'final_shape': {
                    'rows': len(self.df),
                    'columns': len(self.df.columns)
                }
            }, f, indent=2)
        print(f"✓ Feature log saved to {log_path}")
        
        # Save feature list
        feature_list_path = self.output_dir / "feature_list.txt"
        with open(feature_list_path, 'w') as f:
            f.write("NETFLIX FEATURE LIST\n")
            f.write("=" * 60 + "\n\n")
            
            for col in self.df.columns:
                f.write(f"{col}\n")
                f.write(f"  Type: {self.df[col].dtype}\n")
                f.write(f"  Non-null: {self.df[col].notna().sum()}\n")
                if self.df[col].dtype in ['int64', 'float64']:
                    f.write(f"  Mean: {self.df[col].mean():.2f}\n")
                    f.write(f"  Std: {self.df[col].std():.2f}\n")
                f.write("\n")
        
        print(f"✓ Feature list saved to {feature_list_path}")


def main():
    """Main execution"""
    engineer = FeatureEngineer(
        input_path="data/processed/netflix_cleaned.csv",
        output_dir="data/features"
    )
    
    features_df = engineer.run_full_engineering()
    engineer.save_features()
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print(f"Final dataset: {len(features_df)} rows × {len(features_df.columns)} columns")
    print("=" * 60)


if __name__ == "__main__":
    main()
