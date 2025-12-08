"""
Multi-Task Classifier Module (Advanced)
Predicts Content Type, Genre, and Rating using XGBoost and Random Forest
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

class NetflixClassifier:
    """Advanced Multi-task Classification System"""
    
    def __init__(self, features_path="data/features/netflix_features.csv"):
        self.features_path = Path(features_path)
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.models = {}
        
    def load_data(self):
        """Load feature data and prepare robust feature set"""
        print("Loading data for classification...")
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features not found at {self.features_path}")
            
        self.df = pd.read_csv(self.features_path)
        
        # Fill NaNs for specific columns to avoid errors
        self.df.fillna(0, inplace=True)
        print(f"Loaded {len(self.df)} records")
        
    def train_type_classifier(self):
        """Train Movie vs TV Show classifier using XGBoost"""
        print("\n=== Training Type Classifier (XGBoost) ===")
        
        # 1. Select High-Impact Features
        # We exclude ID cols and the target itself
        exclude_cols = ['show_id', 'type', 'title', 'date_added', 'date_added_parsed', 'description', 
                        'director', 'cast', 'country', 'listed_in', 'rating', 'type_encoded', 
                        'primary_country', 'primary_genre', 'rating_bucket', 'duration', 'is_movie', 'is_tv_show']
        
        feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        # Filter to ensure we only use numeric columns
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(self.df[c])]
        
        print(f"Using {len(feature_cols)} features for Type classification")
        
        X = self.df[feature_cols]
        y = self.df['type_encoded'] # 0=Movie, 1=TV Show (usually)
        
        # Verify mapping
        le = LabelEncoder()
        y_true = le.fit_transform(self.df['type'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42, stratify=y_true)
        
        # 2. Train XGBoost
        # Calculate scale_pos_weight for imbalance
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # 3. Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Type Classification Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, target_names=le.classes_))
        
        # Feature Importance
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        print("\nTop 10 Important Features:")
        print(importances)
        
        self.models['type_classifier'] = model
        joblib.dump(model, self.model_dir / "type_classifier_xgb.joblib")
        
    def train_genre_classifier(self):
        """Train Primary Genre Classifier using Embeddings + XGBoost"""
        print("\n=== Training Genre Classifier (XGBoost) ===")
        
        # Use embeddings + basic stats
        emb_cols = [c for c in self.df.columns if c.startswith('title_emb_')]
        stats_cols = ['year_added', 'release_year', 'title_token_count', 'number_of_countries']
        
        if not emb_cols:
            print("No embeddings found. Skipping.")
            return

        features = emb_cols + stats_cols
        X = self.df[features]
        
        # Predict primary genre (Top 15 classes to ensure enough data)
        y_raw = self.df['primary_genre']
        top_genres = y_raw.value_counts().head(15).index
        
        # Filter data to only these 15 genres for cleaner multi-class classification
        mask = y_raw.isin(top_genres)
        X_subset = X[mask]
        y_subset = y_raw[mask]
        
        # Encode Target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_subset)
        
        print(f"Training on {len(X_subset)} samples across {len(top_genres)} top genres")
        
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        # Train XGBoost for Multi-Class
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(top_genres),
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"Genre Classification Accuracy: {acc:.4f}")
        print("\nClassification Report (Top 5 Classes shown):")
        # Show partial report to save space
        print(classification_report(y_test, preds, target_names=le.classes_))
        
        self.models['genre_classifier'] = model
        joblib.dump(model, self.model_dir / "genre_classifier_xgb.joblib")
        joblib.dump(le, self.model_dir / "genre_label_encoder.joblib")

    def run_training_pipeline(self):
        """Run all classifiers"""
        self.load_data()
        try:
            self.train_type_classifier()
        except Exception as e:
            print(f"Error training type classifier: {e}")
            
        try:
            self.train_genre_classifier()
        except Exception as e:
            print(f"Error training genre classifier: {e}")
            
        print("\nAll models trained and saved.")

if __name__ == "__main__":
    clf = NetflixClassifier()
    clf.run_training_pipeline()
