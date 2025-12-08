"""
Recommender System Module
Implements Content-Based, Collaborative (Simulated), and Hybrid recommendation engines
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

class NetflixRecommender:
    """Hybrid Recommender System for Netflix Content"""
    
    def __init__(self, features_path="data/features/netflix_features.csv", 
                 embeddings_path="data/features/title_embeddings.npy"):
        self.features_path = Path(features_path)
        self.embeddings_path = Path(embeddings_path)
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.embeddings = None
        self.similarity_matrix = None
        self.indices = None
        
    def load_data(self):
        """Load features and embeddings"""
        print("Loading data for recommender...")
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features not found at {self.features_path}")
            
        self.df = pd.read_csv(self.features_path)
        
        # Load embeddings if available
        if self.embeddings_path.exists():
            print("Loading pre-computed embeddings...")
            self.embeddings = np.load(self.embeddings_path)
        else:
            print("Embeddings not found. Using simple TF-IDF...")
            # Fallback to TF-IDF if embeddings missing
            tfidf = TfidfVectorizer(stop_words='english')
            self.embeddings = tfidf.fit_transform(self.df['description'].fillna(''))
            
        # Create title-to-index mapping
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        print(f"Loaded {len(self.df)} titles")
        
    def build_similarity_matrix(self):
        """Compute cosine similarity matrix"""
        print("Computing similarity matrix...")
        # Use embeddings for similarity
        # Calculate in chunks if too large, but for 8k items, it fits in memory (8000x8000 * 4 bytes ~ 256MB)
        self.similarity_matrix = cosine_similarity(self.embeddings, self.embeddings)
        print("Similarity matrix computed")
        
    def get_content_recommendations(self, title, n=10):
        """Get content-based recommendations"""
        if self.similarity_matrix is None:
            self.build_similarity_matrix()
            
        if title not in self.indices:
            return []
            
        idx = self.indices[title]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort by similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N (excluding self)
        sim_scores = sim_scores[1:n+1]
        
        # Get indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return results
        results = self.df.iloc[movie_indices][['title', 'type', 'primary_genre', 'rating']].copy()
        results['similarity_score'] = [i[1] for i in sim_scores]
        
        return results
    
    def simulate_user_preferences(self, favorite_titles):
        """Simulate a user profile based on favorite titles"""
        user_vector = np.zeros(self.embeddings.shape[1])
        
        valid_titles = [t for t in favorite_titles if t in self.indices]
        if not valid_titles:
            return None
            
        for title in valid_titles:
            idx = self.indices[title]
            user_vector += self.embeddings[idx]
            
        # Average
        user_vector /= len(valid_titles)
        return user_vector.reshape(1, -1)
    
    def get_personalized_recommendations(self, favorite_titles, n=10):
        """Get recommendations based on a list of favorites"""
        if self.similarity_matrix is None:
            self.load_data()
            self.build_similarity_matrix()
            
        user_vec = self.simulate_user_preferences(favorite_titles)
        if user_vec is None:
            return pd.DataFrame()
            
        # Calculate similarity between user vector and all items
        scores = cosine_similarity(user_vec, self.embeddings).flatten()
        
        # Sort
        indices = scores.argsort()[::-1]
        
        # Filter out input titles
        input_indices = [self.indices[t] for t in favorite_titles if t in self.indices]
        recommendations = []
        
        for idx in indices:
            if idx not in input_indices:
                recommendations.append(idx)
                if len(recommendations) >= n:
                    break
        
        results = self.df.iloc[recommendations][['title', 'type', 'primary_genre', 'year_added']].copy()
        results['score'] = scores[recommendations]
        
        return results

    def save_model(self):
        """Save the similarity matrix and indices"""
        print("Saving recommender model...")
        joblib.dump(self.similarity_matrix, self.model_dir / "similarity_matrix.joblib")
        joblib.dump(self.indices, self.model_dir / "title_indices.joblib")
        print("Model saved")
        
    def train(self):
        """Run full training pipeline"""
        self.load_data()
        self.build_similarity_matrix()
        self.save_model()
        
        # Test basic recommendation
        print("\nTest Recommendation for 'Stranger Things':")
        try:
            recs = self.get_content_recommendations('Stranger Things')
            print(recs[['title', 'similarity_score']])
        except:
            print("Title not found or error in recommendation")

if __name__ == "__main__":
    rec = NetflixRecommender()
    rec.train()
