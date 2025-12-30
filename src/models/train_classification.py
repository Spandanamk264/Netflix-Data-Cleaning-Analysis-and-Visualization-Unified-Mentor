
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
import shap
import joblib
from loguru import logger
import json
import os

logger.add("logs/training.log")

FEATURES_PATH = r"data/processed/netflix_features.csv"
MODEL_DIR = "models/classification"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_classifier():
    logger.info("Loading Data for Classification...")
    df = pd.read_csv(FEATURES_PATH)
    
    # Target: Type (Movie vs TV Show)
    logger.info("Task 1: Type Classification (Movie vs TV Show)")
    le_type = LabelEncoder()
    y_type = le_type.fit_transform(df['type'])
    
    # Features: duration_minutes, year_added, release_age, number_of_genres, director_popularity, embedding_dims?
    # Select numeric features only to allow XGBoost to run safely
    X = df.select_dtypes(include=[np.number]).fillna(0)
    # Exclude IDs or Target leakage if they were numeric (e.g. show_id is string so handled, year_added is good)
    # Ensure 'type' is not in X (it's string usually, but if encoded...)
    cols_to_drop = []
    for c in ['type', 'rating', 'description', 'title', 'director', 'cast', 'country', 'date_added', 'listed_in', 'genre_list']:
        if c in X.columns: cols_to_drop.append(c)
    X = X.drop(columns=cols_to_drop)
    
    # Debug
    logger.info(f"Training features: {X.columns.tolist()}")
    logger.info(f"Data Types:\n{X.dtypes}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_type, test_size=0.2, random_state=42, stratify=y_type)
    
    # Train XGBoost
    clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, eval_metric='logloss')
    clf.fit(X_train, y_train)
    
    # Evaluate
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    
    logger.info(f"Type Classification Results - F1: {f1:.4f}, AUC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=le_type.classes_))
    
    # Save Artifacts
    joblib.dump(clf, f"{MODEL_DIR}/xgb_type_classifier.joblib")
    joblib.dump(le_type, f"{MODEL_DIR}/type_encoder.joblib")
    
    # SHAP Values
    try:
        explainer = shap.TreeExplainer(clf)
        # shap_values = explainer.shap_values(X_test[:100]) # Commented out to prevent crash on some versions
        logger.info("SHAP explainer initialized (skipping calculation for stability)")
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
    
    # Task 2: Rating Bucket (Adult vs Family)
    # Define mapping
    def bucket_rating(r):
        if r in ['TV-MA', 'R', 'NC-17']: return 1 # Adult
        return 0 # Family/Teen
    
    y_rating = df['rating'].apply(bucket_rating)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_rating, test_size=0.2, random_state=42)
    
    clf_rating = xgb.XGBClassifier(n_estimators=100)
    clf_rating.fit(X_train_r, y_train_r)
    
    preds_r = clf_rating.predict(X_test_r)
    f1_r = f1_score(y_test_r, preds_r)
    logger.info(f"Rating Classification Results - F1: {f1_r:.4f}")
    
    joblib.dump(clf_rating, f"{MODEL_DIR}/xgb_rating_classifier.joblib")
    
    logger.success("Training Complete")

if __name__ == "__main__":
    import os
    train_classifier()
