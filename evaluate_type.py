import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Data
print("Loading data...")
df = pd.read_csv('data/features/netflix_features.csv')
print(f"Loaded {len(df)} rows")

# Load Model
print("Loading Type Classifier...")
model = joblib.load('models/type_classifier_xgb.joblib')

# Prepare Data (Same logic as training)
exclude_cols = ['show_id', 'type', 'title', 'date_added', 'date_added_parsed', 'description', 
                'director', 'cast', 'country', 'listed_in', 'rating', 'type_encoded', 
                'primary_country', 'primary_genre', 'rating_bucket', 'duration', 'is_movie', 'is_tv_show']
feature_cols = [c for c in df.columns if c not in exclude_cols]
# Filter numeric only
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

X = df[feature_cols]
le = LabelEncoder()
y = le.fit_transform(df['type'])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Predict
print("Predicting...")
preds = model.predict(X_test)

# Report
acc = accuracy_score(y_test, preds)
print(f"\n✅ Type Classifier Accuracy: {acc*100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, preds, target_names=le.classes_))
