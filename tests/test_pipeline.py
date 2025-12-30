
import pytest
import pandas as pd
import numpy as np
from src.data.quality_audit import DataQualityAuditor
import os

# Create dummy data for testing
@pytest.fixture
def sample_csv(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test.csv"
    df = pd.DataFrame({
        'show_id': ['s1', 's2'],
        'type': ['Movie', 'TV Show'],
        'title': ['Test Movie', 'Test Show'],
        'release_year': [2020, 1900], # 1 outlier
        'date_added': ['January 1, 2020', 'bad date'],
        'duration': ['90 min', '2 Seasons']
    })
    df.to_csv(p, index=False)
    return str(p)

def test_quality_audit(sample_csv):
    auditor = DataQualityAuditor(sample_csv)
    auditor.load_data()
    auditor.check_malformed_dates()
    
    # Expect 1 malformed date
    malformed_issue = [i for i in auditor.report['issues'] if i['type'] == 'malformed_dates']
    assert len(malformed_issue) == 1
    assert malformed_issue[0]['count'] == 1

def test_feature_generation():
    # Test if dummy frame produces correct feature shapes
    df = pd.DataFrame({
        'listed_in': ['Drama, Action', 'Comedy'],
        'country': ['USA', 'India']
    })
    
    # Mock One-Hot
    df['genre_list'] = df['listed_in'].str.split(', ')
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    res = mlb.fit_transform(df['genre_list'])
    
    assert res.shape == (2, 3) # Drama, Action, Comedy
    assert 'Comedy' in mlb.classes_

def test_model_artifacts_exist():
    # Check if critical models are saved
    required = [
        "models/classification/xgb_type_classifier.joblib",
        "models/forecasting/prophet_model.joblib"
    ]
    for p in required:
        assert os.path.exists(p), f"Model artifact {p} missing"
