
import pandas as pd
import numpy as np
from loguru import logger
import json
import os
from datetime import datetime

# Configure Logging
logger.add("logs/data_quality.log", rotation="500 MB")

class DataQualityAuditor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "file": file_path,
            "issues": [],
            "summary": {}
        }
    
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Loaded data from {self.file_path} with shape {self.df.shape}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def check_missing_values(self):
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].to_dict()
        if missing:
            self.report["issues"].append({"type": "missing_values", "details": missing})
            logger.warning(f"Found missing values: {missing}")
    
    def check_duplicates(self):
        dupes = self.df.duplicated().sum()
        if dupes > 0:
            self.report["issues"].append({"type": "duplicates", "count": int(dupes)})
            logger.warning(f"Found {dupes} duplicate rows")

    def check_malformed_dates(self):
        # Specific check for Netflix date_added format
        if 'date_added' in self.df.columns:
            # Try parsing to see failures
            parsed = pd.to_datetime(self.df['date_added'].str.strip(), errors='coerce')
            malformed = parsed.isnull().sum() - self.df['date_added'].isnull().sum()
            if malformed > 0:
                self.report["issues"].append({"type": "malformed_dates", "count": int(malformed)})
                logger.warning(f"Found {malformed} malformed dates")

    def check_outliers(self):
        # Check release_year outliers (e.g. future years or very old)
        if 'release_year' in self.df.columns:
            current_year = datetime.now().year
            future = self.df[self.df['release_year'] > current_year]
            ancient = self.df[self.df['release_year'] < 1900]
            
            if not future.empty:
                 self.report["issues"].append({"type": "outliers_release_year_future", "count": len(future)})
            if not ancient.empty:
                 self.report["issues"].append({"type": "outliers_release_year_ancient", "count": len(ancient)})

    def run_audit(self):
        self.load_data()
        self.check_missing_values()
        self.check_duplicates()
        self.check_malformed_dates()
        self.check_outliers()
        
        # Save Report
        os.makedirs("reports", exist_ok=True)
        report_path = f"reports/quality_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=4)
        
        logger.info(f"Audit completed. Report saved to {report_path}")
        print(json.dumps(self.report, indent=4))

if __name__ == "__main__":
    # Use relative path for better portability
    auditor = DataQualityAuditor(r"data/raw/netflix1.csv")
    auditor.run_audit()
