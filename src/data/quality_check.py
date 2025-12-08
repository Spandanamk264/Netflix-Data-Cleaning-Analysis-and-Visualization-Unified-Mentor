"""
Data Quality Assessment Module
Generates comprehensive data quality reports for Netflix dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataQualityChecker:
    """Comprehensive data quality assessment"""
    
    def __init__(self, data_path: str, output_dir: str = "reports/data_quality"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.report = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load raw CSV data"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df
    
    def check_missing_values(self) -> Dict:
        """Analyze missing values"""
        print("\n=== Checking Missing Values ===")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_report = {
            'total_rows': len(self.df),
            'columns': {}
        }
        
        for col in self.df.columns:
            missing_report['columns'][col] = {
                'missing_count': int(missing[col]),
                'missing_percentage': round(missing_pct[col], 2),
                'dtype': str(self.df[col].dtype)
            }
            if missing[col] > 0:
                print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
        
        return missing_report
    
    def check_duplicates(self) -> Dict:
        """Check for duplicate records"""
        print("\n=== Checking Duplicates ===")
        
        # Full duplicates
        full_dupes = self.df.duplicated().sum()
        
        # Duplicates by show_id
        if 'show_id' in self.df.columns:
            id_dupes = self.df.duplicated(subset=['show_id']).sum()
        else:
            id_dupes = 0
        
        # Duplicates by title
        if 'title' in self.df.columns:
            title_dupes = self.df.duplicated(subset=['title']).sum()
        else:
            title_dupes = 0
        
        duplicate_report = {
            'full_duplicates': int(full_dupes),
            'show_id_duplicates': int(id_dupes),
            'title_duplicates': int(title_dupes)
        }
        
        print(f"  Full duplicates: {full_dupes}")
        print(f"  Show ID duplicates: {id_dupes}")
        print(f"  Title duplicates: {title_dupes}")
        
        return duplicate_report
    
    def check_data_types(self) -> Dict:
        """Verify data types and formats"""
        print("\n=== Checking Data Types ===")
        
        dtype_report = {}
        for col in self.df.columns:
            dtype_report[col] = {
                'dtype': str(self.df[col].dtype),
                'unique_values': int(self.df[col].nunique()),
                'sample_values': self.df[col].dropna().head(3).tolist()
            }
            print(f"  {col}: {self.df[col].dtype} ({self.df[col].nunique()} unique)")
        
        return dtype_report
    
    def check_categorical_consistency(self) -> Dict:
        """Check consistency in categorical fields"""
        print("\n=== Checking Categorical Consistency ===")
        
        consistency_report = {}
        
        # Check 'type' column
        if 'type' in self.df.columns:
            type_values = self.df['type'].value_counts().to_dict()
            consistency_report['type'] = {
                'unique_values': list(type_values.keys()),
                'counts': {str(k): int(v) for k, v in type_values.items()}
            }
            print(f"  Type values: {list(type_values.keys())}")
        
        # Check 'rating' column
        if 'rating' in self.df.columns:
            rating_values = self.df['rating'].value_counts()
            consistency_report['rating'] = {
                'unique_values': rating_values.index.tolist(),
                'counts': {str(k): int(v) for k, v in rating_values.items()},
                'total_unique': int(rating_values.nunique())
            }
            print(f"  Rating values ({len(rating_values)} unique): {rating_values.index.tolist()[:10]}")
        
        # Check 'country' consistency
        if 'country' in self.df.columns:
            country_sample = self.df['country'].dropna().head(10).tolist()
            consistency_report['country'] = {
                'sample_values': country_sample,
                'contains_multiple': any(',' in str(c) for c in country_sample if pd.notna(c))
            }
            print(f"  Country sample: {country_sample[:3]}")
        
        return consistency_report
    
    def check_date_formats(self) -> Dict:
        """Validate date fields"""
        print("\n=== Checking Date Formats ===")
        
        date_report = {}
        
        if 'date_added' in self.df.columns:
            date_col = self.df['date_added'].dropna()
            
            # Try parsing dates
            parsed_dates = pd.to_datetime(date_col, errors='coerce')
            invalid_dates = parsed_dates.isnull().sum()
            
            date_report['date_added'] = {
                'total_non_null': int(len(date_col)),
                'invalid_formats': int(invalid_dates),
                'valid_percentage': round((len(date_col) - invalid_dates) / len(date_col) * 100, 2),
                'sample_values': date_col.head(5).tolist(),
                'date_range': {
                    'min': str(parsed_dates.min()) if not parsed_dates.isnull().all() else None,
                    'max': str(parsed_dates.max()) if not parsed_dates.isnull().all() else None
                }
            }
            print(f"  date_added: {invalid_dates} invalid formats")
            print(f"  Date range: {parsed_dates.min()} to {parsed_dates.max()}")
        
        if 'release_year' in self.df.columns:
            year_col = self.df['release_year'].dropna()
            date_report['release_year'] = {
                'min': int(year_col.min()),
                'max': int(year_col.max()),
                'mean': round(float(year_col.mean()), 2),
                'outliers': int(((year_col < 1900) | (year_col > 2025)).sum())
            }
            print(f"  release_year: {year_col.min()} to {year_col.max()}")
        
        return date_report
    
    def check_duration_formats(self) -> Dict:
        """Validate duration field formats"""
        print("\n=== Checking Duration Formats ===")
        
        duration_report = {}
        
        if 'duration' in self.df.columns:
            duration_col = self.df['duration'].dropna()
            
            # Check for 'min' and 'Season' patterns
            has_min = duration_col.str.contains('min', na=False).sum()
            has_season = duration_col.str.contains('Season', na=False).sum()
            malformed = len(duration_col) - has_min - has_season
            
            duration_report = {
                'total_non_null': int(len(duration_col)),
                'minutes_format': int(has_min),
                'seasons_format': int(has_season),
                'malformed': int(malformed),
                'sample_values': duration_col.head(10).tolist()
            }
            
            print(f"  Minutes format: {has_min}")
            print(f"  Seasons format: {has_season}")
            print(f"  Malformed: {malformed}")
        
        return duration_report
    
    def check_outliers(self) -> Dict:
        """Detect outliers in numeric fields"""
        print("\n=== Checking Outliers ===")
        
        outlier_report = {}
        
        if 'release_year' in self.df.columns:
            year_col = self.df['release_year'].dropna()
            Q1 = year_col.quantile(0.25)
            Q3 = year_col.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((year_col < Q1 - 1.5 * IQR) | (year_col > Q3 + 1.5 * IQR)).sum()
            
            outlier_report['release_year'] = {
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR),
                'outlier_count': int(outliers),
                'outlier_percentage': round(outliers / len(year_col) * 100, 2)
            }
            print(f"  release_year outliers: {outliers} ({outliers/len(year_col)*100:.2f}%)")
        
        return outlier_report
    
    def check_text_fields(self) -> Dict:
        """Analyze text field quality"""
        print("\n=== Checking Text Fields ===")
        
        text_report = {}
        
        text_columns = ['title', 'director', 'listed_in', 'country']
        
        for col in text_columns:
            if col in self.df.columns:
                col_data = self.df[col].dropna()
                
                text_report[col] = {
                    'avg_length': round(col_data.str.len().mean(), 2),
                    'max_length': int(col_data.str.len().max()),
                    'min_length': int(col_data.str.len().min()),
                    'empty_strings': int((col_data == '').sum()),
                    'sample': col_data.head(3).tolist()
                }
                print(f"  {col}: avg length {text_report[col]['avg_length']}")
        
        return text_report
    
    def generate_summary_statistics(self) -> Dict:
        """Generate overall summary statistics"""
        print("\n=== Summary Statistics ===")
        
        summary = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
            'columns': list(self.df.columns),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'object_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        
        print(f"  Total records: {summary['total_records']}")
        print(f"  Total columns: {summary['total_columns']}")
        print(f"  Memory usage: {summary['memory_usage_mb']} MB")
        
        return summary
    
    def run_full_assessment(self) -> Dict:
        """Run complete data quality assessment"""
        print("=" * 60)
        print("NETFLIX DATA QUALITY ASSESSMENT")
        print("=" * 60)
        
        self.load_data()
        
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'dataset': str(self.data_path),
            'summary': self.generate_summary_statistics(),
            'missing_values': self.check_missing_values(),
            'duplicates': self.check_duplicates(),
            'data_types': self.check_data_types(),
            'categorical_consistency': self.check_categorical_consistency(),
            'date_formats': self.check_date_formats(),
            'duration_formats': self.check_duration_formats(),
            'outliers': self.check_outliers(),
            'text_fields': self.check_text_fields()
        }
        
        return self.report
    
    def save_report(self, filename: str = "data_quality_report.json"):
        """Save report to JSON file"""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.report, f, indent=2)
        print(f"\n✓ Report saved to {output_path}")
        
        # Also save a human-readable summary
        summary_path = self.output_dir / "data_quality_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("NETFLIX DATA QUALITY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {self.report['timestamp']}\n")
            f.write(f"Dataset: {self.report['dataset']}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 60 + "\n")
            for key, value in self.report['summary'].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n\nMISSING VALUES\n")
            f.write("-" * 60 + "\n")
            for col, info in self.report['missing_values']['columns'].items():
                if info['missing_count'] > 0:
                    f.write(f"{col}: {info['missing_count']} ({info['missing_percentage']}%)\n")
            
            f.write("\n\nDUPLICATES\n")
            f.write("-" * 60 + "\n")
            for key, value in self.report['duplicates'].items():
                f.write(f"{key}: {value}\n")
        
        print(f"✓ Summary saved to {summary_path}")


def main():
    """Main execution"""
    checker = DataQualityChecker(
        data_path="data/raw/netflix1.csv",
        output_dir="reports/data_quality"
    )
    
    report = checker.run_full_assessment()
    checker.save_report()
    
    print("\n" + "=" * 60)
    print("DATA QUALITY ASSESSMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
