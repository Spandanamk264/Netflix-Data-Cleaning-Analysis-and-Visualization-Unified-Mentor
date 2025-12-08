"""
Data Cleaning Module
Implements deterministic cleaning steps with full transformation logging
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import re
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Deterministic data cleaning with transformation logging"""
    
    def __init__(self, input_path: str, output_dir: str = "data/processed"):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.changelog = []
        
    def log_transformation(self, step: str, details: Dict):
        """Log each transformation step"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'details': details
        }
        self.changelog.append(entry)
        print(f"  ✓ {step}: {details.get('description', '')}")
    
    def load_data(self) -> pd.DataFrame:
        """Load raw data"""
        print("Loading raw data...")
        self.df = pd.read_csv(self.input_path)
        self.log_transformation(
            "load_data",
            {
                'description': f'Loaded {len(self.df)} rows',
                'rows': len(self.df),
                'columns': len(self.df.columns)
            }
        )
        return self.df
    
    def remove_duplicates(self):
        """Remove duplicate records"""
        print("\n=== Removing Duplicates ===")
        initial_count = len(self.df)
        
        # Remove full duplicates
        self.df = self.df.drop_duplicates()
        full_dupes_removed = initial_count - len(self.df)
        
        # Remove show_id duplicates (keep first)
        if 'show_id' in self.df.columns:
            self.df = self.df.drop_duplicates(subset=['show_id'], keep='first')
            id_dupes_removed = initial_count - full_dupes_removed - len(self.df)
        else:
            id_dupes_removed = 0
        
        self.log_transformation(
            "remove_duplicates",
            {
                'description': f'Removed {full_dupes_removed + id_dupes_removed} duplicates',
                'full_duplicates': full_dupes_removed,
                'show_id_duplicates': id_dupes_removed,
                'final_count': len(self.df)
            }
        )
    
    def clean_missing_director(self):
        """Handle missing director values"""
        print("\n=== Cleaning Director Field ===")
        
        if 'director' not in self.df.columns:
            return
        
        missing_count = self.df['director'].isnull().sum()
        
        # Mark missing directors
        self.df['director'] = self.df['director'].fillna('Unknown Director')
        self.df['has_director'] = self.df['director'] != 'Unknown Director'
        
        self.log_transformation(
            "clean_missing_director",
            {
                'description': f'Filled {missing_count} missing directors',
                'missing_count': int(missing_count),
                'fill_value': 'Unknown Director',
                'added_indicator': 'has_director'
            }
        )
    
    def clean_missing_cast(self):
        """Handle missing cast values"""
        print("\n=== Cleaning Cast Field ===")
        
        if 'cast' not in self.df.columns:
            return
        
        missing_count = self.df['cast'].isnull().sum()
        
        # Mark missing cast
        self.df['cast'] = self.df['cast'].fillna('Unknown Cast')
        self.df['has_cast'] = self.df['cast'] != 'Unknown Cast'
        
        self.log_transformation(
            "clean_missing_cast",
            {
                'description': f'Filled {missing_count} missing cast entries',
                'missing_count': int(missing_count),
                'fill_value': 'Unknown Cast',
                'added_indicator': 'has_cast'
            }
        )
    
    def clean_missing_country(self):
        """Handle missing country values"""
        print("\n=== Cleaning Country Field ===")
        
        if 'country' not in self.df.columns:
            return
        
        missing_count = self.df['country'].isnull().sum()
        
        # Mark missing countries
        self.df['country'] = self.df['country'].fillna('Unknown Country')
        self.df['has_country'] = self.df['country'] != 'Unknown Country'
        
        self.log_transformation(
            "clean_missing_country",
            {
                'description': f'Filled {missing_count} missing countries',
                'missing_count': int(missing_count),
                'fill_value': 'Unknown Country',
                'added_indicator': 'has_country'
            }
        )
    
    def normalize_rating(self):
        """Normalize rating values"""
        print("\n=== Normalizing Rating Values ===")
        
        if 'rating' not in self.df.columns:
            return
        
        # Map of rating normalizations
        rating_map = {
            'TV-Y': 'TV-Y',
            'TV-Y7': 'TV-Y7',
            'TV-Y7-FV': 'TV-Y7-FV',
            'TV-G': 'TV-G',
            'TV-PG': 'TV-PG',
            'TV-14': 'TV-14',
            'TV-MA': 'TV-MA',
            'G': 'G',
            'PG': 'PG',
            'PG-13': 'PG-13',
            'R': 'R',
            'NC-17': 'NC-17',
            'NR': 'NR',
            'UR': 'UR'
        }
        
        missing_count = self.df['rating'].isnull().sum()
        
        # Normalize ratings
        self.df['rating'] = self.df['rating'].fillna('NR')  # Not Rated
        self.df['rating'] = self.df['rating'].str.strip()
        
        # Handle any non-standard ratings
        valid_ratings = set(rating_map.keys())
        invalid_mask = ~self.df['rating'].isin(valid_ratings)
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            self.df.loc[invalid_mask, 'rating'] = 'NR'
        
        self.log_transformation(
            "normalize_rating",
            {
                'description': f'Normalized {missing_count + invalid_count} ratings',
                'missing_filled': int(missing_count),
                'invalid_normalized': int(invalid_count),
                'valid_ratings': list(valid_ratings)
            }
        )
    
    def parse_date_added(self):
        """Parse and standardize date_added field"""
        print("\n=== Parsing Date Added ===")
        
        if 'date_added' not in self.df.columns:
            return
        
        missing_count = self.df['date_added'].isnull().sum()
        
        # Parse dates
        self.df['date_added_parsed'] = pd.to_datetime(
            self.df['date_added'], 
            errors='coerce'
        )
        
        # Extract temporal features
        self.df['year_added'] = self.df['date_added_parsed'].dt.year
        self.df['month_added'] = self.df['date_added_parsed'].dt.month
        self.df['day_added'] = self.df['date_added_parsed'].dt.day
        self.df['day_of_week_added'] = self.df['date_added_parsed'].dt.dayofweek
        self.df['is_weekend_add'] = self.df['day_of_week_added'].isin([5, 6])
        
        invalid_dates = self.df['date_added_parsed'].isnull().sum() - missing_count
        
        self.log_transformation(
            "parse_date_added",
            {
                'description': f'Parsed date_added with {invalid_dates} invalid formats',
                'missing_count': int(missing_count),
                'invalid_formats': int(invalid_dates),
                'new_columns': ['date_added_parsed', 'year_added', 'month_added', 
                               'day_added', 'day_of_week_added', 'is_weekend_add']
            }
        )
    
    def parse_duration(self):
        """Parse and convert duration to numeric features"""
        print("\n=== Parsing Duration ===")
        
        if 'duration' not in self.df.columns:
            return
        
        # Initialize columns
        self.df['duration_minutes'] = np.nan
        self.df['season_count'] = np.nan
        
        # Extract minutes for movies
        movie_mask = self.df['duration'].str.contains('min', na=False)
        self.df.loc[movie_mask, 'duration_minutes'] = (
            self.df.loc[movie_mask, 'duration']
            .str.extract(r'(\d+)')[0]
            .astype(float)
        )
        
        # Extract seasons for TV shows
        season_mask = self.df['duration'].str.contains('Season', na=False)
        self.df.loc[season_mask, 'season_count'] = (
            self.df.loc[season_mask, 'duration']
            .str.extract(r'(\d+)')[0]
            .astype(float)
        )
        
        movies_parsed = movie_mask.sum()
        seasons_parsed = season_mask.sum()
        
        self.log_transformation(
            "parse_duration",
            {
                'description': f'Parsed {movies_parsed} movie durations and {seasons_parsed} TV seasons',
                'movies_parsed': int(movies_parsed),
                'seasons_parsed': int(seasons_parsed),
                'new_columns': ['duration_minutes', 'season_count']
            }
        )
    
    def calculate_release_age(self):
        """Calculate release_age_at_add (date_added - release_year)"""
        print("\n=== Calculating Release Age ===")
        
        if 'release_year' not in self.df.columns or 'date_added_parsed' not in self.df.columns:
            return
        
        # Calculate age in days
        self.df['release_year_date'] = pd.to_datetime(
            self.df['release_year'].astype(str) + '-01-01',
            errors='coerce'
        )
        
        self.df['release_age_days'] = (
            self.df['date_added_parsed'] - self.df['release_year_date']
        ).dt.days
        
        self.df['release_age_years'] = self.df['release_age_days'] / 365.25
        
        valid_count = self.df['release_age_days'].notna().sum()
        
        self.log_transformation(
            "calculate_release_age",
            {
                'description': f'Calculated release age for {valid_count} records',
                'valid_count': int(valid_count),
                'new_columns': ['release_age_days', 'release_age_years']
            }
        )
    
    def calculate_days_since_first(self):
        """Calculate days since first record in dataset"""
        print("\n=== Calculating Days Since First Record ===")
        
        if 'date_added_parsed' not in self.df.columns:
            return
        
        first_date = self.df['date_added_parsed'].min()
        self.df['days_since_first_record'] = (
            self.df['date_added_parsed'] - first_date
        ).dt.days
        
        self.log_transformation(
            "calculate_days_since_first",
            {
                'description': f'Calculated days since {first_date}',
                'first_date': str(first_date),
                'new_column': 'days_since_first_record'
            }
        )
    
    def clean_text_fields(self):
        """Clean and standardize text fields"""
        print("\n=== Cleaning Text Fields ===")
        
        text_columns = ['title', 'director', 'cast', 'country', 'listed_in']
        cleaned_count = 0
        
        for col in text_columns:
            if col in self.df.columns:
                # Strip whitespace
                self.df[col] = self.df[col].str.strip()
                
                # Remove extra spaces
                self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
                
                cleaned_count += 1
        
        self.log_transformation(
            "clean_text_fields",
            {
                'description': f'Cleaned {cleaned_count} text columns',
                'columns_cleaned': [col for col in text_columns if col in self.df.columns]
            }
        )
    
    def add_basic_features(self):
        """Add basic derived features"""
        print("\n=== Adding Basic Features ===")
        
        new_features = []
        
        # Title length
        if 'title' in self.df.columns:
            self.df['title_length'] = self.df['title'].str.len()
            new_features.append('title_length')
        
        # Number of genres
        if 'listed_in' in self.df.columns:
            self.df['number_of_genres'] = (
                self.df['listed_in']
                .str.split(',')
                .apply(lambda x: len(x) if isinstance(x, list) else 0)
            )
            new_features.append('number_of_genres')
        
        # Number of countries
        if 'country' in self.df.columns:
            self.df['number_of_countries'] = (
                self.df['country']
                .apply(lambda x: len(str(x).split(',')) if pd.notna(x) and x != 'Unknown Country' else 0)
            )
            new_features.append('number_of_countries')
        
        # Cast count
        if 'cast' in self.df.columns:
            self.df['cast_count'] = (
                self.df['cast']
                .apply(lambda x: len(str(x).split(',')) if pd.notna(x) and x != 'Unknown Cast' else 0)
            )
            new_features.append('cast_count')
        
        self.log_transformation(
            "add_basic_features",
            {
                'description': f'Added {len(new_features)} basic features',
                'new_features': new_features
            }
        )
    
    def run_full_cleaning(self) -> pd.DataFrame:
        """Execute complete cleaning pipeline"""
        print("=" * 60)
        print("NETFLIX DATA CLEANING PIPELINE")
        print("=" * 60)
        
        self.load_data()
        self.remove_duplicates()
        self.clean_missing_director()
        self.clean_missing_cast()
        self.clean_missing_country()
        self.normalize_rating()
        self.parse_date_added()
        self.parse_duration()
        self.calculate_release_age()
        self.calculate_days_since_first()
        self.clean_text_fields()
        self.add_basic_features()
        
        return self.df
    
    def save_cleaned_data(self, filename: str = "netflix_cleaned.csv"):
        """Save cleaned dataset"""
        output_path = self.output_dir / filename
        self.df.to_csv(output_path, index=False)
        print(f"\n✓ Cleaned data saved to {output_path}")
        
        # Save changelog
        changelog_path = Path("logs") / "transformation_changelog.json"
        changelog_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(changelog_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'input_file': str(self.input_path),
                'output_file': str(output_path),
                'transformations': self.changelog,
                'final_shape': {
                    'rows': len(self.df),
                    'columns': len(self.df.columns)
                }
            }, f, indent=2)
        
        print(f"✓ Changelog saved to {changelog_path}")
        
        # Save data dictionary
        data_dict_path = self.output_dir / "data_dictionary.txt"
        with open(data_dict_path, 'w') as f:
            f.write("NETFLIX CLEANED DATA DICTIONARY\n")
            f.write("=" * 60 + "\n\n")
            
            for col in self.df.columns:
                f.write(f"{col}\n")
                f.write(f"  Type: {self.df[col].dtype}\n")
                f.write(f"  Non-null: {self.df[col].notna().sum()}\n")
                f.write(f"  Unique: {self.df[col].nunique()}\n")
                if self.df[col].dtype == 'object':
                    f.write(f"  Sample: {self.df[col].dropna().head(2).tolist()}\n")
                f.write("\n")
        
        print(f"✓ Data dictionary saved to {data_dict_path}")


def main():
    """Main execution"""
    cleaner = DataCleaner(
        input_path="data/raw/netflix1.csv",
        output_dir="data/processed"
    )
    
    cleaned_df = cleaner.run_full_cleaning()
    cleaner.save_cleaned_data()
    
    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE")
    print(f"Final dataset: {len(cleaned_df)} rows × {len(cleaned_df.columns)} columns")
    print("=" * 60)


if __name__ == "__main__":
    main()
