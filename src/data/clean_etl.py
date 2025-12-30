
import pandas as pd
import numpy as np
from loguru import logger
import re

# Configure logging
logger.add("logs/etl_changelog.log", rotation="10 MB")

INPUT_PATH = r"data/raw/netflix1.csv"
OUTPUT_PATH = r"data/processed/netflix_cleaned_v2.csv"

def deterministic_cleaning(input_path, output_path):
    logger.info("Starting deterministic cleaning pipeline")
    
    # 1. Load Data
    df = pd.read_csv(input_path)
    initial_shape = df.shape
    logger.info(f"Loaded {initial_shape[0]} rows, {initial_shape[1]} columns")

    # 2. Impute Missing Values (Deterministic)
    # Strategy: Unknown string for categorical, Mode/Median for others?
    # Requirement: "impute or mark missing director/cast/country"
    fill_values = {
        "director": "Unknown Director",
        "country": "Unknown Country",
        "rating": "UR" # Unrated
    }
    for col, val in fill_values.items():
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(val)
            logger.info(f"Imputed {missing_count} missing values in '{col}' with '{val}'")

    # 3. Normalize Rating Values
    # Fix odd ratings or consolidations if needed (e.g. '74 min', '84 min' appearing in rating column is a known dataset bug)
    # The Netflix dataset notoriously has durations in the rating column for some rows.
    # We must fix this.
    
    def fix_mixed_rating(row):
        rating = str(row['rating'])
        if 'min' in rating:
            # It's a duration, likely
            return "UR"
        return rating

    df['rating'] = df.apply(fix_mixed_rating, axis=1)
    logger.info("Normalized 'rating' column (removed duration artifacts)")

    # 4. Parse and Standardize date_added
    # Convert to datetime objects
    df['date_added_clean'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    failed_dates = df['date_added_clean'].isnull().sum()
    if failed_dates > 0:
        logger.warning(f"{failed_dates} rows have invalid 'date_added'. Dropping or Backfilling?")
        # For now, we drop rows where date_added is essential for time-series, or keeps them?
        # Requirement says "impute or mark". Let's keep them but mark.
    
    # 5. Duration Processing (Split Season vs Minutes)
    def parse_duration(val):
        val = str(val).lower().strip()
        if 'min' in val:
            return int(re.sub(r'[^0-9]', '', val)), 0
        elif 'season' in val:
            return 0, int(re.sub(r'[^0-9]', '', val))
        else:
            return 0, 0

    df[['duration_minutes', 'season_count']] = df['duration'].apply(lambda x: pd.Series(parse_duration(x)))
    
    # 6. Feature Engineering (Basic Pipeline Req)
    # "release_age_at_add (date_added - release_year)"
    df['year_added'] = df['date_added_clean'].dt.year
    df['release_age'] = df['year_added'] - df['release_year']
    
    # Handle negative ages (data errors where added before release)
    df['release_age'] = df['release_age'].apply(lambda x: 0 if x < 0 else x)

    # 7. Save
    df.to_csv(output_path, index=False)
    logger.success(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    deterministic_cleaning(INPUT_PATH, OUTPUT_PATH)
