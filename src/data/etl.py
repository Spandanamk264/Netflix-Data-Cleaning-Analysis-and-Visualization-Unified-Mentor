"""
ETL Module
Extracts data from cleaned CSV, transforms to normalized schema, and loads into PostgreSQL
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

import warnings
warnings.filterwarnings('ignore')

class NetflixETL:
    """ETL Pipeline for Netflix Project"""
    
    def __init__(self, db_url=None):
        load_dotenv()
        self.db_url = db_url or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/netflix_db")
        self.engine = create_engine(self.db_url)
        self.df = None
        
    def extract(self, file_path="data/processed/netflix_cleaned.csv"):
        """Extract data from CSV"""
        print(f"Extracting data from {file_path}...")
        self.df = pd.read_csv(file_path)
        
        # Ensure date format
        self.df['date_added_parsed'] = pd.to_datetime(self.df['date_added_parsed']).dt.date
        print(f"Loaded {len(self.df)} records")
        
    def init_schema(self, schema_path="database/schema/schema.sql"):
        """Initialize database schema"""
        print("Initializing schema...")
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
            
        with self.engine.connect() as conn:
            # Execute statements individually
            for statement in schema_sql.split(';'):
                if statement.strip():
                    conn.execute(text(statement))
            conn.commit()
        print("Schema initialized successfully")

    def load_titles(self):
        """Load titles table"""
        print("Loading titles...")
        titles_df = self.df[[
            'show_id', 'type', 'title', 'date_added_parsed', 
            'release_year', 'rating', 'duration_minutes', 'season_count'
        ]].copy()
        
        titles_df.rename(columns={'date_added_parsed': 'date_added'}, inplace=True)
        
        # Handle description if it exists (not in current cleaned csv, but good to have)
        if 'description' not in titles_df.columns:
            titles_df['description'] = None
            
        titles_df.to_sql('titles', self.engine, if_exists='append', index=False)
        print(f"Loaded {len(titles_df)} titles")

    def load_dimensions(self):
        """Load lookup tables (People, Genres, Countries)"""
        print("Loading dimensions...")
        
        with self.engine.connect() as conn:
            # 1. Genres
            print("  Processing genres...")
            unique_genres = set()
            for items in self.df['listed_in'].str.split(', '):
                if isinstance(items, list):
                    unique_genres.update([g.strip() for g in items])
            
            genres_df = pd.DataFrame({'name': list(unique_genres)})
            genres_df.to_sql('genres', self.engine, if_exists='append', index=False)
            
            # 2. Countries
            print("  Processing countries...")
            unique_countries = set()
            for items in self.df['country'].str.split(', '):
                if isinstance(items, list):
                    unique_countries.update([c.strip() for c in items if c != 'Unknown Country'])
            
            countries_df = pd.DataFrame({'name': list(unique_countries)})
            countries_df.to_sql('countries', self.engine, if_exists='append', index=False)
            
            # 3. People (Directors + Cast)
            print("  Processing people...")
            unique_people = set()
            
            # Directors
            for items in self.df['director'].str.split(', '):
                if isinstance(items, list):
                    unique_people.update([p.strip() for p in items if p != 'Unknown Director'])
                    
            # Cast
            for items in self.df['cast'].str.split(', '):
                if isinstance(items, list):
                    unique_people.update([p.strip() for p in items if p != 'Unknown Cast'])
            
            people_df = pd.DataFrame({'name': list(unique_people)})
            people_df.to_sql('people', self.engine, if_exists='append', index=False)
            
        print("Dimensions loaded")

    def load_junctions(self):
        """Load junction tables"""
        print("Loading junction tables (this may take a moment)...")
        
        # Load necessary IDs from DB
        with self.engine.connect() as conn:
            genres_map = pd.read_sql("SELECT name, genre_id FROM genres", conn).set_index('name')['genre_id'].to_dict()
            countries_map = pd.read_sql("SELECT name, country_id FROM countries", conn).set_index('name')['country_id'].to_dict()
            people_map = pd.read_sql("SELECT name, person_id FROM people", conn).set_index('name')['person_id'].to_dict()

        # Helper to prepare junction data
        def prepare_junction(column, map_dict, id_col_name):
            data = []
            for _, row in self.df.iterrows():
                if pd.notna(row[column]):
                    items = [x.strip() for x in row[column].split(', ')]
                    for item in items:
                        if item in map_dict:
                            data.append({'show_id': row['show_id'], id_col_name: map_dict[item]})
            return pd.DataFrame(data).drop_duplicates()

        # 1. Title-Genre
        tg_df = prepare_junction('listed_in', genres_map, 'genre_id')
        tg_df.to_sql('title_genre', self.engine, if_exists='append', index=False)
        print(f"  Loaded {len(tg_df)} genre relations")
        
        # 2. Title-Country
        tc_df = prepare_junction('country', countries_map, 'country_id')
        tc_df.to_sql('title_country', self.engine, if_exists='append', index=False)
        print(f"  Loaded {len(tc_df)} country relations")
        
        # 3. Title-Person (Director & Cast)
        people_data = []
        for _, row in self.df.iterrows():
            # Directors
            if pd.notna(row['director']):
                items = [x.strip() for x in row['director'].split(', ')]
                for item in items:
                    if item in people_map:
                        people_data.append({
                            'show_id': row['show_id'], 
                            'person_id': people_map[item],
                            'role': 'Director'
                        })
            # Cast
            if pd.notna(row['cast']):
                items = [x.strip() for x in row['cast'].split(', ')]
                for item in items:
                    if item in people_map:
                        people_data.append({
                            'show_id': row['show_id'], 
                            'person_id': people_map[item],
                            'role': 'Cast'
                        })
                        
        tp_df = pd.DataFrame(people_data).drop_duplicates()
        tp_df.to_sql('title_person', self.engine, if_exists='append', index=False)
        print(f"  Loaded {len(tp_df)} person relations")

    def run_pipeline(self):
        """Run full ETL pipeline"""
        try:
            self.extract()
            self.init_schema()
            self.load_titles()
            self.load_dimensions()
            self.load_junctions()
            print("\nETL Pipeline completed successfully!")
        except Exception as e:
            print(f"\nError in ETL pipeline: {e}")
            print("Ensure PostgreSQL is running and credentials are correct in .env")

if __name__ == "__main__":
    etl = NetflixETL()
    etl.run_pipeline()
