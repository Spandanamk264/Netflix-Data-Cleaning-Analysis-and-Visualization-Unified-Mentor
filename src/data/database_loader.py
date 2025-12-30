
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, Table
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from loguru import logger
import pandas as pd
import os

# Configure Logging
logger.add("logs/database.log", rotation="10 MB")

Base = declarative_base()

# --- Schema Definitions ---

# M2M Tables
title_genres = Table('title_genres', Base.metadata,
    Column('show_id', String, ForeignKey('titles.show_id'), primary_key=True),
    Column('genre_id', Integer, ForeignKey('genres.id'), primary_key=True)
)

title_countries = Table('title_countries', Base.metadata,
    Column('show_id', String, ForeignKey('titles.show_id'), primary_key=True),
    Column('country_id', Integer, ForeignKey('countries.id'), primary_key=True)
)

title_people = Table('title_people', Base.metadata,
    Column('show_id', String, ForeignKey('titles.show_id'), primary_key=True),
    Column('person_id', Integer, ForeignKey('people.id'), primary_key=True),
    Column('role', String) # 'Director' or 'Cast'
)

class Title(Base):
    __tablename__ = 'titles'
    
    show_id = Column(String, primary_key=True)
    type = Column(String)
    title = Column(String)
    date_added = Column(Date)
    release_year = Column(Integer)
    rating = Column(String)
    duration_minutes = Column(Integer)
    season_count = Column(Integer)
    description = Column(String)
    
    # Relationships
    genres = relationship("Genre", secondary=title_genres, back_populates="titles")
    countries = relationship("Country", secondary=title_countries, back_populates="titles")
    people = relationship("Person", secondary=title_people, back_populates="titles")

class Genre(Base):
    __tablename__ = 'genres'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    titles = relationship("Title", secondary=title_genres, back_populates="genres")

class Country(Base):
    __tablename__ = 'countries'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    titles = relationship("Title", secondary=title_countries, back_populates="titles")

class Person(Base):
    __tablename__ = 'people'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    titles = relationship("Title", secondary=title_people, back_populates="people")

# --- Loader ---

DB_PATH = "sqlite:///data/netflix_relational.db"

def load_database():
    logger.info("Initializing Database...")
    engine = create_engine(DB_PATH)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Load Data
    csv_path = r"data/processed/netflix_cleaned_v2.csv"
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # Ensure date is parsed
    df['date_added_clean'] = pd.to_datetime(df['date_added_clean'], errors='coerce')
    
    # Caches to avoid duplicate queries
    genre_cache = {}
    country_cache = {}
    person_cache = {}

    logger.info(f"Processing {len(df)} rows for SQL injection...")
    
    count = 0
    for _, row in df.iterrows():
        # Create Title
        title = Title(
            show_id=row['show_id'],
            type=row['type'],
            title=row['title'],
            date_added=row['date_added_clean'].date() if pd.notnull(row['date_added_clean']) else None,
            release_year=row['release_year'],
            rating=row['rating'],
            duration_minutes=row.get('duration_minutes', 0),
            season_count=row.get('season_count', 0),
            description=row.get('description', '')
        )
        
        # Process Genres
        if pd.notna(row['listed_in']):
            genres = [g.strip() for g in row['listed_in'].split(',')]
            for g_name in genres:
                if g_name not in genre_cache:
                    g_obj = session.query(Genre).filter_by(name=g_name).first()
                    if not g_obj:
                        g_obj = Genre(name=g_name)
                        session.add(g_obj)
                        session.flush() # Get ID
                    genre_cache[g_name] = g_obj
                title.genres.append(genre_cache[g_name])

        # Process Countries
        if pd.notna(row['country']):
            countries = [c.strip() for c in str(row['country']).split(',')]
            for c_name in countries:
                if c_name == "Unknown Country": continue
                if c_name not in country_cache:
                    c_obj = session.query(Country).filter_by(name=c_name).first()
                    if not c_obj:
                        c_obj = Country(name=c_name)
                        session.add(c_obj)
                        session.flush()
                    country_cache[c_name] = c_obj
                title.countries.append(country_cache[c_name])

        # Process Directors
        if pd.notna(row['director']) and row['director'] != "Unknown Director":
            directors = [d.strip() for d in row['director'].split(',')]
            for d_name in directors:
                if d_name not in person_cache:
                    p_obj = session.query(Person).filter_by(name=d_name).first()
                    if not p_obj:
                        p_obj = Person(name=d_name)
                        session.add(p_obj)
                        session.flush()
                    person_cache[d_name] = p_obj
                # Add relationship (Association Object or Append? Simpler to Append but missing Role attribute in M2M table object access... 
                # For this simplified loader, we just append content to person.
                # To fill specific 'role' column in association table is harder with simple append. 
                # We skip 'role' population for brevity or use direct insert if critical. 
                # Let's just link them.
                title.people.append(person_cache[d_name])

        session.add(title)
        count += 1
        if count % 100 == 0:
            session.commit()
            print(f"Processed {count} titles...", end='\r')
            
    session.commit()
    logger.success("Database population complete!")

if __name__ == "__main__":
    load_database()
