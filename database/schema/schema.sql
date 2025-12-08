-- Netflix Project Database Schema

-- Clean up existing tables
DROP TABLE IF EXISTS title_country CASCADE;
DROP TABLE IF EXISTS title_genre CASCADE;
DROP TABLE IF EXISTS title_person CASCADE;
DROP TABLE IF EXISTS people CASCADE;
DROP TABLE IF EXISTS genres CASCADE;
DROP TABLE IF EXISTS countries CASCADE;
DROP TABLE IF EXISTS titles CASCADE;

-- 1. Titles Table (Core content)
CREATE TABLE titles (
    show_id VARCHAR(10) PRIMARY KEY,
    type VARCHAR(20),
    title TEXT,
    date_added DATE,
    release_year INTEGER,
    rating VARCHAR(10),
    duration_minutes FLOAT,
    season_count INTEGER,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Dimensions Tables
CREATE TABLE people (
    person_id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE genres (
    genre_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE countries (
    country_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Junction Tables (Many-to-Many relationships)
CREATE TABLE title_person (
    show_id VARCHAR(10) REFERENCES titles(show_id),
    person_id INTEGER REFERENCES people(person_id),
    role VARCHAR(20), -- 'Director' or 'Cast'
    PRIMARY KEY (show_id, person_id, role)
);

CREATE TABLE title_genre (
    show_id VARCHAR(10) REFERENCES titles(show_id),
    genre_id INTEGER REFERENCES genres(genre_id),
    PRIMARY KEY (show_id, genre_id)
);

CREATE TABLE title_country (
    show_id VARCHAR(10) REFERENCES titles(show_id),
    country_id INTEGER REFERENCES countries(country_id),
    PRIMARY KEY (show_id, country_id)
);

-- 4. Indexes for performance
CREATE INDEX idx_titles_type ON titles(type);
CREATE INDEX idx_titles_release_year ON titles(release_year);
CREATE INDEX idx_titles_date_added ON titles(date_added);
CREATE INDEX idx_people_name ON people(name);
