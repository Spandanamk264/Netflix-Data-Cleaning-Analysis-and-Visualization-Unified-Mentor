
import panel as pn
import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Debug print
print("Initializing Panel App...")

# Simplified extension
try:
    # Improved CSS for high contrast
    css = """
    :root { --design-primary-color: #b20710; }
    body { color: white !important; }
    .bk-root { color: white !important; }
    .bk-input-group > label, .bk-slider-title, .bk-root label { 
        color: #ffffff !important; 
        font-weight: bold;
    }
    """
    pn.extension('plotly', design='material', global_css=[css])
    hv.renderer('bokeh').theme = 'dark_minimal'
except:
    pn.extension('plotly')

# --- CONFIGURATION ---
PALETTE = ["#e50914", "#b20710", "#221f1f", "#f5f5f1"]
ACCENT_COLOR = "#b20710"

# --- DATA LOADER ---
@pn.cache
def load_data():
    print("Loading data...")
    try:
        # Use absolute path
        path = r'd:/Unified_internship/netflix_project/data/processed/netflix_cleaned.csv'
        df = pd.read_csv(path)
        df['date_added'] = pd.to_datetime(df['date_added'])
        df['year_added'] = df['date_added'].dt.year
        
        # Robust feature creation
        df['combined_features'] = df['title'].astype(str) + " " + \
                                  df['director'].fillna('') + " " + \
                                  df['cast'].fillna('') + " " + \
                                  df['listed_in'].fillna('') + " " + \
                                  df['description'].fillna('')
        print(f"Data Loaded: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Data Load Error: {e}")
        # Create dummy data so app doesn't crash
        return pd.DataFrame({
            'title': ['Example Movie'], 
            'type': ['Movie'], 
            'release_year': [2020], 
            'listed_in': ['Drama'],
            'description': ['Test'],
            'year_added': [2020],
            'rating': ['TV-MA'],
            'combined_features': ['Example Movie']
        })

df = load_data()

# --- RECS ENGINE ---
def get_recommendations(title):
    # Simple placeholder if no data or title not found
    if df.empty or title not in df['title'].values:
        return pd.DataFrame()
    
    # Just return random sample for demo if ML fails (to keep UI alive)
    try:
        # In a real app we'd cache the matrix. For now, compute on fly or skip.
        # Computing TFIDF on every click is slow, but for <10k rows acceptable for demo.
        # Optimizing: limit to top 1000 for speed
        subset = df.head(5000) 
        if title not in subset['title'].values:
            return pd.DataFrame()
            
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(subset['combined_features'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) # Heavy operation!
        
        indices = pd.Series(subset.index, index=subset['title']).drop_duplicates()
        idx = indices[title]
        
        # If duplicated titles, take first
        if isinstance(idx, pd.Series): idx = idx.iloc[0]
            
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:7]
        movie_indices = [i[0] for i in sim_scores]
        return subset.iloc[movie_indices][['title', 'type', 'listed_in', 'release_year', 'description']]
    except Exception as e:
        print(f"Rec Error: {e}")
        return pd.DataFrame()

# --- WIDGETS ---
print("Creating Widgets...")
min_year = int(df['release_year'].min()) if not df.empty else 2000
max_year = int(df['release_year'].max()) if not df.empty else 2023

year_slider = pn.widgets.IntRangeSlider(
    name='Release Year', 
    start=min_year, end=max_year, value=(2010, max_year)
)

types = list(df['type'].unique()) if not df.empty else ['Movie']
type_selector = pn.widgets.MultiChoice(
    name='Content Type', options=types, value=types
)

# Fix: Flatten genres properly
try:
    if not df.empty and 'listed_in' in df.columns:
        genre_list = df['listed_in'].dropna().astype(str).str.split(',').explode().str.strip().unique().tolist()
        genres = sorted(genre_list)
    else:
        genres = []
except:
    genres = []

genre_selector = pn.widgets.MultiChoice(
    name='Genres', options=genres, value=[], placeholder='Filter by genres...'
)

# --- BINDING ---
def filter_df(years, types, genres):
    if df.empty: return df
    mask = (df['release_year'] >= years[0]) & (df['release_year'] <= years[1])
    mask &= (df['type'].isin(types))
    if genres:
        mask &= df['listed_in'].apply(lambda x: any(g in str(x) for g in genres))
    return df[mask]

# We use pn.bind instead of hvplot.bind sometimes for clearer flow, but hvplot.bind is fine
idf = hvplot.bind(filter_df, year_slider, type_selector, genre_selector)

# --- CHARTS ---
def plot_trend(data):
    if data is None or data.empty: return pn.pane.Markdown("No Data")
    counts = data.groupby('year_added')['title'].count().reset_index()
    counts = data.groupby('year_added')['title'].count().reset_index()
    return counts.hvplot.area(x='year_added', y='title', color=ACCENT_COLOR, responsive=True, height=250).opts(bgcolor='rgba(0,0,0,0)', show_grid=False, fontscale=1.1)

def plot_genres(data):
    if data is None or data.empty: return pn.pane.Markdown("No Data")
    # Quick count
    s = data['listed_in'].str.split(',').explode().str.strip()
    # Quick count
    s = data['listed_in'].str.split(',').explode().str.strip()
    return s.value_counts().head(10).sort_values().hvplot.barh(responsive=True, height=250, color=ACCENT_COLOR).opts(bgcolor='rgba(0,0,0,0)', fontscale=1.1, show_grid=True, xlim=(0, None))

# --- TEMPLATE ---
print("Building Template...")
template = pn.template.FastListTemplate(
    title='Netflix Analytics Suite',
    sidebar=[
        pn.pane.Markdown("## Filters"),
        year_slider,
        type_selector,
        genre_selector
    ],
    main=[
        pn.Row(
            pn.Card(hvplot.bind(plot_trend, idf), title="Content Trend"),
            pn.Card(hvplot.bind(plot_genres, idf), title="Top Genres")
        ),
        pn.pane.Markdown("### AI Recommender (Beta)"),
        pn.pane.Markdown("Use the search bar on the sidebar or main content to find titles (Not fully implemented in this demo view).") # Simplified
    ],
    accent_base_color=ACCENT_COLOR,
    header_background=ACCENT_COLOR,
    theme="dark"
)

template.servable()
print("App Ready.")
