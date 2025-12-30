
import panel as pn
import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Initialize Panel extension specifically with relevant extensions
pn.extension('tabulator', 'echarts', 'plotly', design='material')

# --- CONFIGURATION ---
PALETTE = ["#e50914", "#b20710", "#221f1f", "#f5f5f1"]  # Netflix-ish Red, Dark Red, Black, White
ACCENT_COLOR = "#b20710"

# --- DATA LOADER ---
@pn.cache(ttl=600)  # Cache for 10 minutes
def load_data():
    try:
        # Load processed data
        df = pd.read_csv('d:/Unified_internship/netflix_project/data/processed/netflix_cleaned.csv')
        df['date_added'] = pd.to_datetime(df['date_added'])
        df['year_added'] = df['date_added'].dt.year
        
        # Ensure we have a string column for text processing
        df['combined_features'] = df['title'] + " " + df['director'].fillna('') + " " + df['cast'].fillna('') + " " + df['listed_in'].fillna('') + " " + df['description'].fillna('')
        return df
    except Exception as e:
        # Fallback to creating a dummy df if file missing (for robustness during dev)
        return pd.DataFrame()

df = load_data()

# --- PRECOMPUTE SIMILARITY MATRIX (Lazy Load) ---
tfidf_matrix = None
indices = None

def get_recommendations(title, cosine_sim=None):
    global tfidf_matrix, indices
    if tfidf_matrix is None:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['combined_features'].fillna(''))
        
        # Compute cosine similarity (subset for speed if needed, here full)
        # For performance on large datasets, consider approximate NN, but for <10k rows linear is fine
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    if cosine_sim is None: 
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return df.iloc[movie_indices][['title', 'type', 'listed_in', 'release_year', 'description']]
    except KeyError:
        return pd.DataFrame()

# --- WIDGETS ---
# Sidebar controls
min_year = int(df['release_year'].min()) if not df.empty else 2000
max_year = int(df['release_year'].max()) if not df.empty else 2023

year_slider = pn.widgets.IntRangeSlider(
    name='Release Year', 
    start=min_year, end=max_year, value=(2010, max_year), 
    step=1
)

types = list(df['type'].unique()) if not df.empty else ['Movie', 'TV Show']
type_selector = pn.widgets.MultiChoice(
    name='Content Type', options=types, value=types
)

genres = sorted(list(set([x.strip() for y in df['listed_in'].str.split(',') for x in y]))) if not df.empty else []
genre_selector = pn.widgets.MultiChoice(
    name='Genres', options=genres, value=[], placeholder='Filter by genres...'
)

# --- PIPELINE ---
def filter_df(df, years, types, genres):
    if df.empty: return df
    mask = (df['release_year'] >= years[0]) & (df['release_year'] <= years[1])
    mask &= (df['type'].isin(types))
    if genres:
        # Check if any selected genre is present
        genre_mask = df['listed_in'].apply(lambda x: any(g in x for g in genres))
        mask &= genre_mask
    return df[mask]

# Create interactive dataframe
idf = hvplot.bind(filter_df, df, year_slider, type_selector, genre_selector)

# --- PLOTS ---
def plot_trend(data):
    if data.empty: return "No Data"
    counts = data.groupby('year_added')['show_id'].count().reset_index()
    return counts.hvplot.area(
        x='year_added', y='show_id', 
        title='Content Added Over Time', 
        color=ACCENT_COLOR, alpha=0.6,
        responsive=True, height=300
    )

def plot_top_genres(data):
    if data.empty: return "No Data"
    # Split genres for accurate counting
    s = data['listed_in'].str.split(',').explode().str.strip()
    top_n = s.value_counts().head(10).sort_values()
    return top_n.hvplot.barh(
        title='Top 10 Genres', 
        color=ACCENT_COLOR,
        responsive=True, height=300,
        grid=True
    )

def plot_rating_dist(data):
    if data.empty: return "No Data"
    return data['rating'].value_counts().hvplot.bar(
        title='Rating Distribution',
        rot=45, color='#221f1f',
        responsive=True, height=300
    )

# --- KPI CARDS ---
def kpi_cards(data):
    if data.empty:
        return pn.Row(pn.indicators.Number(name="Total", value=0))
    
    total = len(data)
    movies = len(data[data['type'] == 'Movie'])
    tv = len(data[data['type'] == 'TV Show'])
    
    # Custom HTML Card Style
    def card(title, value, icon):
        return pn.pane.HTML(f"""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 8px; border-left: 5px solid {ACCENT_COLOR}; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="font-size: 14px; color: #aaa; text-transform: uppercase;">{title}</div>
            <div style="font-size: 32px; font-weight: bold; margin-top: 5px;">{value:,}</div>
            <div style="font-size: 24px; position: absolute; right: 20px; top: 20px; opacity: 0.2;">{icon}</div>
        </div>
        """, sizing_mode="stretch_width")

    return pn.Row(
        card("Total Titles", total, "🎞️"),
        card("Movies", movies, "🎥"),
        card("TV Shows", tv, "📺"),
        sizing_mode="stretch_width"
    )

# --- RECOMMENDER TAB ---
title_input = pn.widgets.AutocompleteInput(
    name='Select a Title', 
    options=sorted(df['title'].unique().tolist()) if not df.empty else [],
    placeholder='Start typing...',
    case_sensitive=False,
    min_characters=1
)
rec_btn = pn.widgets.Button(name='Get Recommendations', button_type='primary', icon='robot')
rec_output = pn.Column()

def update_recs(event):
    if not title_input.value:
        rec_output.objects = [pn.pane.Alert("Please select a title first!", alert_type="warning")]
        return
    
    rec_output.objects = [pn.indicators.LoadingSpinner(value=True, width=50, height=50)]
    try:
        recs = get_recommendations(title_input.value)
        if recs.empty:
            rec_output.objects = [pn.pane.Alert("No recommendations found.", alert_type="warning")]
            return
        
        # Format as nice cards
        cards = []
        for i, row in recs.iterrows():
            card = pn.pane.HTML(f"""
            <div style="background-color: #2b2b2b; padding: 15px; margin-bottom: 10px; border-radius: 5px; border-left: 3px solid #e50914;">
                <div style="font-weight: bold; font-size: 1.1em; color: white;">{row['title']}</div>
                <div style="color: #bbb; font-size: 0.9em;">{row['type']} • {row['release_year']}</div>
                <div style="color: #999; font-size: 0.8em; margin-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{row['description']}</div>
            </div>
            """, sizing_mode="stretch_width")
            cards.append(card)
        
        rec_output.objects = cards
    except Exception as e:
        rec_output.objects = [pn.pane.Alert(f"Error: {e}", alert_type="danger")]

rec_btn.on_click(update_recs)

# --- LAYOUT ---
# Bind plots to interactive dataframe
d_trend = hvplot.bind(plot_trend, idf)
d_genres = hvplot.bind(plot_top_genres, idf)
d_rating = hvplot.bind(plot_rating_dist, idf)
d_kpi = hvplot.bind(kpi_cards, idf)

# Dashboard Structure using FastListTemplate (Professional Dark Theme)
template = pn.template.FastListTemplate(
    title='Netflix Analytics Suite',
    sidebar=[
        pn.pane.Markdown("## 🎛️ Filters"),
        year_slider,
        type_selector,
        genre_selector,
        pn.pane.Markdown("---"),
        pn.pane.Markdown("### 🤖 ML Model"),
        pn.pane.Markdown("Hybrid filtering enabled.", style={'color': '#888', 'font-size': '0.8em'})
    ],
    main=[
        pn.Row(d_kpi),
        pn.Row(
            pn.Column(d_trend, title="Growth Strategy"),
            pn.Column(d_genres, title="Content Distribution")
        ),
        pn.Row(
            pn.Column(d_rating, title="Demographics"),
            pn.Column(
                pn.pane.Markdown("### 🧠 AI Recommender"),
                title_input,
                rec_btn,
                rec_output,
                sizing_mode="stretch_width"
            )
        )
    ],
    accent_base_color=ACCENT_COLOR,
    header_background=ACCENT_COLOR,
    background_color="#141414",
    theme="dark",
    theme_toggle=False
)

template.servable()
