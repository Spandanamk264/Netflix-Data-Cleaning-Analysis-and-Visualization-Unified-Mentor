
import panel as pn
import pandas as pd
import hvplot.pandas
import holoviews as hv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- SETUP & STYLE ---
pn.extension('plotly', design='material', css_files=['styles.css'])
hv.renderer('bokeh').theme = 'dark_minimal'

# Load External CSS manually for assurance (sometimes css_files arg needs relative path handling)
with open('d:/Unified_internship/netflix_project/src/dashboard/styles.css', 'r') as f:
    pn.config.raw_css.append(f.read())

# --- DATA LOADING (Cached) ---
@pn.cache(ttl=600)
def load_data():
    try:
        path = r'd:/Unified_internship/netflix_project/data/processed/netflix_cleaned.csv'
        df = pd.read_csv(path)
        df['date_added'] = pd.to_datetime(df['date_added'])
        df['year_added'] = df['date_added'].dt.year
        
        # Robust feature creation for ML
        df['combined_features'] = df['title'].astype(str) + " " + \
                                  df['listed_in'].fillna('') + " " + \
                                  df['description'].fillna('')
        return df
    except Exception as e:
        return pd.DataFrame()

df = load_data()

# --- COMPONENTS ---

def create_kpi_card(title, value, icon):
    return pn.pane.HTML(f"""
        <div class="kpi-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="kpi-title">{title}</div>
                <div style="font-size: 1.5rem;">{icon}</div>
            </div>
            <div class="kpi-value">{value}</div>
        </div>
    """, sizing_mode='stretch_width')

# Widgets
min_year = int(df['release_year'].min()) if not df.empty else 2000
max_year = int(df['release_year'].max()) if not df.empty else 2023

year_slider = pn.widgets.IntRangeSlider(name='Release Year', start=min_year, end=max_year, value=(2010, max_year))
type_selector = pn.widgets.MultiChoice(name='Type', options=list(df['type'].unique()) if not df.empty else [], value=list(df['type'].unique()) if not df.empty else [])

# Plotting Functions
def plot_timeline(years, types):
    if df.empty: return pn.pane.Markdown("No Data")
    filtered = df[(df['release_year'].between(years[0], years[1])) & (df['type'].isin(types))]
    counts = filtered.groupby('year_added')['show_id'].count().reset_index()
    
    return counts.hvplot.area(
        x='year_added', y='show_id', 
        title='Content Velocity', 
        color='#E50914', alpha=0.6,
        height=300, responsive=True,
        grid=True, line_width=2
    ).opts(bgcolor='#1f1f1f', fontscale=1.2, show_grid=False)

def plot_top_genres(years, types):
    if df.empty: return pn.pane.Markdown("No Data")
    filtered = df[(df['release_year'].between(years[0], years[1])) & (df['type'].isin(types))]
    s = filtered['listed_in'].str.split(',').explode().str.strip()
    return s.value_counts().head(8).sort_values().hvplot.barh(
        title='Top Genres', color='#E50914',
        height=300, responsive=True
    ).opts(bgcolor='#1f1f1f', fontscale=1.2, show_grid=True, invert_xaxis=False)

# Bindings
timeline = pn.bind(plot_timeline, year_slider, type_selector)
genres_plot = pn.bind(plot_top_genres, year_slider, type_selector)

# KPIS Dynamic
def update_kpis(years, types):
    if df.empty: return pn.Row()
    filtered = df[(df['release_year'].between(years[0], years[1])) & (df['type'].isin(types))]
    return pn.Row(
        create_kpi_card("Total Titles", f"{len(filtered):,}", "🎬"),
        create_kpi_card("Movies", f"{len(filtered[filtered['type']=='Movie']):,}", "🎥"),
        create_kpi_card("TV Shows", f"{len(filtered[filtered['type']=='TV Show']):,}", "📺"),
        sizing_mode='stretch_width'
    )

kpis_row = pn.bind(update_kpis, year_slider, type_selector)

# --- LAYOUT ---
# Header
header = pn.pane.HTML("""
<div style="background: black; padding: 20px; border-bottom: 2px solid #E50914; display: flex; align-items: center;">
    <h1 style="color: #E50914; margin: 0; font-weight: 800; letter-spacing: -1px;">NETFLIX <span style="color:white; font-weight:300;">ANALYTICS</span></h1>
    <div style="flex-grow:1;"></div>
    <div style="color: #888; font-size: 0.9rem;">ENTERPRISE EDITION</div>
</div>
""", sizing_mode='stretch_width')

# Main Template with Vanilla CSS Injection
template = pn.template.FastListTemplate(
    title='',
    sidebar=[
        pn.pane.Markdown("### 🔍 Filters", style={'color':'white'}),
        year_slider,
        type_selector,
        pn.pane.Markdown("---"),
        pn.pane.Markdown("### ℹ️ About", style={'color':'white'}),
        pn.pane.Markdown("This dashboard showcases advanced data engineering capabilities using Panel and HoloViews.", style={'color':'#aaa', 'font-size':'0.85rem'})
    ],
    main=[
        header,
        pn.Column(
            kpis_row,
            pn.Row(
                pn.Column(timeline, css_classes=['chart-box']),
                pn.Column(genres_plot, css_classes=['chart-box'])
            ),
            sizing_mode='stretch_width'
        )
    ],
    accent_base_color="#E50914",
    header_background="#000000",
    theme="dark",
    theme_toggle=False
)

template.servable()
