
import panel as pn
import pandas as pd
import hvplot.pandas
import holoviews as hv

# --- INITIALIZATION ---
pn.extension('plotly', 'tabulator', design='material')
hv.renderer('bokeh').theme = 'dark_minimal'

# --- CSS STYLING (The Fix) ---
# We use a CSS string that forces high contrast
# 1. Body and Root text -> White
# 2. Sidebar labels -> Bright White + Bold
# 3. Chart Filters -> White
high_contrast_css = """
:root {
    --design-primary-color: #E50914;
    --background-color: #141414;
    --panel-surface-color: #1f1f1f;
}

/* Force global text color */
body {
    color: #ffffff !important;
    font-family: 'Helvetica Neue', Arial, sans-serif;
}

/* Sidebar Labels - The main culprit of invisibility */
.bk-root label, .bk-root .bk-slider-title {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1.05em !important;
    text-shadow: 0px 1px 2px black;
}

/* Input boxes text */
.bk-input {
    color: white !important;
    background-color: #333 !important;
    border: 1px solid #555 !important;
}

/* KPI Cards */
.kpi-card-box {
    background: #1f1f1f;
    border-radius: 8px;
    padding: 20px;
    border-left: 5px solid #E50914;
    box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    color: white;
}

/* Tabulator/Tables */
.tabulator {
    background-color: #1f1f1f !important;
    color: white !important;
}
"""

pn.config.raw_css.append(high_contrast_css)

# --- DATA ---
@pn.cache(ttl=600)
def load_data():
    try:
        df = pd.read_csv('d:/Unified_internship/netflix_project/data/processed/netflix_cleaned.csv')
        df['year_added'] = pd.to_datetime(df['date_added']).dt.year
        return df.dropna(subset=['year_added'])
    except:
        return pd.DataFrame({'year_added': [2020], 'type': ['Movie'], 'listed_in': ['Action'], 'show_id':['s1']})

df = load_data()

# --- WIDGETS ---
year_slider = pn.widgets.IntRangeSlider(
    name='Release Year Range', 
    start=2000, end=2021, value=(2015, 2021),
    sizing_mode='stretch_width'
)

type_selector = pn.widgets.MultiChoice(
    name='Content Type', 
    options=['Movie', 'TV Show'], 
    value=['Movie', 'TV Show'],
    sizing_mode='stretch_width'
)

# --- PLOTTING ---
def plot_trend(years, types):
    if df.empty: return pn.pane.Markdown("No Data")
    mask = (df['year_added'].between(years[0], years[1])) & (df['type'].isin(types))
    data = df[mask].groupby('year_added')['show_id'].count().reset_index()
    
    return data.hvplot.area(
        x='year_added', y='show_id', title='Growth of Content',
        color='#E50914', alpha=0.7, height=320, responsive=True
    ).opts(
        bgcolor='#1f1f1f', 
        fontscale=1.2, 
        gridstyle={'grid_line_color': '#444'}
    )

def plot_genres(years, types):
    if df.empty: return pn.pane.Markdown("No Data")
    mask = (df['year_added'].between(years[0], years[1])) & (df['type'].isin(types))
    s = df[mask]['listed_in'].str.split(',').explode().str.strip()
    return s.value_counts().head(10).sort_values().hvplot.barh(
        title='Top Genres', color='#E50914', height=320, responsive=True
    ).opts(
        bgcolor='#1f1f1f', fontscale=1.2, 
        gridstyle={'grid_line_color': '#444'},
        invert_xaxis=False
    )

# --- KPIS ---
def get_kpis(years, types):
    mask = (df['year_added'].between(years[0], years[1])) & (df['type'].isin(types))
    curr = df[mask]
    
    return pn.Row(
        pn.pane.HTML(f"""<div class='kpi-card-box'><h3>Total Titles</h3><h1>{len(curr):,}</h1></div>""", sizing_mode='stretch_width'),
        pn.pane.HTML(f"""<div class='kpi-card-box'><h3>Movies</h3><h1>{len(curr[curr['type']=='Movie']):,}</h1></div>""", sizing_mode='stretch_width'),
        pn.pane.HTML(f"""<div class='kpi-card-box'><h3>TV Shows</h3><h1>{len(curr[curr['type']=='TV Show']):,}</h1></div>""", sizing_mode='stretch_width'),
        sizing_mode='stretch_width'
    )

# Binds
d_trend = pn.bind(plot_trend, year_slider, type_selector)
d_genres = pn.bind(plot_genres, year_slider, type_selector)
d_kpis = pn.bind(get_kpis, year_slider, type_selector)

# --- LAYOUT ---
# We use FastListTemplate because we verified it works in V2
template = pn.template.FastListTemplate(
    title='NETFLIX ANALYTICS ELITE',
    sidebar=[
        pn.pane.Markdown("## Filters", style={'color': 'white'}),
        year_slider,
        type_selector,
        pn.pane.Markdown("---"),
        pn.pane.Markdown("Use these filters to explore the dataset.", style={'color': '#aaa'})
    ],
    main=[
        pn.Row(d_kpis),
        pn.Row(d_trend, d_genres)
    ],
    accent_base_color="#E50914",
    header_background="#000000",
    background_color="#141414",
    theme="dark",
    shadow=False
)

template.servable()
