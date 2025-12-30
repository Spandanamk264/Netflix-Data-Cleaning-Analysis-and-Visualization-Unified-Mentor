
import panel as pn
import pandas as pd
import hvplot.pandas
import holoviews as hv

# --- INITIALIZATION ---
pn.extension('plotly', design='material')
hv.renderer('bokeh').theme = 'dark_minimal'

# --- CSS STYLING ---
# We force these styles to ensure visibility regardless of theme defaults
custom_css = """
:root {
    --design-primary-color: #E50914;
    --background-color: #121212;
    --surface-color: #1e1e1e;
    --text-color: #ffffff;
}
body {
    background-color: var(--background-color);
    color: var(--text-color);
    font-family: 'Roboto', sans-serif;
}
.bk-root {
    background-color: transparent !important;
}
/* Force text white in charts and widgets */
.bk-root .bk {
    color: white !important;
}
/* KPIs */
.kpi-card {
    background-color: var(--surface-color);
    border-left: 5px solid var(--design-primary-color);
    border-radius: 8px;
    padding: 15px;
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.kpi-value {
    font-size: 2em;
    font-weight: bold;
    margin: 5px 0;
}
.kpi-label {
    text-transform: uppercase;
    font-size: 0.85em;
    color: #b3b3b3;
}
/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 10px;
    background: #121212;
}
::-webkit-scrollbar-thumb {
    background: #333; 
    border-radius: 5px;
}
"""

pn.config.raw_css.append(custom_css)

# --- DATA MOCK ---
# robust load
try:
    df = pd.read_csv('d:/Unified_internship/netflix_project/data/processed/netflix_cleaned.csv')
    df['year_added'] = pd.to_datetime(df['date_added']).dt.year
    df = df.dropna(subset=['year_added', 'listed_in', 'type'])
except:
    # Fallback data
    df = pd.DataFrame({
        'title': ['Mock Movie A', 'Mock Movie B'], 
        'type': ['Movie', 'Movie'],
        'release_year': [2020, 2021],
        'listed_in': ['Drama', 'Action'],
        'year_added': [2020, 2021],
        'show_id': ['s1', 's2']
    })

# --- WIDGETS ---
year_slider = pn.widgets.EditableRangeSlider(
    name='Year-Range', 
    start=int(df['release_year'].min()), 
    end=int(df['release_year'].max()), 
    value=(2015, 2021),
    step=1
)
type_select = pn.widgets.MultiChoice(
    name='Content Type', 
    options=list(df['type'].unique()), 
    value=list(df['type'].unique())
)

# --- PLOTS & KPIS ---
def get_dashboard_components(year_range, types):
    mask = (df['release_year'].between(year_range[0], year_range[1])) & (df['type'].isin(types))
    curr_df = df[mask]
    
    # 1. KPIs
    total = len(curr_df)
    movies = len(curr_df[curr_df['type']=='Movie'])
    tv = len(curr_df[curr_df['type']=='TV Show'])
    
    kpis = pn.Row(
        pn.pane.HTML(f"""<div class="kpi-card"><div class="kpi-label">Selected Titles</div><div class="kpi-value">{total}</div></div>""", sizing_mode='stretch_width'),
        pn.pane.HTML(f"""<div class="kpi-card"><div class="kpi-label">Movies</div><div class="kpi-value">{movies}</div></div>""", sizing_mode='stretch_width'),
        pn.pane.HTML(f"""<div class="kpi-card"><div class="kpi-label">TV Shows</div><div class="kpi-value">{tv}</div></div>""", sizing_mode='stretch_width'),
        sizing_mode='stretch_width'
    )
    
    # 2. Charts
    # Trend
    trend = curr_df.groupby('year_added')['show_id'].count().reset_index()
    fig_trend = trend.hvplot.area(
        x='year_added', y='show_id', title='Content Addition Trend',
        color='#E50914', alpha=0.6, height=350, responsive=True
    ).opts(bgcolor='#1e1e1e', show_grid=True, fontscale=1.1, text_color='white')
    
    # Genres
    genres = curr_df['listed_in'].str.split(',').explode().str.strip().value_counts().head(10).sort_values()
    fig_genres = genres.hvplot.barh(
        title='Top Genres', color='#E50914', height=350, responsive=True
    ).opts(bgcolor='#1e1e1e', show_grid=True, fontscale=1.1, text_color='white')
    
    return pn.Column(
        kpis,
        pn.Row(fig_trend, fig_genres, sizing_mode='stretch_width'),
        sizing_mode='stretch_width'
    )

# Bind
dashboard_view = pn.bind(get_dashboard_components, year_slider, type_select)

# --- TEMPLATE ---
template = pn.template.MaterialTemplate(
    title='NETFLIX ANALYTICS PRO',
    sidebar=[
        pn.pane.Markdown("### Controls"),
        year_slider,
        type_select,
        pn.pane.Markdown("---"),
        pn.pane.Markdown("Use the filters above to slice the dataset.")
    ],
    main=[dashboard_view],
    theme='dark',
    header_background='#000000'
)

template.servable()
