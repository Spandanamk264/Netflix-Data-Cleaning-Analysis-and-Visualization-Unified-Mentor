
import panel as pn
import pandas as pd
import hvplot.pandas

# --- CONFIG ---
pn.extension('plotly', 'tabulator')

# --- DATA ---
@pn.cache
def load_data():
    try:
        df = pd.read_csv('d:/Unified_internship/netflix_project/data/processed/netflix_cleaned_v2.csv')
        # V2 CSV has 'year_added' already. Just ensure it's numeric and drop NaNs for the dashboard
        return df.dropna(subset=['year_added'])
    except:
        return pd.DataFrame({'year_added':[2021], 'title':['Demo'], 'type':['Movie'], 'listed_in':['Action'], 'show_id':['s1']})

df = load_data()

# --- STYLES ---
CARD_STYLE = {
    'background': 'white', 
    'border-radius': '8px', 
    'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 
    'padding': '15px',
    'border-top': '4px solid #E50914'
}

KPI_STYLE = {
    'background': 'white',
    'border-radius': '8px',
    'padding': '20px',
    'border-left': '5px solid #E50914',
    'box-shadow': '0 4px 6px rgba(0,0,0,0.1)'
}

# --- CHARTS ---
def plot_trend(years):
    data = df[df['year_added'].between(years[0], years[1])].groupby('year_added')['show_id'].count().reset_index()
    return data.hvplot.area(
        x='year_added', y='show_id', title='Content Growth',
        color='#E50914', height=350, responsive=True
    ).opts(fontsize={'title': '12pt'}, show_grid=True)

def plot_genres(years):
    data = df[df['year_added'].between(years[0], years[1])]
    s = data['listed_in'].str.split(',').explode().str.strip().value_counts().head(10).sort_values()
    return s.hvplot.barh(
        title='Top Genres', color='#221f1f', height=350, responsive=True
    ).opts(fontsize={'title': '12pt'}, show_grid=True)

# --- KPIS ---
def plot_kpis(years):
    data = df[df['year_added'].between(years[0], years[1])]
    total = len(data)
    movies = len(data[data['type']=='Movie'])
    tv = len(data[data['type']=='TV Show'])
    
    return pn.Row(
        pn.pane.HTML(f"<div style='font-size:14px; color:#777'>TOTAL TITLES</div><div style='font-size:32px; font-weight:bold'>{total:,}</div>", styles=KPI_STYLE, sizing_mode='stretch_width'),
        pn.pane.HTML(f"<div style='font-size:14px; color:#777'>MOVIES</div><div style='font-size:32px; font-weight:bold'>{movies:,}</div>", styles=KPI_STYLE, sizing_mode='stretch_width'),
        pn.pane.HTML(f"<div style='font-size:14px; color:#777'>TV SHOWS</div><div style='font-size:32px; font-weight:bold'>{tv:,}</div>", styles=KPI_STYLE, sizing_mode='stretch_width'),
        sizing_mode='stretch_width'
    )

# --- WIDGETS ---
year_slider = pn.widgets.IntRangeSlider(name='Year Range', start=2000, end=2021, value=(2010, 2021))

# --- LAYOUT ---
# Bindings
d_trend = pn.bind(plot_trend, year_slider)
d_genres = pn.bind(plot_genres, year_slider)
d_kpis = pn.bind(plot_kpis, year_slider)

dashboard = pn.template.FastListTemplate(
    title='NETFLIX ANALYTICS SUITE',
    sidebar=[pn.pane.Markdown("## Filters"), year_slider],
    main=[
        pn.Column(
            pn.pane.Markdown("### Overview"),
            d_kpis,
            pn.Row(
                pn.Card(d_trend, title="Growth Strategy", styles=CARD_STYLE),
                pn.Card(d_genres, title="Market Distribution", styles=CARD_STYLE)
            )
        )
    ],
    accent_base_color="#E50914",
    header_background="#E50914",
    background_color="#f5f5f5",
    theme="default"
)

dashboard.servable()
