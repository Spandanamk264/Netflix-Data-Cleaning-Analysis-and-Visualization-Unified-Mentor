
import panel as pn
import pandas as pd
import hvplot.pandas

# --- CONFIG ---
# Standard extension without 'design' to avoid conflicts with FastListTemplate
pn.extension('plotly', 'tabulator')

# --- DATA ---
@pn.cache
def load_data():
    try:
        df = pd.read_csv('d:/Unified_internship/netflix_project/data/processed/netflix_cleaned.csv')
        df['year_added'] = pd.to_datetime(df['date_added']).dt.year
        return df
    except:
        return pd.DataFrame({'year_added':[2021], 'title':['Demo'], 'type':['Movie'], 'listed_in':['Action']})

df = load_data()

# --- STYLES ---
# We use a white theme with Red accents (Netflix Brand)
# This guarantees text is black on white (visible!)
CARD_STYLE = {
    'background': 'white', 
    'border-radius': '8px', 
    'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 
    'padding': '15px',
    'border-top': '4px solid #E50914'
}

# --- CHARTS ---
def plot_trend(years):
    data = df[df['year_added'].between(years[0], years[1])].groupby('year_added')['show_id'].count().reset_index()
    return data.hvplot.area(
        x='year_added', y='show_id', title='Content Growth',
        color='#E50914', height=300, responsive=True
    ).opts(fontsize={'title': '12pt'}, show_grid=True)

def plot_genres(years):
    data = df[df['year_added'].between(years[0], years[1])]
    s = data['listed_in'].str.split(',').explode().str.strip().value_counts().head(10).sort_values()
    return s.hvplot.barh(
        title='Top Genres', color='#221f1f', height=300, responsive=True
    ).opts(fontsize={'title': '12pt'}, show_grid=True)

# --- WIDGETS ---
year_slider = pn.widgets.IntRangeSlider(name='Year Range', start=2000, end=2021, value=(2010, 2021))

# --- LAYOUT ---
dashboard = pn.template.FastListTemplate(
    title='NETFLIX ANALYTICS (High Contrast)',
    sidebar=[pn.pane.Markdown("## Filters"), year_slider],
    main=[
        pn.Row(
            pn.Card(pn.bind(plot_trend, year_slider), title="Content Trend", styles=CARD_STYLE),
            pn.Card(pn.bind(plot_genres, year_slider), title="Genre Distribution", styles=CARD_STYLE)
        )
    ],
    accent_base_color="#E50914",
    header_background="#E50914",
    background_color="#f5f5f5",
    theme="default" # Forces Light Mode
)

dashboard.servable()
