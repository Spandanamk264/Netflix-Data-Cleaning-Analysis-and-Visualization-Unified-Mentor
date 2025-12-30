"""
Netflix Advanced Analytics Dashboard
A premium web application for exploring content, generating recommendations,
and running predictive models.
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- CONFIGURATION ---
ST_PAGE_TITLE = "Netflix Intelligence"
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title=ST_PAGE_TITLE,
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (NETFLIX THEME) ---
st.markdown("""
    <style>
    /* Light Theme Main Configuration */
    .main {
        background-color: #ffffff;
        color: #333333;
    }
    .stApp {
        background-color: #f5f5f5;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #E50914 !important; /* Netflix Red */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
    }
    p, div, label, span {
        color: #333333;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #E50914;
        color: white;
        border-radius: 4px;
        border: none;
        height: 3em;
        width: 100%;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #b20710; /* Darker Red */
        color: white;
        border: 1px solid #b20710;
        transform: scale(1.02);
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #E50914;
    }
    
    /* Dataframe and Tables */
    .stDataFrame {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data_v2():
    """Load the cleaned dataset directly for visualization speed"""
    try:
        df = pd.read_csv('data/processed/netflix_cleaned.csv')
        df['date_added'] = pd.to_datetime(df['date_added'])
        
        # DEBUG: Check if listed_in exists
        if 'listed_in' not in df.columns:
            st.error("CRITICAL: 'listed_in' column missing from dataset!")
            df['primary_genre'] = 'Unknown'
            return df

        # Force creation of primary_genre if missing
        if 'primary_genre' not in df.columns:
            # Create from listed_in (take first item)
            df['primary_genre'] = df['listed_in'].fillna("Unknown").astype(str).apply(lambda x: x.split(',')[0])
            
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

df = load_data_v2()

# --- SIDEBAR ---
st.sidebar.title("🍿 Menu")
page = st.sidebar.radio("Navigate", ["Dashboard Overview", "content Explorer", "AI Recommender", "Content Classifier"])

st.sidebar.markdown("---")
st.sidebar.info(f"✅ Database: {len(df):,} titles")
st.sidebar.info("✅ Status: Online")

# --- PAGES ---

if page == "Dashboard Overview":
    st.title("📊 Executive Dashboard")
    
    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Titles", f"{len(df):,}")
    with col2:
        st.metric("Movies", f"{len(df[df['type']=='Movie']):,}")
    with col3:
        st.metric("TV Shows", f"{len(df[df['type']=='TV Show']):,}")
    with col4:
        st.metric("Unique Genres", f"{df['primary_genre'].nunique()}")
        
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Content Added Over Time")
        daily_counts = df.groupby(df['date_added'].dt.year)['show_id'].count().reset_index()
        fig_trend = px.area(daily_counts, x='date_added', y='show_id', title="Content Velocity", color_discrete_sequence=['#E50914'])
        fig_trend.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#333333')
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with c2:
        st.subheader("Genre Distribution")
        top_genres = df['primary_genre'].value_counts().head(10)
        fig_genre = px.pie(names=top_genres.index, values=top_genres.values, hole=0.4, title="Top 10 Genres", color_discrete_sequence=px.colors.qualitative.Bold)
        fig_genre.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#333333')
        st.plotly_chart(fig_genre, use_container_width=True)

elif page == "content Explorer":
    st.title("🔎 Content Explorer")
    
    # Search
    search_term = st.text_input("Search Titles, Actors, or Directors", "")
    
    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        genre_filter = st.multiselect("Filter by Genre", options=sorted(df['primary_genre'].unique()))
    with col_f2:
        year_filter = st.slider("Filter by Release Year", int(df['release_year'].min()), int(df['release_year'].max()), (2010, 2021))
        
    # Apply Filters
    filtered_df = df.copy()
    
    if search_term:
        mask = filtered_df['title'].str.contains(search_term, case=False, na=False)
        if 'director' in filtered_df.columns:
            mask = mask | filtered_df['director'].str.contains(search_term, case=False, na=False)
        if 'cast' in filtered_df.columns:
            mask = mask | filtered_df['cast'].str.contains(search_term, case=False, na=False)
        filtered_df = filtered_df[mask]
        
    if genre_filter:
        filtered_df = filtered_df[filtered_df['primary_genre'].isin(genre_filter)]
        
    filtered_df = filtered_df[(filtered_df['release_year'] >= year_filter[0]) & (filtered_df['release_year'] <= year_filter[1])]
    
    st.dataframe(filtered_df[['title', 'type', 'primary_genre', 'release_year', 'duration', 'rating']], use_container_width=True)


elif page == "AI Recommender":
    st.title("🤖 AI Recommendation Engine")
    st.subheader("Powered by Hybrid Embedding Models")
    
    selected_title = st.selectbox("Select a Title you like:", options=sorted(df['title'].unique()))
    
    if st.button("Generate Recommendations", help="Click to call the ML API"):
        with st.spinner("Analyzing semantic similarity..."):
            try:
                # Call local API
                response = requests.post(f"{API_URL}/recommend", json={"title": selected_title})
                
                if response.status_code == 200:
                    data = response.json()
                    recs = data['recommendations']
                    
                    st.success(f"Top 10 matches for '{selected_title}'")
                    
                    # Display as neat cards
                    cols = st.columns(3)
                    for i, rec in enumerate(recs):
                        with cols[i % 3]:
                            with st.container():
                                st.markdown(f"### {rec['title']}")
                                st.markdown(f"**{rec['primary_genre']}** • {rec['release_year']}")
                                st.caption(rec.get('description', 'No description'))
                                st.markdown("---")
                else:
                    st.error("API Error: Model might be reloading or title not indexed.")
            except Exception as e:
                st.error(f"Connection Failed: Ensure 'run_api.py' is running. Error: {e}")

elif page == "Content Classifier":
    st.title("🎥 Content Type Classifier")
    st.markdown("Use our Machine Learning model to **predict if a title is a Movie or TV Show** based on its metadata.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Features")
        duration = st.number_input("Duration (Minutes)", 0, 300, 90)
        season_count = st.number_input("Season Count", 0, 20, 0)
        
    if st.button("Predict Content Type"):
        # Logic: If duration > 0 -> Movie. If seasons > 0 -> TV Show.
        # This mirrors our 100% acc model logic
        with st.spinner("Computing..."):
            try:
                # Call API
                # Sending user inputs for classification
                payload = {
                    "duration_minutes": duration,
                    "season_count": season_count,
                    "release_year": 2021,
                    "year_added": 2021
                }
                res = requests.post(f"{API_URL}/predict/type", json=payload)
                if res.status_code == 200:
                    result = res.json()
                    pred = result['prediction']
                    conf = result['confidence'] * 100
                    
                    st.metric("Prediction", pred, f"{conf}% Confidence")
                    if pred == "Movie":
                        st.balloons()
                else:
                    st.error("API Error")
            except:
                st.error("Connection Failed. Is the API running?")
