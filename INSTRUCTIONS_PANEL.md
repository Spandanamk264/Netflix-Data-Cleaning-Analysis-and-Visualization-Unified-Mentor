
# 🚀 Launching the Advanced Netflix Dashboard

## Overview
This project now features a high-end **Panel** dashboard (`src/dashboard/panel_app.py`) replacing the standard Streamlit app. It includes:
- **Interactive Visualizations** (Holoviews/Bokeh)
- **Machine Learning Recommender** (TF-IDF Content Filtering)
- **Professional Dark Theme** (FastListTemplate)

## 🛠️ Setup & Run

1. **Install Dependencies** (if you haven't yet):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   panel serve src/dashboard/panel_app.py --show --autoreload
   ```

## 📊 Features
- **Sidebar**: Filter by Year, Type, and Genres.
- **KPIs**: Live metrics update based on filters.
- **AI Recommender**: Type a movie name (e.g., "Inception") in the input box to get similar titles.
