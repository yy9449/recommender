# main.py

import streamlit as st
import pandas as pd
import warnings
import requests
import io
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced
from hybrid import smart_hybrid_recommendation

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads, merges, and cleans data from GitHub, returning two dataframes."""
    try:
        movies_url = "https://raw.githubusercontent.com/carine-l/recommender-main/main/movie_recommende/movies.csv"
        imdb_url = "https://raw.githubusercontent.com/carine-l/recommender-main/main/movie_recommende/imdb_top_1000.csv"
        ratings_url = "https://raw.githubusercontent.com/carine-l/recommender-main/main/movie_recommende/user_movie_rating.csv"

        movies_df = pd.read_csv(movies_url)
        imdb_df = pd.read_csv(imdb_url)
        user_ratings_df = pd.read_csv(ratings_url)
        
        # Ensure Movie_ID exists
        if 'Movie_ID' not in movies_df.columns:
            movies_df['Movie_ID'] = range(len(movies_df))
            
        # Merge dataframes
        merged_df = pd.merge(movies_df, imdb_df, on='Series_Title', how='left')

        # Robustly handle conflicting columns from the merge
        for col in ['Genre', 'Overview', 'Director']:
            col_x, col_y = f'{col}_x', f'{col}_y'
            if col_y in merged_df.columns and col_x in merged_df.columns:
                merged_df[col] = merged_df[col_y].fillna(merged_df[col_x])
        
        return merged_df, user_ratings_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# --- Main App Logic ---
merged_df, user_ratings_df = load_data()

if merged_df is not None:
    # --- Sidebar for User Input ---
    st.sidebar.header("Select Your Preferences")
    
    # Movie selection dropdown
    movie_list = sorted(merged_df['Series_Title'].unique())
    selected_movie = st.sidebar.selectbox("Choose a movie you like:", movie_list)

    # Algorithm selection
    algorithm = st.sidebar.radio(
        "Choose a recommendation algorithm:",
        ('Hybrid', 'Content-Based', 'Collaborative')
    )

    if st.sidebar.button("Get Recommendations"):
        # --- Recommendation Logic ---
        with st.spinner("Finding recommendations..."):
            if algorithm == 'Content-Based':
                recommendations = content_based_filtering_enhanced(merged_df, selected_movie)
            elif algorithm == 'Collaborative':
                recommendations = collaborative_filtering_enhanced(merged_df, user_ratings_df, selected_movie)
            else: # Default to Hybrid
                recommendations = smart_hybrid_recommendation(merged_df, user_ratings_df, selected_movie)

            # --- Display Results ---
            st.subheader(f"Recommendations based on '{selected_movie}' using {algorithm} Filtering:")
            
            if recommendations is not None and not recommendations.empty:
                # Display results in a clean table
                display_cols = ['Series_Title', 'Genre', 'IMDB_Rating']
                display_df = recommendations[display_cols].reset_index(drop=True)
                st.table(display_df)
            else:
                st.warning("Could not find recommendations for this movie/algorithm combination.")
else:
    st.error("Application cannot start. Please check the data source.")