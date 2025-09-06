# main.py

import streamlit as st
import pandas as pd
import warnings
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced
from hybrid import smart_hybrid_recommendation

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

@st.cache_data
def load_data():
    """Loads and cleans data needed for the app."""
    try:
        movies_df = pd.read_csv('movies.csv')
        imdb_df = pd.read_csv('imdb_top_1000.csv')
        user_ratings_df = pd.read_csv('user_movie_rating.csv')
        
        if 'Movie_ID' not in movies_df.columns:
            movies_df['Movie_ID'] = range(len(movies_df))
            
        merged_df = pd.merge(movies_df, imdb_df, on='Series_Title', how='left')

        # Robustly handle conflicting columns after the merge
        for col in ['Genre', 'Overview', 'Director']:
            col_x, col_y = f'{col}_x', f'{col}_y'
            if col_y in merged_df.columns and col_x in merged_df.columns:
                merged_df[col] = merged_df[col_y].fillna(merged_df[col_x])
        
        return merged_df, user_ratings_df
    except FileNotFoundError as e:
        st.error(f"Error: A data file was not found. Please make sure all CSV files are in your repository. Details: {e}")
        return None, None

# --- Main App Logic ---
merged_df, user_ratings_df = load_data()

if merged_df is not None:
    st.sidebar.header("Select Your Preferences")
    
    movie_list = sorted(merged_df['Series_Title'].dropna().unique())
    selected_movie = st.sidebar.selectbox("Choose a movie you like:", movie_list)

    algorithm = st.sidebar.radio(
        "Choose a recommendation algorithm:",
        ('Hybrid', 'Content-Based', 'Collaborative')
    )

    if st.sidebar.button("Get Recommendations"):
        with st.spinner("Finding recommendations..."):
            if algorithm == 'Content-Based':
                recommendations = content_based_filtering_enhanced(merged_df, selected_movie)
            elif algorithm == 'Collaborative':
                recommendations = collaborative_filtering_enhanced(merged_df, user_ratings_df, selected_movie)
            else: # Default to Hybrid
                recommendations = smart_hybrid_recommendation(merged_df, user_ratings_df, selected_movie)

            st.subheader(f"Recommendations based on '{selected_movie}' using {algorithm} Filtering:")
            
            if recommendations is not None and not recommendations.empty:
                display_cols = ['Series_Title', 'Genre', 'IMDB_Rating']
                # Ensure all display columns exist before trying to show them
                existing_display_cols = [col for col in display_cols if col in recommendations.columns]
                display_df = recommendations[existing_display_cols].reset_index(drop=True)
                st.table(display_df)
            else:
                st.warning("Could not find recommendations for this movie/algorithm combination.")