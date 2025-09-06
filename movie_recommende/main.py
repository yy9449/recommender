# main.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import io

# --- Attempt to Import Recommendation Functions ---
# This structure ensures the app won't crash if a file or function is missing/incorrect.
try:
    from content_based import content_based_filtering_enhanced
    CONTENT_AVAILABLE = True
except ImportError:
    CONTENT_AVAILABLE = False
    st.error("Could not import 'content_based.py'. The Content-Based model will not be available.")

try:
    from collaborative import collaborative_filtering_enhanced
    COLLABORATIVE_AVAILABLE = True
except ImportError:
    COLLABORATIVE_AVAILABLE = False
    st.error("Could not import 'collaborative.py'. The Collaborative model will not be available.")

try:
    # IMPORTANT: Your hybrid.py has 'svd_hybrid_recommendation', not 'smart_hybrid_recommendation'
    from hybrid import svd_hybrid_recommendation
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    st.error("Could not import 'hybrid.py'. The Hybrid model will not be available.")


warnings.filterwarnings('ignore')

# =========================
# Streamlit Configuration
# =========================
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🎬 Movie Recommendation System")
st.markdown("---")

# =========================
# GitHub CSV Loading (Corrected)
# =========================
@st.cache_data
def load_all_data():
    """Loads and prepares all data from the GitHub repository."""
    base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    try:
        movies_df = pd.read_csv(base_url + "movies.csv")
        imdb_df = pd.read_csv(base_url + "imdb_top_1000.csv")
        user_ratings_df = pd.read_csv(base_url + "user_movie_rating.csv")

        if 'Movie_ID' not in movies_df.columns:
            movies_df['Movie_ID'] = range(len(movies_df))
        
        merged_df = pd.merge(movies_df, imdb_df, on='Series_Title', how='left')

        # Robustly handle conflicting columns after the merge
        for col in ['Genre', 'Overview', 'Director']:
            col_x, col_y = f'{col}_x', f'{col}_y'
            if col_y in merged_df.columns and col_x in merged_df.columns:
                merged_df[col] = merged_df[col_y].fillna(merged_df[col_x])
        
        # Store ratings in session state for other modules if they need it
        st.session_state['user_ratings_df'] = user_ratings_df
        
        return merged_df, user_ratings_df

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to load data from GitHub. Please ensure the repository is public and URLs are correct. Error: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ An error occurred during data loading or processing: {e}")
        return None, None

# =========================
# Main App Logic
# =========================
merged_df, user_ratings_df = load_all_data()

if merged_df is not None:
    st.sidebar.header("✨ Your Recommendation Engine")
    
    # Create the list of available algorithms
    available_algorithms = []
    if HYBRID_AVAILABLE: available_algorithms.append('Hybrid')
    if CONTENT_AVAILABLE: available_algorithms.append('Content-Based')
    if COLLABORATIVE_AVAILABLE: available_algorithms.append('Collaborative')
    
    if not available_algorithms:
        st.error("FATAL: No recommendation models could be loaded. Please check your Python files.")
    else:
        # User Inputs
        movie_list = sorted(merged_df['Series_Title'].dropna().unique())
        movie_title = st.sidebar.selectbox("1. Choose a movie you like:", movie_list)
        
        genre_list = sorted(merged_df['Genre'].dropna().unique())
        genre_input = st.sidebar.selectbox("2. (Optional) Select a preferred genre:", ["Any"] + genre_list)

        algorithm = st.sidebar.radio("3. Choose an algorithm:", available_algorithms)

        if st.sidebar.button("🚀 Get Recommendations"):
            with st.spinner("Brewing your movie list..."):
                results = None
                genre_for_filtering = genre_input if genre_input != "Any" else None

                if algorithm == 'Content-Based':
                    results = content_based_filtering_enhanced(merged_df, target_title=movie_title, genre=genre_for_filtering)
                elif algorithm == 'Collaborative':
                    results = collaborative_filtering_enhanced(merged_df, target_movie=movie_title)
                elif algorithm == 'Hybrid':
                    # Calling the correct function name from your hybrid.py
                    results = svd_hybrid_recommendation(merged_df, target_movie=movie_title, genre=genre_for_filtering)

                # --- Display Results ---
                st.subheader(f"Recommendations for '{movie_title}'")
                st.write(f"Using **{algorithm}** model")

                if results is not None and not results.empty:
                    # Ensure display columns exist
                    display_cols = ['Series_Title', 'Genre', 'IMDB_Rating']
                    cols_to_display = [col for col in display_cols if col in results.columns]
                    st.table(results[cols_to_display].head(8).reset_index(drop=True))
                else:
                    st.error("❌ No recommendations found. Please try a different movie or algorithm.")