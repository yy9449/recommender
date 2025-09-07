# main.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import io
import os

# --- Correctly Integrated Algorithm Imports ---
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced
from hybrid import smart_hybrid_recommendation

warnings.filterwarnings('ignore')

# =========================
# Streamlit Page Configuration
# =========================
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")
st.title("🎬 Movie Recommendation System")

# =========================
# Data Loading and Preparation
# =========================
@st.cache_data
def load_and_prepare_data():
    """Loads and merges data from GitHub with a local fallback, handling column name changes."""
    github_base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    
    try:
        movies_df = pd.read_csv(github_base_url + "movies.csv")
        imdb_df = pd.read_csv(github_base_url + "imdb_top_1000.csv")
        user_ratings_df = pd.read_csv(github_base_url + "user_movie_rating.csv")
        st.session_state['data_source'] = "GitHub"
    except Exception:
        st.warning("⚠️ GitHub loading failed. Attempting to load local files...")
        try:
            movies_df = pd.read_csv("movies.csv")
            imdb_df = pd.read_csv("imdb_top_1000.csv")
            user_ratings_df = pd.read_csv("user_movie_rating.csv")
            st.session_state['data_source'] = "Local Files"
        except FileNotFoundError:
            st.error("❌ CRITICAL ERROR: Required CSV files not found locally. App cannot run.")
            return None, None
            
    # --- The CRITICAL Merge Step ---
    # We merge imdb (left) with movies (right). Pandas will add '_x' and '_y' to overlapping columns.
    # Our algorithms have been fixed to use the '_x' columns from the richer imdb_df.
    merged_df = pd.merge(imdb_df, movies_df, on="Series_Title", how="left", suffixes=('_x', '_y'))

    # Ensure a consistent Movie_ID column
    if 'Movie_ID' not in merged_df.columns:
        merged_df['Movie_ID'] = merged_df.index
    
    return merged_df.drop_duplicates(subset=['Series_Title']).reset_index(drop=True), user_ratings_df

# =========================
# Main Application
# =========================
def main():
    merged_df, user_ratings_df = load_and_prepare_data()

    if merged_df is None:
        st.stop()

    st.success(f"🎉 Datasets loaded from {st.session_state.get('data_source', 'source')}. Ready to recommend!")

    # --- Sidebar for User Input ---
    with st.sidebar:
        st.header("🎯 Recommendation Settings")

        movie_titles = sorted(merged_df['Series_Title'].dropna().unique())
        movie_title = st.selectbox("1. Select a Movie:", [""] + movie_titles)
        
        # Use the correct 'Genre_x' column
        genre_col = 'Genre_x'
        all_genres = sorted(merged_df[genre_col].dropna().str.split(', ').explode().unique())
        genre_input = st.selectbox("2. Filter by Genre (Optional):", [""] + all_genres)
        
        algorithm = st.selectbox("3. Choose Algorithm:", ["Hybrid", "Content-Based", "Collaborative"])
        top_n = st.slider("4. Number of Recommendations:", 5, 20, 10)
        
        generate_button = st.button("🚀 Generate Recommendations", type="primary")

    # --- Recommendation and Display Logic ---
    if generate_button:
        if not movie_title:
            st.error("❌ Please select a movie from the dropdown to get recommendations.")
            return

        with st.spinner("Finding movies you'll love..."):
            results = pd.DataFrame()

            if algorithm == "Content-Based":
                results = content_based_filtering_enhanced(merged_df, movie_title, top_n)
            elif algorithm == "Collaborative":
                results = collaborative_filtering_enhanced(merged_df, user_ratings_df, movie_title, top_n)
            elif algorithm == "Hybrid":
                results = smart_hybrid_recommendation(merged_df, user_ratings_df, movie_title, top_n)

            if not results.empty:
                if genre_input:
                    results = results[results[genre_col].str.contains(genre_input, na=False, case=False)]

                if not results.empty:
                    st.subheader(f"Top {len(results)} Recommendations for '{movie_title}'")
                    for i in range(0, len(results), 5):
                        cols = st.columns(5)
                        for j, (_, row) in enumerate(results.iloc[i:i+5].iterrows()):
                            with cols[j]:
                                if pd.notna(row.get('Poster_Link')):
                                    st.image(row['Poster_Link'], use_column_width=True)
                                st.markdown(f"**{row['Series_Title']}**")
                                st.write(f"⭐ {row.get('IMDB_Rating', 'N/A')}")
                else:
                    st.warning(f"No recommendations for '{movie_title}' matched the genre '{genre_input}'.")
            else:
                st.error(f"Could not generate recommendations for '{movie_title}' with the {algorithm} algorithm.")

if __name__ == "__main__":
    main()
