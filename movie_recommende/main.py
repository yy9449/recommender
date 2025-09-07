# main.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import requests
import io
import os

# --- Corrected Algorithm Imports ---
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced
from hybrid import smart_hybrid_recommendation

warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide")

# =========================
# Data Loading and Preparation
# =========================
@st.cache_data
def load_data():
    """Loads and merges data from GitHub with a local fallback."""
    github_base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    
    try:
        movies_df = pd.read_csv(github_base_url + "movies.csv")
        imdb_df = pd.read_csv(github_base_url + "imdb_top_1000.csv")
        user_ratings_df = pd.read_csv(github_base_url + "user_movie_rating.csv")
        st.session_state['data_source'] = "GitHub"
    except Exception as e:
        st.warning("⚠️ Could not load data from GitHub, attempting to load local files...")
        try:
            movies_df = pd.read_csv("movies.csv")
            imdb_df = pd.read_csv("imdb_top_1000.csv")
            user_ratings_df = pd.read_csv("user_movie_rating.csv")
            st.session_state['data_source'] = "Local"
        except FileNotFoundError:
            st.error("❌ CRITICAL: Could not find required CSV files locally. Please ensure 'movies.csv', 'imdb_top_1000.csv', and 'user_movie_rating.csv' are in the same directory.")
            return None, None
            
    # --- Perform the merge and cleaning ---
    imdb_df.rename(columns={'Series_Title': 'Title'}, inplace=True)
    movies_df.rename(columns={'Series_Title': 'Title'}, inplace=True)
    # Use an outer merge to keep all movies, then clean up
    merged_df = pd.merge(imdb_df, movies_df, on="Title", how="outer", suffixes=('_x', '_y'))

    # Create a unique Movie_ID
    if 'Movie_ID' not in merged_df.columns:
        if 'Movie_ID_y' in merged_df.columns:
            merged_df['Movie_ID'] = merged_df['Movie_ID_y'].fillna(0).astype(int)
        else:
            merged_df['Movie_ID'] = range(len(merged_df))
    
    # Rename title back for consistency
    merged_df.rename(columns={'Title': 'Series_Title'}, inplace=True)
    
    return merged_df.drop_duplicates(subset=['Series_Title']), user_ratings_df

# =========================
# Main Application
# =========================
def main():
    st.title("🎬 Movie Recommendation System")

    merged_df, user_ratings_df = load_data()

    if merged_df is None:
        st.stop()

    st.success(f"🎉 Datasets loaded successfully from {st.session_state.get('data_source', 'source')}!")

    # --- Sidebar for User Input ---
    with st.sidebar:
        st.header("🎯 Recommendation Settings")

        # Movie and Genre Selection
        movie_titles = sorted(merged_df['Series_Title'].dropna().unique())
        movie_title = st.selectbox("Select a Movie (for Content, Hybrid, Collaborative):", [""] + movie_titles)
        
        genre_col = 'Genre_x' # Consistently use the genre from imdb_top_1000
        all_genres = sorted(merged_df[genre_col].dropna().str.split(', ').explode().unique())
        genre_input = st.selectbox("Filter by Genre (Optional):", [""] + all_genres)
        
        # Algorithm and Top N Selection
        algorithm = st.selectbox("🔬 Choose Algorithm:", ["Hybrid", "Content-Based", "Collaborative"])
        top_n = st.slider("📊 Number of Recommendations:", 5, 20, 10)

        generate_button = st.button("🚀 Generate Recommendations", type="primary")

    # --- Recommendation Logic ---
    if generate_button:
        if not movie_title:
            st.error("❌ Please select a movie to get recommendations.")
            return

        with st.spinner("Finding movies you'll love..."):
            results = pd.DataFrame()

            if algorithm == "Content-Based":
                results = content_based_filtering_enhanced(merged_df, movie_title, top_n)
            elif algorithm == "Collaborative":
                results = collaborative_filtering_enhanced(merged_df, user_ratings_df, movie_title, top_n)
            elif algorithm == "Hybrid":
                results = smart_hybrid_recommendation(merged_df, user_ratings_df, movie_title, top_n)

            # --- Filter and Display Results ---
            if not results.empty:
                if genre_input:
                    results = results[results[genre_col].str.contains(genre_input, na=False, case=False)]

                if not results.empty:
                    st.subheader(f"Recommendations for '{movie_title}'")
                    # Display posters
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
                st.error(f"Could not generate recommendations for '{movie_title}' with the {algorithm} algorithm. The movie might not have enough data.")

if __name__ == "__main__":
    main()
