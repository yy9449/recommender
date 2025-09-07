# main.py

import streamlit as st
import pandas as pd
import warnings
import requests
import io

# --- A fully synchronized set of imports ---
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
# Data Loading and Preparation (Completely Rewritten for Consistency)
# =========================
@st.cache_data
def load_and_prepare_data():
    """
    Loads, merges, and cleans data from GitHub with a local fallback.
    This new version creates a single, clean dataframe with consistent column names.
    """
    github_base_url = "https://raw.githubusercontent.com/yy9449/recommender/main/movie_recommende/"
    
    try:
        imdb_df = pd.read_csv(github_base_url + "imdb_top_1000.csv")
        movies_df = pd.read_csv(github_base_url + "movies.csv")
        user_ratings_df = pd.read_csv(github_base_url + "user_movie_rating.csv")
        st.session_state['data_source'] = "GitHub"
    except Exception:
        st.warning("⚠️ GitHub loading failed. Attempting to load local files...")
        try:
            imdb_df = pd.read_csv("imdb_top_1000.csv")
            movies_df = pd.read_csv("movies.csv")
            user_ratings_df = pd.read_csv("user_movie_rating.csv")
            st.session_state['data_source'] = "Local Files"
        except FileNotFoundError:
            st.error("❌ CRITICAL ERROR: Required CSV files not found. The app cannot run.")
            return None, None
            
    # --- The Definitive Merge and Clean Strategy ---
    # Merge the dataframes, creating '_x' and '_y' suffixes for overlapping columns.
    merged_df = pd.merge(imdb_df, movies_df, on="Series_Title", how="left", suffixes=('_imdb', '_movies'))

    # Coalesce overlapping columns: prefer the richer '_imdb' data, fall back to '_movies'.
    # This creates ONE clean column and removes the confusing suffixes.
    for col in ['Genre', 'Overview', 'Director']:
        imdb_col = f'{col}_imdb'
        movies_col = f'{col}_movies'
        if imdb_col in merged_df and movies_col in merged_df:
            merged_df[col] = merged_df[imdb_col].fillna(merged_df[movies_col])
            merged_df.drop(columns=[imdb_col, movies_col], inplace=True)

    # Ensure a consistent Movie_ID column exists for linking
    if 'Movie_ID' not in merged_df.columns:
        merged_df['Movie_ID'] = merged_df.index
    merged_df['Movie_ID'] = merged_df['Movie_ID'].fillna(pd.Series(merged_df.index)).astype(int)

    return merged_df.drop_duplicates(subset=['Series_Title']).reset_index(drop=True), user_ratings_df

# =========================
# Main Application
# =========================
def main():
    merged_df, user_ratings_df = load_and_prepare_data()

    if merged_df is None:
        st.stop()

    st.success(f"🎉 Datasets loaded and prepared from {st.session_state.get('data_source', 'a source')}. Ready to recommend!")

    with st.sidebar:
        st.header("🎯 Recommendation Settings")
        movie_titles = sorted(merged_df['Series_Title'].dropna().unique())
        movie_title = st.selectbox("1. Select a Movie:", [""] + movie_titles)
        
        all_genres = sorted(merged_df['Genre'].dropna().str.split(', ').explode().unique())
        genre_input = st.selectbox("2. Filter by Genre (Optional):", [""] + all_genres)
        
        algorithm = st.selectbox("3. Choose Algorithm:", ["Hybrid", "Content-Based", "Collaborative"])
        top_n = st.slider("4. Number of Recommendations:", 5, 20, 10)
        generate_button = st.button("🚀 Generate Recommendations", type="primary")

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

            if results is not None and not results.empty:
                if genre_input:
                    results = results[results['Genre'].str.contains(genre_input, na=False, case=False)]

                if not results.empty:
                    st.subheader(f"Top {len(results)} Recommendations for '{movie_title}'")
                    for i in range(0, len(results), 5):
                        cols = st.columns(5)
                        for j, (_, row) in enumerate(results.iloc[i:i+5].iterrows()):
                            with cols[j]:
                                st.image(row.get('Poster_Link', ''), use_column_width=True, caption=row['Series_Title'])
                                st.write(f"⭐ {row.get('IMDB_Rating', 'N/A')}")
                else:
                    st.warning(f"No recommendations for '{movie_title}' matched the genre '{genre_input}'.")
            else:
                st.error(f"Could not generate recommendations for '{movie_title}' with the {algorithm} algorithm.")

if __name__ == "__main__":
    main()
